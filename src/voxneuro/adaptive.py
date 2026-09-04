"""Rank-normalized, dimension-adaptive multi-view Grassmannian classifier (VoxNeuro-AS).

This module extends the core method in :mod:`voxneuro.method` with two
fold-local components that were developed after the original submission:

1. **Gaussian rank normalization** of the recording-level features. Each
   feature is mapped through the empirical distribution function of the
   *training-fold recordings* and then through the standard normal quantile
   function, replacing z-standardization. The mapping is monotone and
   scale-free, so heavy-tailed acoustic descriptors no longer dominate the
   Euclidean summary distances.

2. **A supervised-subspace stage that is activated only in the
   high-dimensional regime** (feature count larger than the number of
   training recordings). A partial-least-squares (PLS) projection fitted on
   the training-fold recordings with class-balanced targets maps every
   recording to ``q`` coordinates; the subject views (rank-``r`` subspace and
   mean-dispersion summary), the median-heuristic Gaussian kernels and the
   fused kernel are then built in that space exactly as in the original
   method. One class-weighted SVM is trained per subspace dimension in
   ``subspace_dims``; their decision functions are divided by the standard
   deviation of the training decision values and averaged (a selection-free
   ensemble). No synthetic subjects are generated in this branch: the
   class-weighted SVM already balances the classes and a G-SMOTE ablation
   reduced balanced accuracy in the compressed space.

When the gate is inactive (``p <= n_train_recordings``) the model reduces to
the original fused Grassmann-Euclidean SVM with rank normalization and the
original G-SMOTE imbalance gate.

Everything that is fitted (normalizer, projection, subspaces, bandwidths,
classifiers, decision scales) uses the current outer-training fold only, so
the evaluation remains leakage-safe at the subject level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC, SVC

from .method import (
    EvaluationResult,
    _classification_record,
    _fit_fused_model,
    _g_smote,
    _prediction_rows,
    _standard_error,
    _subject_rows,
    _validate_frame,
    build_subject_views,
    chordal_squared,
    euclidean_squared,
)

DEFAULT_SUBSPACE_DIMS: tuple[int, ...] = (8, 16, 32)
RANK_KNOTS = 300


# --------------------------------------------------------------------------- #
# Fold-local preprocessing
# --------------------------------------------------------------------------- #
def fit_normalizer(X_train: np.ndarray, kind: str = "rank"):
    """Fit the recording-level normalizer on training-fold recordings only.

    ``kind="rank"`` is Gaussian rank normalization (empirical CDF followed by
    the standard normal quantile function, ``RANK_KNOTS`` quantile knots);
    ``kind="standard"`` is the original z-standardization.
    """
    if kind == "standard":
        return StandardScaler().fit(X_train)
    if kind == "rank":
        n_quantiles = int(min(RANK_KNOTS, X_train.shape[0]))
        return QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=10**9,  # never subsample: the mapping is a deterministic function of the training fold
            random_state=0,
        ).fit(X_train)
    raise ValueError("normalization must be 'rank' or 'standard'.")


def fit_supervised_projection(
    X_train_rows: np.ndarray, y_train_rows: np.ndarray, n_components: int
) -> PLSRegression:
    """PLS projection fitted on training recordings with class-balanced targets.

    Each recording carries its subject's label. The regression target is
    ``+1/n_1`` for class-1 recordings and ``-1/n_0`` for class-0 recordings, so
    both classes receive equal total weight in the covariance that PLS
    maximizes.
    """
    y = np.asarray(y_train_rows, dtype=int)
    n1 = max(int(np.sum(y == 1)), 1)
    n0 = max(int(np.sum(y == 0)), 1)
    target = np.where(y == 1, 1.0 / n1, -1.0 / n0)
    return PLSRegression(n_components=int(n_components), scale=False).fit(X_train_rows, target)


def subspace_gate_active(n_features: int, n_train_recordings: int, gate: str = "auto") -> bool:
    """Prespecified dimensionality gate for the supervised-subspace stage."""
    if gate == "on":
        return True
    if gate == "off":
        return False
    if gate == "auto":
        return int(n_features) > int(n_train_recordings)
    raise ValueError("gate must be 'auto', 'on' or 'off'.")


def concentration_statistics(Q: Sequence[np.ndarray], summaries: np.ndarray) -> dict[str, float]:
    """Coefficient of variation of pairwise subspace and summary distances (diagnostic)."""
    d_g = np.sqrt(chordal_squared(Q))
    d_e = np.sqrt(euclidean_squared(summaries))
    iu = np.triu_indices(len(Q), k=1)
    vg, ve = d_g[iu], d_e[iu]
    return {
        "chordal_mean": float(vg.mean()),
        "chordal_cv": float(vg.std() / vg.mean()),
        "euclidean_cv": float(ve.std() / ve.mean()),
    }


# --------------------------------------------------------------------------- #
# The adaptive model
# --------------------------------------------------------------------------- #
@dataclass
class _SubspaceMember:
    projection: PLSRegression
    model: object  # _FusedModel
    scale: float


@dataclass
class AdaptiveModel:
    """Fitted rank-normalized, dimension-adaptive fused model for one training fold."""

    normalizer: object
    gate_active: bool
    rank: int
    allow_rank_deficient: bool
    members: list[_SubspaceMember]
    full_model: object | None
    synthetic_subjects: int

    def _views(self, X_rows: np.ndarray, row_ids: np.ndarray, row_labels: np.ndarray, subject_ids: Sequence[str]):
        return build_subject_views(X_rows, row_ids, row_labels, subject_ids, self.rank, self.allow_rank_deficient)

    def decision_function(
        self, X: np.ndarray, row_ids: np.ndarray, row_labels: np.ndarray, subject_ids: Sequence[str]
    ) -> np.ndarray:
        rows = np.flatnonzero(np.isin(row_ids, list(subject_ids)))
        Xn = np.empty_like(X, dtype=float)
        Xn[rows] = self.normalizer.transform(X[rows])
        if not self.gate_active:
            Q, s, _ = self._views(Xn, row_ids, row_labels, subject_ids)
            return self.full_model.decision_function(Q, s)
        scores = []
        for member in self.members:
            Z = np.empty((X.shape[0], member.projection.n_components), dtype=float)
            Z[rows] = member.projection.transform(Xn[rows])
            Q, s, _ = self._views(Z, row_ids, row_labels, subject_ids)
            scores.append(member.model.decision_function(Q, s) / member.scale)
        return np.mean(scores, axis=0)


def fit_adaptive_model(
    X: np.ndarray,
    row_ids: np.ndarray,
    row_labels: np.ndarray,
    train_ids: Sequence[str],
    *,
    rank: int = 3,
    weight: float = 0.5,
    C: float = 1.0,
    normalization: str = "rank",
    subspace_dims: Sequence[int] = DEFAULT_SUBSPACE_DIMS,
    gate: str = "auto",
    g_smote_neighbors: int = 5,
    random_state: int = 42,
    allow_rank_deficient: bool = False,
) -> AdaptiveModel:
    """Fit the adaptive model on the training subjects ``train_ids`` only."""
    train_rows = np.flatnonzero(np.isin(row_ids, list(train_ids)))
    normalizer = fit_normalizer(X[train_rows], normalization)
    Xn = np.empty_like(X, dtype=float)
    Xn[train_rows] = normalizer.transform(X[train_rows])
    active = subspace_gate_active(X.shape[1], train_rows.size, gate)

    if not active:
        Q_train, s_train, y_train = build_subject_views(
            Xn, row_ids, row_labels, train_ids, rank, allow_rank_deficient
        )
        Q_aug, s_aug, y_aug, synthetic = _g_smote(
            Q_train, s_train, y_train, random_state=random_state, neighbors=g_smote_neighbors
        )
        full = _fit_fused_model(Q_aug, s_aug, y_aug, weight=weight, C=C)
        return AdaptiveModel(normalizer, False, rank, allow_rank_deficient, [], full, synthetic)

    members: list[_SubspaceMember] = []
    y_rows = np.asarray(row_labels, dtype=int)[train_rows]
    for q in subspace_dims:
        if q < rank:
            raise ValueError("Every subspace dimension must be at least the Grassmann rank.")
        if q > X.shape[1]:
            raise ValueError("A subspace dimension cannot exceed the number of features.")
        projection = fit_supervised_projection(Xn[train_rows], y_rows, q)
        Z = np.empty((X.shape[0], int(q)), dtype=float)
        Z[train_rows] = projection.transform(Xn[train_rows])
        Q_train, s_train, y_train = build_subject_views(
            Z, row_ids, row_labels, train_ids, rank, allow_rank_deficient
        )
        model = _fit_fused_model(Q_train, s_train, y_train, weight=weight, C=C)
        scale = float(np.std(model.decision_function(Q_train, s_train))) + 1e-12
        members.append(_SubspaceMember(projection, model, scale))
    return AdaptiveModel(normalizer, True, rank, allow_rank_deficient, members, None, 0)


# --------------------------------------------------------------------------- #
# Subject-level evaluation with comparators
# --------------------------------------------------------------------------- #
def evaluate_adaptive(
    frame: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    drop_cols: Sequence[str] = (),
    rank: int = 3,
    n_splits: int = 5,
    weight: float = 0.5,
    C: float = 1.0,
    normalization: str = "rank",
    subspace_dims: Sequence[int] = DEFAULT_SUBSPACE_DIMS,
    gate: str = "auto",
    g_smote_neighbors: int = 5,
    random_state: int = 42,
    allow_rank_deficient: bool = False,
    include_members: bool = True,
) -> EvaluationResult:
    """Leakage-safe subject-level evaluation of the adaptive model and its comparators.

    Reported methods (all under the same normalization and identical folds):

    ``Adaptive_Subspace_Fusion``
        the proposed model (gate decided per training fold);
    ``Fused_Grassmann_GSMOTE``
        the original full-space fused model with the G-SMOTE gate;
    ``Euclidean_RBF_GSMOTE``
        the matched Euclidean ablation (fusion weight 0);
    ``Balanced_LogReg``, ``Balanced_LinearSVM``, ``SMOTE_LogReg``
        the linear comparators of the original protocol;
    ``Subspace_q<k>`` (optional)
        the individual ensemble members, only when the gate is active.
    """
    if rank < 1:
        raise ValueError("rank must be positive.")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1].")

    df, feature_cols = _validate_frame(frame, id_col, label_col, drop_cols)
    X = df[feature_cols].to_numpy(dtype=float)
    row_ids = df[id_col].to_numpy(dtype=str)
    row_labels = df[label_col].to_numpy(dtype=int)
    subject_frame = df[[id_col, label_col]].drop_duplicates(id_col)
    subject_ids = subject_frame[id_col].to_numpy(dtype=str)
    subject_labels = subject_frame[label_col].to_numpy(dtype=int)
    row_map = _subject_rows(row_ids)

    class_counts = np.bincount(subject_labels)
    if np.min(class_counts[class_counts > 0]) < n_splits:
        raise ValueError("Each class must contain at least n_splits subjects.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    with threadpool_limits(limits=1):  # single-threaded BLAS: bit-for-bit reproducible decision values
        _run_folds(
            X, row_ids, row_labels, row_map, subject_ids, subject_labels, splitter, metrics, predictions,
            rank=rank, weight=weight, C=C, normalization=normalization, subspace_dims=subspace_dims, gate=gate,
            g_smote_neighbors=g_smote_neighbors, random_state=random_state,
            allow_rank_deficient=allow_rank_deficient, include_members=include_members,
        )

    fold_metrics = pd.DataFrame(metrics)
    summary_rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        summary_rows.append(
            {
                "method": method,
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_se": _standard_error(group["balanced_accuracy"]),
                "macro_f1_mean": float(group["macro_f1"].mean()),
                "macro_f1_se": _standard_error(group["macro_f1"]),
                "n_folds": int(len(group)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy_mean", "macro_f1_mean"], ascending=False
    )
    pooled = fold_metrics.groupby("method", sort=False)[["TN", "FP", "FN", "TP"]].sum().reset_index()
    return EvaluationResult(
        fold_metrics=fold_metrics,
        summary_metrics=summary.reset_index(drop=True),
        pooled_confusions=pooled,
        predictions=pd.DataFrame(predictions),
        feature_columns=tuple(feature_cols),
    )


def _run_folds(
    X, row_ids, row_labels, row_map, subject_ids, subject_labels, splitter, metrics, predictions, *,
    rank, weight, C, normalization, subspace_dims, gate, g_smote_neighbors, random_state,
    allow_rank_deficient, include_members,
) -> None:
    def record(fold: int, name: str, test_ids, y_test, scores, synthetic: int = 0) -> None:
        pred = (np.asarray(scores) > 0.0).astype(int)
        metrics.append(_classification_record(fold, name, y_test, pred, synthetic))
        predictions.extend(_prediction_rows(fold, name, test_ids, y_test, pred, scores))

    for fold, (train_index, test_index) in enumerate(splitter.split(subject_ids, subject_labels), start=1):
        train_ids = subject_ids[train_index]
        test_ids = subject_ids[test_index]
        train_rows = np.concatenate([row_map[sid] for sid in train_ids])
        test_rows = np.concatenate([row_map[sid] for sid in test_ids])
        fold_seed = random_state * 1000 + fold

        # Proposed model.
        adaptive = fit_adaptive_model(
            X, row_ids, row_labels, train_ids, rank=rank, weight=weight, C=C,
            normalization=normalization, subspace_dims=subspace_dims, gate=gate,
            g_smote_neighbors=g_smote_neighbors, random_state=fold_seed,
            allow_rank_deficient=allow_rank_deficient,
        )
        y_test = subject_labels[test_index]
        record(fold, "Adaptive_Subspace_Fusion", test_ids,
               y_test, adaptive.decision_function(X, row_ids, row_labels, test_ids), adaptive.synthetic_subjects)
        if include_members and adaptive.gate_active:
            Xn = np.empty_like(X, dtype=float)
            Xn[test_rows] = adaptive.normalizer.transform(X[test_rows])
            for q, member in zip(subspace_dims, adaptive.members):
                Z = np.empty((X.shape[0], int(q)), dtype=float)
                Z[test_rows] = member.projection.transform(Xn[test_rows])
                Q, s, _ = build_subject_views(Z, row_ids, row_labels, test_ids, rank, allow_rank_deficient)
                record(fold, f"Subspace_q{int(q)}", test_ids, y_test, member.model.decision_function(Q, s))

        # Comparators under the same normalization and folds.
        normalizer = adaptive.normalizer
        Xn = np.empty_like(X, dtype=float)
        Xn[train_rows] = normalizer.transform(X[train_rows])
        Xn[test_rows] = normalizer.transform(X[test_rows])
        Q_train, s_train, y_train = build_subject_views(Xn, row_ids, row_labels, train_ids, rank, allow_rank_deficient)
        Q_test, s_test, _ = build_subject_views(Xn, row_ids, row_labels, test_ids, rank, allow_rank_deficient)

        Q_aug, s_aug, y_aug, synthetic = _g_smote(
            Q_train, s_train, y_train, random_state=fold_seed, neighbors=g_smote_neighbors
        )
        fused = _fit_fused_model(Q_aug, s_aug, y_aug, weight=weight, C=C)
        record(fold, "Fused_Grassmann_GSMOTE", test_ids, y_test, fused.decision_function(Q_test, s_test), synthetic)
        euclid = _fit_fused_model(Q_aug, s_aug, y_aug, weight=0.0, C=C)
        record(fold, "Euclidean_RBF_GSMOTE", test_ids, y_test, euclid.decision_function(Q_test, s_test), synthetic)

        logistic = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=5000, random_state=random_state
        ).fit(s_train, y_train)
        record(fold, "Balanced_LogReg", test_ids, y_test, logistic.decision_function(s_test))
        linear = LinearSVC(class_weight="balanced", C=C, max_iter=20000, random_state=random_state).fit(s_train, y_train)
        record(fold, "Balanced_LinearSVM", test_ids, y_test, linear.decision_function(s_test))

        from imblearn.over_sampling import SMOTE  # local import keeps the optional dependency lazy

        minimum = int(np.bincount(y_train).min())
        if minimum >= 2 and np.bincount(y_train)[0] != np.bincount(y_train)[1]:
            smote = SMOTE(random_state=random_state + fold, k_neighbors=min(g_smote_neighbors, minimum - 1))
            s_res, y_res = smote.fit_resample(s_train, y_train)
            synthetic_euclidean = len(y_res) - len(y_train)
        else:
            s_res, y_res, synthetic_euclidean = s_train, y_train, 0
        smote_logistic = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=5000, random_state=random_state
        ).fit(s_res, y_res)
        record(fold, "SMOTE_LogReg", test_ids, y_test, smote_logistic.decision_function(s_test), synthetic_euclidean)
