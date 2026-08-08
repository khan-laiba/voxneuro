"""Core VoxNeuro subject-level learning method.

The module is intentionally self-contained: data validation, subject-level
representations, Grassmann geometry, G-SMOTE, comparators, cross-validation,
and result export are implemented here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


@dataclass(frozen=True)
class EvaluationResult:
    """Tables produced by subject-level evaluation."""

    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    pooled_confusions: pd.DataFrame
    predictions: pd.DataFrame
    feature_columns: tuple[str, ...]

    def save(self, output_dir: str | Path) -> None:
        """Write all result tables to ``output_dir``."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
        self.summary_metrics.to_csv(out / "summary_metrics.csv", index=False)
        self.pooled_confusions.to_csv(out / "pooled_confusions.csv", index=False)
        self.predictions.to_csv(out / "out_of_fold_predictions.csv", index=False)


@dataclass
class _FusedModel:
    weight: float
    C: float
    gamma_g: float
    gamma_e: float
    Q_train: list[np.ndarray]
    s_train: np.ndarray
    classifier: SVC

    def decision_function(self, Q_test: Sequence[np.ndarray], s_test: np.ndarray) -> np.ndarray:
        Kg = np.exp(-self.gamma_g * chordal_squared(Q_test, self.Q_train))
        Ke = np.exp(-self.gamma_e * euclidean_squared(s_test, self.s_train))
        K = self.weight * Kg + (1.0 - self.weight) * Ke
        return np.asarray(self.classifier.decision_function(K), dtype=float)

    def predict(self, Q_test: Sequence[np.ndarray], s_test: np.ndarray) -> np.ndarray:
        return (self.decision_function(Q_test, s_test) > 0.0).astype(int)


def _standard_error(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _validate_frame(
    frame: pd.DataFrame,
    id_col: str,
    label_col: str,
    drop_cols: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    required = {id_col, label_col, *drop_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = frame.copy()
    if df[id_col].isna().any() or df[label_col].isna().any():
        raise ValueError("Subject IDs and labels must not contain missing values.")

    label_counts = df.groupby(id_col)[label_col].nunique(dropna=False)
    inconsistent = label_counts[label_counts != 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each subject must have one constant label. Invalid subjects: "
            + ", ".join(map(str, inconsistent.index.tolist()))
        )

    labels = sorted(pd.unique(df[label_col]))
    if len(labels) != 2:
        raise ValueError(f"Exactly two classes are required; found {labels}.")
    label_map = {labels[0]: 0, labels[1]: 1}
    df[label_col] = df[label_col].map(label_map).astype(int)
    df[id_col] = df[id_col].astype(str)

    feature_cols = [c for c in df.columns if c not in {id_col, label_col, *drop_cols}]
    if not feature_cols:
        raise ValueError("No feature columns remain after exclusions.")
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Feature columns contain missing/non-numeric values: {bad[:10]}")
    df[feature_cols] = numeric.astype(float)
    return df, feature_cols


def _subject_rows(ids: np.ndarray) -> dict[str, np.ndarray]:
    mapping: dict[str, list[int]] = defaultdict(list)
    for index, sid in enumerate(ids):
        mapping[str(sid)].append(index)
    return {sid: np.asarray(rows, dtype=int) for sid, rows in mapping.items()}


def _numerical_rank(singular_values: np.ndarray, rows: int, cols: int) -> int:
    if singular_values.size == 0:
        return 0
    tolerance = max(rows, cols) * np.finfo(float).eps * singular_values[0]
    return int(np.sum(singular_values > tolerance))


def build_subject_views(
    X_scaled: np.ndarray,
    row_ids: np.ndarray,
    row_labels: np.ndarray,
    subject_ids: Sequence[str],
    rank: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build Grassmann and Euclidean views for complete subjects.

    The function validates that every subject matrix has numerical rank at
    least ``rank``. This keeps all returned subspaces on one fixed
    Grassmann manifold ``Gr(rank, d)``.
    """
    Q_list: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    labels: list[int] = []

    for sid in subject_ids:
        idx = np.flatnonzero(row_ids == sid)
        if idx.size < rank:
            raise ValueError(
                f"Subject {sid!r} has {idx.size} recordings, fewer than requested rank {rank}."
            )
        Xi = np.asarray(X_scaled[idx], dtype=float)  # recordings x features
        mean = Xi.mean(axis=0)
        std = Xi.std(axis=0, ddof=0)
        M = Xi.T  # features x recordings
        U, singular_values, _ = np.linalg.svd(M, full_matrices=False)
        observed_rank = _numerical_rank(singular_values, *M.shape)
        if observed_rank < rank:
            raise ValueError(
                f"Subject {sid!r} has numerical rank {observed_rank}, "
                f"below requested rank {rank}."
            )
        Q_list.append(U[:, :rank])
        summaries.append(np.concatenate([mean, std]))
        labels.append(int(row_labels[idx[0]]))

    return Q_list, np.vstack(summaries), np.asarray(labels, dtype=int)


def chordal_squared(
    Q_left: Sequence[np.ndarray],
    Q_right: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Squared chordal distances between fixed-rank Grassmann points."""
    if Q_right is None:
        Q_right = Q_left
    if not Q_left or not Q_right:
        return np.empty((len(Q_left), len(Q_right)), dtype=float)
    rank = Q_left[0].shape[1]
    if any(Q.shape[1] != rank for Q in [*Q_left, *Q_right]):
        raise ValueError("All subspaces must have the same rank.")
    out = np.empty((len(Q_left), len(Q_right)), dtype=float)
    for i, Qa in enumerate(Q_left):
        for j, Qb in enumerate(Q_right):
            out[i, j] = rank - float(np.linalg.norm(Qa.T @ Qb, ord="fro") ** 2)
    return np.maximum(out, 0.0)


def euclidean_squared(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Pairwise squared Euclidean distances."""
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    distances = (
        np.sum(A * A, axis=1, keepdims=True)
        + np.sum(B * B, axis=1, keepdims=True).T
        - 2.0 * (A @ B.T)
    )
    return np.maximum(distances, 0.0)


def _median_gamma(distances: np.ndarray) -> float:
    if distances.shape[0] == distances.shape[1]:
        values = distances[np.triu_indices(distances.shape[0], k=1)]
    else:
        values = distances.ravel()
    values = values[np.isfinite(values) & (values > 0.0)]
    median = float(np.median(values)) if values.size else 1.0
    return 1.0 / (2.0 * median + 1e-12)


def grassmann_geodesic(Qa: np.ndarray, Qb: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two Grassmann points along a shortest geodesic."""
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1].")
    overlap = Qa.T @ Qb
    U, cosine, Vt = np.linalg.svd(overlap)
    V = Vt.T
    cosine = np.clip(cosine, -1.0, 1.0)
    theta = np.arccos(cosine)
    sine = np.sin(theta)

    residual = Qb @ V - Qa @ U @ np.diag(cosine)
    inverse_sine = np.zeros_like(sine)
    nonzero = sine > 1e-10
    inverse_sine[nonzero] = 1.0 / sine[nonzero]
    tangent_basis = residual @ np.diag(inverse_sine)

    Qt = (
        Qa @ U @ np.diag(np.cos(t * theta))
        + tangent_basis @ np.diag(np.sin(t * theta))
    ) @ V.T
    Qt, _ = np.linalg.qr(Qt)
    return Qt[:, : Qa.shape[1]]


def _g_smote(
    Q: Sequence[np.ndarray],
    summaries: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    neighbors: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        return list(Q), np.asarray(summaries), np.asarray(labels), 0

    minority = int(classes[np.argmin(counts)])
    synth_count = int(np.max(counts) - np.min(counts))
    minority_indices = np.flatnonzero(labels == minority)
    if minority_indices.size < 2:
        raise ValueError("G-SMOTE requires at least two minority subjects.")

    Q_minority = [Q[i] for i in minority_indices]
    s_minority = summaries[minority_indices]
    distances = chordal_squared(Q_minority)
    np.fill_diagonal(distances, np.inf)
    k = min(neighbors, minority_indices.size - 1)
    nearest = np.argsort(distances, axis=1)[:, :k]
    rng = np.random.default_rng(random_state)

    Q_new: list[np.ndarray] = []
    s_new: list[np.ndarray] = []
    for _ in range(synth_count):
        anchor = int(rng.integers(minority_indices.size))
        partner = int(rng.choice(nearest[anchor]))
        t = float(rng.random())
        Q_new.append(grassmann_geodesic(Q_minority[anchor], Q_minority[partner], t))
        s_new.append(s_minority[anchor] + t * (s_minority[partner] - s_minority[anchor]))

    return (
        [*Q, *Q_new],
        np.vstack([summaries, np.vstack(s_new)]),
        np.concatenate([labels, np.full(synth_count, minority, dtype=int)]),
        synth_count,
    )


def _fit_fused_model(
    Q_train: Sequence[np.ndarray],
    s_train: np.ndarray,
    y_train: np.ndarray,
    weight: float,
    C: float,
) -> _FusedModel:
    d_g = chordal_squared(Q_train)
    d_e = euclidean_squared(s_train)
    gamma_g = _median_gamma(d_g)
    gamma_e = _median_gamma(d_e)
    K = weight * np.exp(-gamma_g * d_g) + (1.0 - weight) * np.exp(-gamma_e * d_e)
    classifier = SVC(kernel="precomputed", C=C, class_weight="balanced")
    classifier.fit(K, y_train)
    return _FusedModel(weight, C, gamma_g, gamma_e, list(Q_train), s_train, classifier)


def _classification_record(
    fold: int,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    synthetic_subjects: int = 0,
) -> dict[str, float | int | str]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "fold": fold,
        "method": method,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "synthetic_subjects": int(synthetic_subjects),
    }


def _prediction_rows(
    fold: int,
    method: str,
    subject_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "fold": fold,
            "method": method,
            "subject_id": sid,
            "true_label": int(yt),
            "predicted_label": int(yp),
            "decision_score": float(score),
        }
        for sid, yt, yp, score in zip(subject_ids, y_true, y_pred, scores)
    ]


def evaluate_repeated_measurements(
    frame: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    drop_cols: Sequence[str] = (),
    rank: int = 3,
    n_splits: int = 5,
    weight: float = 0.5,
    C: float = 1.0,
    g_smote_neighbors: int = 5,
    random_state: int = 42,
) -> EvaluationResult:
    """Run leakage-safe subject-level evaluation on repeated measurements."""
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

    for fold, (train_subject_index, test_subject_index) in enumerate(
        splitter.split(subject_ids, subject_labels), start=1
    ):
        train_ids = subject_ids[train_subject_index]
        test_ids = subject_ids[test_subject_index]
        train_rows = np.concatenate([row_map[sid] for sid in train_ids])
        test_rows = np.concatenate([row_map[sid] for sid in test_ids])

        scaler = StandardScaler().fit(X[train_rows])
        X_scaled = np.empty_like(X, dtype=float)
        X_scaled[train_rows] = scaler.transform(X[train_rows])
        X_scaled[test_rows] = scaler.transform(X[test_rows])

        Q_train, s_train, y_train = build_subject_views(
            X_scaled, row_ids, row_labels, train_ids, rank
        )
        Q_test, s_test, y_test = build_subject_views(
            X_scaled, row_ids, row_labels, test_ids, rank
        )

        # Balanced logistic regression.
        logistic = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=5000, random_state=random_state
        ).fit(s_train, y_train)
        scores = logistic.decision_function(s_test)
        pred = (scores > 0.0).astype(int)
        metrics.append(_classification_record(fold, "Balanced_LogReg", y_test, pred))
        predictions.extend(_prediction_rows(fold, "Balanced_LogReg", test_ids, y_test, pred, scores))

        # Balanced linear SVM.
        linear = LinearSVC(
            class_weight="balanced", C=C, max_iter=20000, random_state=random_state
        ).fit(s_train, y_train)
        scores = linear.decision_function(s_test)
        pred = (scores > 0.0).astype(int)
        metrics.append(_classification_record(fold, "Balanced_LinearSVM", y_test, pred))
        predictions.extend(_prediction_rows(fold, "Balanced_LinearSVM", test_ids, y_test, pred, scores))

        # Euclidean SMOTE + logistic regression.
        minimum = int(np.bincount(y_train).min())
        if minimum >= 2 and np.bincount(y_train)[0] != np.bincount(y_train)[1]:
            k_neighbors = min(g_smote_neighbors, minimum - 1)
            smote = SMOTE(random_state=random_state + fold, k_neighbors=k_neighbors)
            s_resampled, y_resampled = smote.fit_resample(s_train, y_train)
            synthetic_euclidean = len(y_resampled) - len(y_train)
        else:
            s_resampled, y_resampled = s_train, y_train
            synthetic_euclidean = 0
        smote_logistic = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=5000, random_state=random_state
        ).fit(s_resampled, y_resampled)
        scores = smote_logistic.decision_function(s_test)
        pred = (scores > 0.0).astype(int)
        metrics.append(
            _classification_record(fold, "SMOTE_LogReg", y_test, pred, synthetic_euclidean)
        )
        predictions.extend(_prediction_rows(fold, "SMOTE_LogReg", test_ids, y_test, pred, scores))

        # Fused Grassmann–Euclidean model with fold-local G-SMOTE.
        Q_aug, s_aug, y_aug, synthetic_geometric = _g_smote(
            Q_train,
            s_train,
            y_train,
            random_state=random_state * 1000 + fold,
            neighbors=g_smote_neighbors,
        )
        fused = _fit_fused_model(Q_aug, s_aug, y_aug, weight=weight, C=C)
        scores = fused.decision_function(Q_test, s_test)
        pred = (scores > 0.0).astype(int)
        metrics.append(
            _classification_record(
                fold, "Fused_Grassmann_GSMOTE", y_test, pred, synthetic_geometric
            )
        )
        predictions.extend(
            _prediction_rows(
                fold, "Fused_Grassmann_GSMOTE", test_ids, y_test, pred, scores
            )
        )

    fold_metrics = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)

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
    pooled = (
        fold_metrics.groupby("method", sort=False)[["TN", "FP", "FN", "TP"]]
        .sum()
        .reset_index()
    )

    return EvaluationResult(
        fold_metrics=fold_metrics,
        summary_metrics=summary.reset_index(drop=True),
        pooled_confusions=pooled,
        predictions=prediction_frame,
        feature_columns=tuple(feature_cols),
    )
