from __future__ import annotations

import numpy as np
import pandas as pd

from voxneuro.adaptive import (
    evaluate_adaptive,
    fit_adaptive_model,
    fit_normalizer,
    fit_supervised_projection,
    subspace_gate_active,
)
from voxneuro.method import _validate_frame


def _synthetic_frame(n_subjects: int = 30, n_features: int = 8, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        label = 0 if subject < n_subjects // 3 else 1
        subject_effect = rng.normal(loc=0.35 * label, scale=0.2, size=n_features)
        for recording in range(3):
            features = subject_effect + rng.normal(scale=0.25, size=n_features)
            rows.append({"subject": f"S{subject:02d}", "recording": recording, "label": label,
                         **{f"x{k}": value for k, value in enumerate(features)}})
    return pd.DataFrame(rows)


def test_gate_follows_the_dimensionality_rule():
    assert subspace_gate_active(753, 604, "auto")
    assert not subspace_gate_active(45, 192, "auto")
    assert subspace_gate_active(45, 192, "on")
    assert not subspace_gate_active(753, 604, "off")


def test_rank_normalizer_maps_training_rows_monotonically_to_normal_scores():
    rng = np.random.default_rng(0)
    X_train = rng.lognormal(size=(120, 4))
    normalizer = fit_normalizer(X_train, "rank")
    Z = normalizer.transform(X_train)
    # Training rows map to (approximately) standard normal scores; the mapping is monotone per feature.
    iqr = np.subtract(*np.percentile(Z, [75, 25]))
    assert abs(np.median(Z)) < 0.1 and 0.85 < iqr / 1.349 < 1.2
    order = np.argsort(X_train[:, 0])
    assert np.all(np.diff(Z[order, 0]) >= -1e-12)


def test_supervised_projection_dimension_and_target_coding():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 12))
    y = np.repeat([0, 1, 1], 30)
    projection = fit_supervised_projection(X, y, 4)
    assert projection.transform(X).shape == (90, 4)
    # the class-count coding is an affine transformation of the labels, so the rotations equal those of 0/1 targets
    from sklearn.cross_decomposition import PLSRegression
    plain = PLSRegression(n_components=4, scale=False).fit(X, y.astype(float))
    assert np.allclose(np.abs(projection.x_rotations_), np.abs(plain.x_rotations_), atol=1e-8)


def test_rank_normalizer_ignores_held_out_rows():
    rng = np.random.default_rng(5)
    X_train = rng.lognormal(size=(100, 3))
    X_other = rng.lognormal(size=(50, 3)) * 10
    a = fit_normalizer(X_train, "rank").transform(X_train)
    b = fit_normalizer(X_train, "rank").transform(np.vstack([X_train, X_other]))[:100]
    assert np.allclose(a, b)


def test_empty_or_duplicate_subspace_dims_are_rejected():
    import pytest
    frame = _synthetic_frame(n_subjects=30, n_features=120)
    df, feats = _validate_frame(frame, "subject", "label", ["recording"])
    X = df[feats].to_numpy(float); row_ids = df["subject"].to_numpy(str); row_labels = df["label"].to_numpy(int)
    train_ids = sorted(set(row_ids))[:24]
    with pytest.raises(ValueError):
        fit_adaptive_model(X, row_ids, row_labels, train_ids, subspace_dims=(), random_state=3)
    with pytest.raises(ValueError):
        fit_adaptive_model(X, row_ids, row_labels, train_ids, subspace_dims=(4, 4), random_state=3)


def test_low_dimensional_frame_reduces_to_full_space_model():
    frame = _synthetic_frame()
    df, feats = _validate_frame(frame, "subject", "label", ["recording"])
    X = df[feats].to_numpy(float)
    row_ids = df["subject"].to_numpy(str)
    row_labels = df["label"].to_numpy(int)
    train_ids = sorted(set(row_ids))[:24]
    model = fit_adaptive_model(X, row_ids, row_labels, train_ids, random_state=3)
    assert not model.gate_active and model.full_model is not None and model.members == []
    scores = model.decision_function(X, row_ids, row_labels, sorted(set(row_ids))[24:])
    assert scores.shape == (6,)


def test_high_dimensional_frame_activates_subspace_ensemble():
    frame = _synthetic_frame(n_subjects=30, n_features=120)  # 120 features > 72 training recordings
    df, feats = _validate_frame(frame, "subject", "label", ["recording"])
    X = df[feats].to_numpy(float)
    row_ids = df["subject"].to_numpy(str)
    row_labels = df["label"].to_numpy(int)
    train_ids = sorted(set(row_ids))[:24]
    model = fit_adaptive_model(X, row_ids, row_labels, train_ids, subspace_dims=(4, 8), random_state=3)
    assert model.gate_active and len(model.members) == 2 and model.synthetic_subjects == 0
    scores = model.decision_function(X, row_ids, row_labels, sorted(set(row_ids))[24:])
    assert scores.shape == (6,) and np.all(np.isfinite(scores))


def test_end_to_end_evaluation_reports_every_method_once_per_subject():
    frame = _synthetic_frame(n_subjects=30, n_features=120)
    result = evaluate_adaptive(frame, id_col="subject", label_col="label", drop_cols=["recording"],
                               n_splits=3, subspace_dims=(4, 8), random_state=5)
    methods = set(result.summary_metrics["method"])
    assert {"Adaptive_Subspace_Fusion", "Fused_Grassmann_GSMOTE", "Euclidean_RBF_GSMOTE", "Balanced_LogReg",
            "Balanced_LinearSVM", "SMOTE_LogReg", "Balanced_RandomForest", "PLS_DA", "Subspace_q4", "Subspace_q8"} <= methods
    counts = result.predictions.groupby("method")["subject_id"].nunique()
    assert (counts == 30).all()
    assert set(result.pooled_confusions.columns) >= {"TN", "FP", "FN", "TP"}
