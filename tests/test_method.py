from __future__ import annotations

import numpy as np
import pandas as pd

from voxneuro.method import (
    chordal_squared,
    evaluate_repeated_measurements,
    grassmann_geodesic,
)


def _synthetic_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(24):
        label = 0 if subject < 10 else 1
        subject_effect = rng.normal(loc=0.35 * label, scale=0.2, size=8)
        for recording in range(3):
            features = subject_effect + rng.normal(scale=0.25, size=8)
            rows.append(
                {
                    "subject": f"S{subject:02d}",
                    "recording": recording,
                    "label": label,
                    **{f"x{k}": value for k, value in enumerate(features)},
                }
            )
    return pd.DataFrame(rows)


def test_geodesic_returns_orthonormal_basis():
    rng = np.random.default_rng(1)
    Qa, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    Qb, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    Qt = grassmann_geodesic(Qa[:, :3], Qb[:, :3], 0.4)
    np.testing.assert_allclose(Qt.T @ Qt, np.eye(3), atol=1e-8)


def test_chordal_distance_is_symmetric_and_zero_on_diagonal():
    rng = np.random.default_rng(2)
    Q = []
    for _ in range(4):
        basis, _ = np.linalg.qr(rng.normal(size=(10, 3)))
        Q.append(basis[:, :3])
    distances = chordal_squared(Q)
    np.testing.assert_allclose(distances, distances.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-10)


def test_end_to_end_evaluation_produces_one_prediction_per_subject_and_method():
    frame = _synthetic_frame()
    result = evaluate_repeated_measurements(
        frame,
        id_col="subject",
        label_col="label",
        drop_cols=["recording"],
        rank=3,
        n_splits=3,
        random_state=11,
    )
    methods = {
        "Balanced_LogReg",
        "Balanced_LinearSVM",
        "SMOTE_LogReg",
        "Fused_Grassmann_GSMOTE",
    }
    assert set(result.summary_metrics["method"]) == methods
    assert len(result.predictions) == 24 * len(methods)
    counts = result.predictions.groupby(["method", "subject_id"]).size()
    assert counts.eq(1).all()
    assert result.fold_metrics["balanced_accuracy"].between(0.0, 1.0).all()
    assert result.fold_metrics["macro_f1"].between(0.0, 1.0).all()


def test_identifier_collisions_are_rejected():
    import pytest
    from voxneuro.method import _validate_frame
    rows = []
    for sid, label in ((1, 0), ("1", 1), (2, 0), (3, 1)):
        for r in range(3):
            rows.append({"subject": sid, "label": label, "x0": float(r), "x1": float(r) + 1.0, "x2": 2.0})
    frame = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        _validate_frame(frame, "subject", "label", [])


def test_rank_above_feature_count_is_rejected():
    import pytest
    from voxneuro.method import build_subject_views
    X = np.random.default_rng(0).normal(size=(6, 2))
    ids = np.array(["a"] * 3 + ["b"] * 3); labels = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        build_subject_views(X, ids, labels, ["a", "b"], rank=3, allow_rank_deficient=True)
