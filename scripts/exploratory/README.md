# Exploratory development scripts (archived, not part of the reported method)

These scripts document the method-development search that preceded the adaptive-subspace fused model
(`voxneuro.adaptive`, release v2.0.0). They were run on the default outer partition (seed 42) of both cohorts,
i.e. on the *development* partition, and their results must not be read as confirmatory estimates.
They import each other by their distributed module names (`family_mkl.py`, `subspace_dev.py`, `exploration2.py`) and expect the two cohort CSVs under `data/` (`ReplicatedAcousticFeatures-ParkinsonDatabase.csv`, `pd_speech_features_clean.csv`) and an existing `robustness_out/` directory; run them with `PYTHONPATH=src:scripts/exploratory python scripts/exploratory/<script>.py` from the repository root.

| Script | Content | Results |
| --- | --- | --- |
| `exploration1.py`–`exploration3.py` | family-wise alignment-weighted multiple-kernel SVMs (ALIGNF), MMD set kernels, record-level SVMs, order statistics, permanent kernels, reliability-weighted metrics, nested thresholds, subagging, PLS-DA | `results/exploratory/exploration1-3.csv` |
| `exploration4_pls.py` | supervised (PLS) projection of recordings before the Grassmann/Euclidean views | `results/exploratory/exploration4.csv` |
| `family_mkl.py`, `family_mkl_robust.py` | family-wise alignment MKL with nested selection; 24 seeds + 20 partitions | `results/exploratory/family_mkl_robust.csv` |
| `subspace_dev.py`, `subspace_dev_robust.py`, `unified_sweep.py` | development versions of the adaptive-subspace model and candidate ensembles; 24 seeds + 20 partitions | `results/exploratory/subspace_dev_robust.csv`, `unified_sweep.csv` |

The family-wise alignment-weighted combination improved PD-252 but degraded the low-dimensional UCI-489 cohort
(`family_mkl_robust.csv`) and was therefore not adopted. The definitive implementation and the reported numbers are
produced by `src/voxneuro/adaptive.py` and `scripts/adaptive_package.py`.
