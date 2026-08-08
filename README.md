# VoxNeuro Minimal Method Repository

This repository contains the full executable code needed to demonstrate the core VoxNeuro method described in the MIT URTC paper:

- subject-disjoint cross-validation;
- training-fold-only standardization;
- repeated recordings represented as a Grassmann subspace;
- a complementary Euclidean mean–variability representation;
- positive-semidefinite fusion of Grassmann and Euclidean Gaussian kernels;
- Grassmann-geodesic SMOTE for imbalanced training folds;
- balanced logistic regression, SMOTE + logistic regression, and balanced linear-SVM comparators;
- fold-level predictions, summary metrics, and pooled confusion counts.


## Method summary

For subject $i$, repeated standardized recordings are stacked as

```math
M_i = [\widetilde{x}_{i1}, \ldots, \widetilde{x}_{im_i}] \in \mathbb{R}^{d \times m_i}.
```

A thin singular value decomposition gives a rank $r$ orthonormal basis $Q_i$, representing a point on $\mathrm{Gr}(r,d)$. The Euclidean view is

```math
s_i = [\mu_i; \sigma_i],
```

where $\mu_i$ and $\sigma_i$ are the coordinate-wise mean and standard deviation across recordings. The fused kernel is

```math
K(i,j) = w\exp\!\left[-\gamma_G d_{\mathrm{ch}}^2(Q_i,Q_j)\right]
+ (1-w)\exp\!\left[-\gamma_E \lVert s_i-s_j \rVert_2^2\right].
```

For an imbalanced training fold, the minority class is balanced by geodesic interpolation between neighboring minority subspaces, with the same interpolation coefficient used for the Euclidean view.

## Repository layout

```text
VoxNeuro_Method_Minimal/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/voxneuro/
│   ├── __init__.py
│   ├── method.py
│   └── cli.py
└── tests/
    └── test_method.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development/testing:

```bash
python -m pip install pytest
pytest -q
```

## Required CSV structure

Each row must be one recording. Required columns:

- a subject identifier, supplied with `--id-col`;
- a binary label, supplied with `--label-col`;
- numeric feature columns;
- optional metadata columns excluded with `--drop-cols`.

The label must be constant within each subject.

## Example commands

UCI-489-style columns:

```bash
voxneuro \
  --csv /path/to/uci489.csv \
  --id-col ID \
  --label-col Status \
  --drop-cols Recording \
  --rank 3 \
  --output-dir results/uci489
```

PD-252-style columns:

```bash
voxneuro \
  --csv /path/to/pd_speech_features.csv \
  --id-col id \
  --label-col class \
  --rank 3 \
  --output-dir results/pd252
```

## Outputs

The command writes:

- `fold_metrics.csv` — BA, macro-F1, and confusion counts for every method/fold;
- `summary_metrics.csv` — mean and standard error over outer folds;
- `pooled_confusions.csv` — pooled TN, FP, FN, and TP;
- `out_of_fold_predictions.csv` — one held-out prediction per subject and method.

## Reproducibility controls

- Outer folds are generated at the subject level.
- The scaler is fit only to training-fold recordings.
- Synthetic subjects are created only from training-fold minority subjects.
- Random generators use explicit seeds.
- The requested subspace rank is validated for every subject representation.
