# VoxNeuro

**Leakage-safe multi-view Grassmannian learning for repeated Parkinsonian speech**

VoxNeuro is a reference implementation of the speech-classification pipeline described by Laiba Khan and Manas Waghe in *VoxNeuro: Accessible and Leakage-Safe Parkinson’s Screening Support via Multi-View Grassmannian Speech Analysis*, submitted to *Bioengineering* (MDPI) in 2026 and currently under review. The manuscript supersedes an earlier five-page conference version prepared for MIT URTC 2026.

The central design choice is simple but important: the subject, not an individual recording, is the statistical observation. Every recording from one person stays in the same outer fold, and all scaling, bandwidth estimation, and synthetic-subject generation are restricted to training data.

> [!IMPORTANT]
> This is research software, not a diagnostic system or medical device. It consumes precomputed, engineered CSV features rather than raw audio. Its outputs must not be used as clinical advice or as a substitute for neurological assessment.

## What this repository implements

- stratified, subject-disjoint outer cross-validation;
- training-fold-only recording-level standardization;
- a fixed-rank Grassmann representation of each subject's repeated recordings;
- a complementary Euclidean vector containing coordinate-wise means and population standard deviations;
- squared chordal distance and Gaussian kernels on the Grassmann and Euclidean views;
- positive-semidefinite weighted kernel fusion;
- project-specific Grassmann-geodesic synthetic minority oversampling (G-SMOTE);
- class-weighted logistic-regression and linear-SVM comparators;
- Euclidean SMOTE followed by logistic regression;
- fold metrics, cross-fold summaries, pooled confusion counts, and subject-level out-of-fold predictions.

This is not the complete paper analysis archive. The current package does **not** include the paper's matched Euclidean RBF + G-SMOTE ablation, feature-family permutation analysis, deployed web application, exact archived fold assignments, or submitted result files. Subject-level out-of-fold prediction files for the submitted manuscript will be archived here, as stated in the manuscript's Data Availability Statement.

## Method summary

For subject $i$, let $\widetilde{x}_{ij}\in\mathbb{R}^d$ be recording $j$ after standardization with statistics fitted only on the current outer-training fold. The repeated recordings are stacked as

```math
M_i = [\widetilde{x}_{i1}, \ldots, \widetilde{x}_{im_i}] \in \mathbb{R}^{d \times m_i}.
```

A thin singular value decomposition provides an orthonormal basis $Q_i$ for a point on $\mathrm{Gr}(r,d)$. The complementary Euclidean view is

```math
s_i = [\mu_i^\top,\sigma_i^\top]^\top \in \mathbb{R}^{2d},
```

where $\mu_i$ and $\sigma_i$ are the coordinate-wise mean and population standard deviation across that subject's standardized recordings. The squared chordal distance is

```math
d_{\mathrm{ch}}^2([Q_i],[Q_j])
= r-\lVert Q_i^\top Q_j\rVert_F^2,
```

and the fused kernel is

```math
K(i,j)
= w\exp\!\left[-\gamma_G d_{\mathrm{ch}}^2([Q_i],[Q_j])\right]
+ (1-w)\exp\!\left[-\gamma_E\lVert s_i-s_j\rVert_2^2\right].
```

The two Gaussian blocks are positive semidefinite, so their nonnegative weighted sum is also positive semidefinite. The implementation estimates each bandwidth from positive, squared, pairwise training distances using

```math
\gamma = \left[2\,\mathrm{median}(\delta^2)+10^{-12}\right]^{-1}.
```

For an imbalanced training fold, G-SMOTE selects a minority subject and one of its nearest minority neighbors under squared chordal distance. It follows a shortest Grassmann geodesic and uses the same draw $t\sim\mathcal{U}(0,1)$ to interpolate the Euclidean view. Synthetic subjects are generated only inside the training fold until its class counts match. This project-specific name does not refer to the separate Geometric SMOTE algorithm.

The paper experiments used three recordings per subject and fixed $r=3$, $w=0.5$, SVM $C=1$, five nearest minority neighbors, five stratified outer folds, and random seed 42. UCI-489 is balanced, so no synthetic subjects were generated for it.

## Results 

| Dataset | Paper model | Balanced accuracy, mean ± SE | Macro-F1, mean ± SE |
| --- | --- | ---: | ---: |
| UCI-489 | Grassmann-Euclidean fusion | 0.850 ± 0.042 | 0.847 ± 0.043 |
| PD-252 | Grassmann-Euclidean fusion + G-SMOTE | 0.749 ± 0.039 | 0.760 ± 0.035 |


## Datasets

The datasets are public and de-identified but are not bundled with this repository.

| Paper label | Official source | Cohort | Recordings | Modeled variables in the paper | Distributed file |
| --- | --- | ---: | ---: | ---: | --- |
| UCI-489 | [Parkinson Dataset with replicated acoustic features](https://archive.ics.uci.edu/dataset/489/parkinson%2Bdataset%2Bwith%2Breplicated%2Bacoustic%2Bfeatures) | 80 subjects: 40 PD, 40 control | 3 per subject | 45 | `ReplicatedAcousticFeatures-ParkinsonDatabase.csv` |
| PD-252 | [Parkinson's Disease Classification, UCI dataset 470](https://archive.ics.uci.edu/dataset/470/parkinson%2Bs%2Bdisease%2Bclassification) | 252 subjects: 188 PD, 64 control | 3 per subject | 753 | `pd_speech_features.csv` inside `pd_speech_features.rar` |

`PD-252` is the paper's shorthand for the 252-subject cohort in the dataset, with 470 recordings across 252 unique subjects; it is **not** a UCI dataset number. Its official UCI identifier is 470. Both datasets contain engineered features, not raw speech recordings.

## Installation

Python 3.10 or newer is required.

### macOS or Linux

```bash
git clone https://github.com/khan-laiba/voxneuro.git
cd voxneuro

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Windows PowerShell

```powershell
git clone https://github.com/khan-laiba/voxneuro.git
cd voxneuro

py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Preparing and running the public datasets

### UCI-489

Download and extract `ReplicatedAcousticFeatures-ParkinsonDatabase.csv` from the [official UCI-489 page](https://archive.ics.uci.edu/dataset/489/parkinson%2Bdataset%2Bwith%2Breplicated%2Bacoustic%2Bfeatures), then run the rank-3 smoke test:

```bash
voxneuro \
  --csv /path/to/ReplicatedAcousticFeatures-ParkinsonDatabase.csv \
  --id-col ID \
  --label-col Status \
  --drop-cols Recording \
  --rank 3 \
  --output-dir results/uci489-rank3
```

`Gender` is intentionally retained as a modeled variable, matching the paper.

### PD-252

Download the archive from the [official UCI-470 page](https://archive.ics.uci.edu/dataset/470/parkinson%2Bs%2Bdisease%2Bclassification). UCI distributes an outer ZIP containing `pd_speech_features.rar`; extract both archives.

The CSV has a feature-family grouping row above its actual header. The CLI does not currently expose a `skiprows` option, so create a clean CSV once:

```python
from pathlib import Path

import pandas as pd

source = Path("/path/to/pd_speech_features.csv")
target = source.with_name("pd_speech_features_clean.csv")

pd.read_csv(source, header=1).to_csv(target, index=False)
print(target)
```

Then run the rank-3 smoke test:

```bash
voxneuro \
  --csv /path/to/pd_speech_features_clean.csv \
  --id-col id \
  --label-col class \
  --rank 3 \
  --output-dir results/pd252-rank3
```

The `gender` column is intentionally retained as a modeled variable, matching the paper.

## Input CSV contract

Each row must represent one recording. The package requires:

- one nonmissing subject identifier column;
- one nonmissing binary label column;
- a constant label within each subject;
- numeric feature columns with no missing values;
- optional metadata columns named with `--drop-cols`;
- at least `--rank` recordings and numerical rank at least `--rank` for every subject;
- at least `--splits` subjects from each class.

Exactly two unique labels are required. The implementation sorts them and maps the first to `0` and the second to `1`. This mapping is not currently exported, so use explicit `0`/`1` labels when the interpretation of TN, FP, FN, and TP matters.

## CLI reference

```text
voxneuro --csv PATH --id-col COLUMN --label-col COLUMN [options]
```

| Option | Default | Meaning |
| --- | ---: | --- |
| `--csv` | required | Repeated-measurement CSV path |
| `--id-col` | required | Subject identifier column |
| `--label-col` | required | Binary label column |
| `--drop-cols` | none | Non-feature columns to exclude |
| `--output-dir` | `results` | Destination for CSV outputs |
| `--rank` | `3` | Fixed Grassmann rank |
| `--splits` | `5` | Stratified subject-level outer folds |
| `--weight` | `0.5` | Grassmann-kernel fusion weight $w$ |
| `--C` | `1.0` | SVM penalty |
| `--gsmote-neighbors` | `5` | Maximum minority neighbors used by G-SMOTE and Euclidean SMOTE |
| `--seed` | `42` | Cross-validation and augmentation seed |

Run `voxneuro --help` for the installed command's authoritative option list.

## Implemented methods

| Output identifier | Method |
| --- | --- |
| `Balanced_LogReg` | Class-weighted logistic regression on the Euclidean summary |
| `Balanced_LinearSVM` | Class-weighted linear SVM on the Euclidean summary |
| `SMOTE_LogReg` | Euclidean SMOTE on training summaries, followed by class-weighted logistic regression |
| `Fused_Grassmann_GSMOTE` | Fold-local G-SMOTE followed by a class-weighted, precomputed-kernel SVC on the fused kernel |

When a training fold is already balanced, no samples are synthesized. In that case `SMOTE_LogReg` receives the unmodified training data, and the fused output retains its method identifier while recording zero synthetic subjects.

## Outputs

The output directory contains:

| File | Contents |
| --- | --- |
| `fold_metrics.csv` | Fold, method, balanced accuracy, macro-F1, TN, FP, FN, TP, and synthetic-subject count |
| `summary_metrics.csv` | Mean and standard error of balanced accuracy and macro-F1 across outer folds |
| `pooled_confusions.csv` | Pooled TN, FP, FN, and TP by method |
| `out_of_fold_predictions.csv` | One held-out prediction and decision score per subject and method |

Pooled confusion rates and arithmetic means of fold-level metrics answer different summaries and need not be numerically identical.

## Repository layout

```text
voxneuro/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── voxneuro/
│       ├── __init__.py
│       ├── cli.py
│       └── method.py
└── tests/
    └── test_method.py
```

## Tests

After installing the package:

```bash
python -m pip install pytest
pytest -q
```

The current suite contains three focused tests covering geodesic orthonormality, chordal-distance symmetry and diagonal behavior, and an end-to-end subject-level evaluation smoke test.

## Reproducibility notes

- Outer folds are stratified and generated at the subject level.
- The scaler is fit only to recordings from training subjects.
- Kernel bandwidths are estimated only from augmented training representations.
- Synthetic subjects are created only inside imbalanced training folds.
- Randomized components receive explicit seeds.
- Subject-level out-of-fold predictions make aggregate metrics auditable.
- Dependency versions are lower-bounded rather than locked.
- The repository does not store the paper's exact fold assignments, prepared datasets, or published result artifacts.

## Paper, project, and citations

**Paper**

Laiba Khan and Manas Waghe, “VoxNeuro: Accessible and Leakage-Safe Parkinson’s Screening Support via Multi-View Grassmannian Speech Analysis,” submitted to *Bioengineering* (MDPI), 2026. Under review; this entry and `CITATION.cff` will be updated with the final citation upon publication.

```bibtex
@unpublished{khan2026voxneuro,
  author = {Khan, Laiba and Waghe, Manas},
  title  = {VoxNeuro: Accessible and Leakage-Safe Parkinson's Screening Support
            via Multi-View Grassmannian Speech Analysis},
  note   = {Submitted to Bioengineering (MDPI); under review},
  year   = {2026}
}
```

**Project**

- [VoxNeuro source repository](https://github.com/khan-laiba/voxneuro)
- [VoxNeuro research-stage web application](https://voxneuro.org)

The web application is separate from this package. No application-collected data were used in the paper's reported analyses.

**Datasets**

- C. J. Pérez, “Parkinson Dataset with replicated acoustic features,” UCI Machine Learning Repository, 2016. [Dataset page](https://archive.ics.uci.edu/dataset/489/parkinson%2Bdataset%2Bwith%2Breplicated%2Bacoustic%2Bfeatures) · [doi:10.24432/C5701F](https://doi.org/10.24432/C5701F)
- C. Sakar, G. Serbes, A. Gunduz, H. Nizam, and B. Sakar, “Parkinson's Disease Classification,” UCI Machine Learning Repository, 2018. [Dataset page](https://archive.ics.uci.edu/dataset/470/parkinson%2Bs%2Bdisease%2Bclassification) · [doi:10.24432/C5MS4X](https://doi.org/10.24432/C5MS4X)
- C. O. Sakar et al., “A comparative analysis of speech signal processing algorithms for Parkinson's disease classification and the use of the tunable Q-factor wavelet transform,” *Applied Soft Computing*, vol. 74, pp. 255-263, 2019. [doi:10.1016/j.asoc.2018.10.022](https://doi.org/10.1016/j.asoc.2018.10.022)

The UCI dataset pages list both datasets under the Creative Commons Attribution 4.0 International license. Cite the datasets when using them.

## License

The VoxNeuro source code is released under the [MIT License](LICENSE). The UCI datasets are licensed separately under the Creative Commons Attribution 4.0 International license; their terms are not changed by this repository's code license.
