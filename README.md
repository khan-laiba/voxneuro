# VoxNeuro

**Leakage-safe multi-view Grassmannian learning for repeated Parkinsonian speech**

VoxNeuro is a reference implementation of the speech-classification pipeline described by Laiba Khan and Manas Waghe in *VoxNeuro: Leakage-Safe Parkinson’s Screening Support via Multi-View Grassmannian Speech Analysis*, prepared for submission to *Bioengineering* (MDPI) in 2026 (unpublished).

The central design choice is simple but important: the subject, not an individual recording, is the statistical observation. Every recording from one person stays in the same outer fold, and all scaling, bandwidth estimation, and synthetic-subject generation are restricted to training data.

> [!IMPORTANT]
> This is research software, not a diagnostic system or medical device. It consumes precomputed, engineered CSV features rather than raw audio. Its outputs must not be used as clinical advice or as a substitute for neurological assessment.

## What this repository implements

- stratified, subject-disjoint outer cross-validation;
- training-fold-only recording-level normalization (Gaussian rank normalization for the manuscript's model; z-standardization for the retained original variant);
- a fixed-rank Grassmann representation of each subject's repeated recordings;
- a complementary Euclidean vector containing coordinate-wise means and population standard deviations;
- squared chordal distance and Gaussian kernels on the Grassmann and Euclidean views;
- positive-semidefinite weighted kernel fusion;
- project-specific Grassmann-geodesic synthetic minority oversampling (G-SMOTE);
- standard comparators on the subject summaries: class-weighted logistic regression, class-weighted linear SVM, Euclidean SMOTE followed by logistic regression, a class-weighted random forest, and PLS-DA with the number of components chosen by inner cross-validation;
- the full-space Euclidean-kernel ablation (fusion weight 0 on the same training folds and, where active, the same G-SMOTE augmentation);
- fold metrics, cross-fold summaries, pooled confusion counts, and subject-level out-of-fold predictions;
- **the adaptive-subspace fused model, referred to as VoxNeuro in the manuscript** (`voxneuro.adaptive`, default since v2.0.0): fold-local Gaussian rank normalization of the recordings and, when the feature count exceeds the number of training recordings, a supervised-subspace ensemble (PLS projections to $q\in\{8,16,32\}$ dimensions, one fused Grassmann–Euclidean SVM per dimension, decision values averaged after division by their training-fold standard deviation). When the gate is inactive the model reduces to the rank-normalized fused model with the G-SMOTE gate.

The default CLI (`--model adaptive`) emits the manuscript's model, its ablations (including the full-space Euclidean-kernel ablation `Euclidean_RBF_GSMOTE`), the standard comparators, and the individual subspace members. The feature-family permutation analysis is provided by [`scripts/robustness.py`](scripts/robustness.py) with its archived outputs (`results/robustness/permutation_sensitivity.csv`, `family_map.csv`). The deployed web application is a separate project and is not included. Subject-level out-of-fold predictions with fold identifiers are archived under `results/` for the reference-partition evaluations of both cohorts under both normalizations, as stated in the manuscript's Data Availability Statement; the partition, dimension-sweep, view-ablation, and augmentation-seed analyses are archived as run-level summaries.

## Method summary

The manuscript's model (`--model adaptive`, the default) applies Gaussian rank normalization fitted on the training-fold recordings and, when the feature count exceeds the number of training recordings, a supervised PLS subspace ensemble before the views below are built; the formulas below describe the shared subject representation and fused kernel, written for the z-standardized full-space variant that `--model original` runs.

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

The paper experiments used three recordings per subject and fixed $r=3$, $w=0.5$, SVM $C=1$, five nearest minority neighbors, five stratified outer folds, and outer-fold seed 42; the original PD-252 augmentation state was not retained, and the archived default-seed and seed-sensitivity runs use the documented derived seeds. UCI-489 is balanced, so no synthetic subjects were generated for it.

## Results

Subject-level, five-fold subject-disjoint cross-validation at the reference partition (seed 42), mean ± fold-level SE (descriptive). The reference-partition estimates of the adaptive model are design-stage estimates (the configuration was fixed after inspecting them); the 20-partition study is an internal post-development resampling-sensitivity analysis on the same subjects, not independent validation.

| Dataset | Model | Balanced accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| UCI-489 | VoxNeuro (`Adaptive_Subspace_Fusion`; gate inactive = rank-normalized full-space model) | 0.875 ± 0.040 | 0.873 ± 0.040 |
| UCI-489 | Full-space Euclidean-kernel ablation (`Euclidean_RBF_GSMOTE`, rank-normalized) | 0.863 ± 0.036 | 0.860 ± 0.037 |
| UCI-489 | Z-standardized full-space model (`--model original`) | 0.850 ± 0.042 | 0.847 ± 0.043 |
| UCI-489 | PLS-DA / class-weighted random forest / class-weighted logistic regression / class-weighted linear SVM | 0.825 / 0.800 / 0.800 / 0.800 | 0.822 / 0.799 / 0.796 / 0.797 |
| PD-252 | VoxNeuro (`Adaptive_Subspace_Fusion`) | 0.816 ± 0.040 | 0.811 ± 0.034 |
| PD-252 | Rank-normalized full-space model + G-SMOTE (`Fused_Grassmann_GSMOTE`) | 0.782 ± 0.035 | 0.781 ± 0.028 |
| PD-252 | Full-space Euclidean-kernel ablation + G-SMOTE (`Euclidean_RBF_GSMOTE`, rank-normalized) | 0.766 ± 0.035 | 0.757 ± 0.028 |
| PD-252 | Z-standardized full-space model + G-SMOTE (`--model original`, default augmentation seed) | 0.731 ± 0.032 | 0.749 ± 0.031 |
| PD-252 | SMOTE + logistic regression / class-weighted logistic regression / PLS-DA / class-weighted linear SVM / class-weighted random forest | 0.777 / 0.769 / 0.747 / 0.734 / 0.671 | 0.786 / 0.774 / 0.744 / 0.747 / 0.696 |

Across 20 alternative outer subject partitions (StratifiedKFold seeds 0–19), the adaptive-subspace fused model averaged 0.784 ± 0.017 balanced accuracy and 0.785 ± 0.014 macro-F1 on PD-252 versus 0.721 ± 0.020 and 0.734 ± 0.019 for the z-standardized full-space model (paired differences +0.062 and +0.050, positive in all 20 partitions on both metrics) and 0.752 ± 0.024 / 0.757 ± 0.024 for class-weighted logistic regression, the highest-scoring standard comparator (VoxNeuro higher in 19 of 20 partitions on each metric); on UCI-489 it averaged 0.853 ± 0.019 / 0.852 ± 0.019 versus 0.853 ± 0.013 / 0.851 ± 0.014 for the z-standardized full-space model (similar mean performance). Pooled PD-252 confusion counts of VoxNeuro at the reference partition: TP 169, FN 19, TN 47, FP 17 (sensitivity 0.899, specificity 0.734).

[`results/adaptive/`](results/adaptive/) archives the adaptive-model package produced by [`scripts/adaptive_package.py`](scripts/adaptive_package.py) under the pinned environment: `adaptive_comparators.csv` (the model, its ablations and the standard comparators — class-weighted logistic regression, class-weighted linear SVM, SMOTE + logistic regression, class-weighted random forest, PLS-DA with nested component selection — and the full-space Euclidean-kernel ablation; both cohorts, rank normalization and z-standardization), `adaptive_default_<cohort>_<normalization>/` (fold metrics, pooled confusions, subject-level out-of-fold predictions), `adaptive_partitions.csv` (20 outer partitions, all models, both normalizations, plus the gate forced on for UCI-489), `adaptive_dimension_sweep.csv` (single subspace dimensions 4–64 versus the ensemble), `adaptive_weight_ablation.csv` (fusion weight 0, 0.5, 1 inside the adaptive model), `adaptive_gsmote_ablation.csv` (G-SMOTE inside the subspace branch, 24 seeds) and `adaptive_concentration.csv` (coefficient of variation of pairwise chordal and Euclidean distances before and after the projection).

```bash
PYTHONPATH=src python scripts/adaptive_package.py --uci <UCI-489 csv> --pd <PD-252 clean csv> --out results/adaptive
```

The retained original variant (`--model original`, z-standardized full-space model) and its supplementary analyses (manuscript Appendix A) are archived below.

[`results/pd252-seed-study/`](results/pd252-seed-study/) archives a paired G-SMOTE seed-sensitivity study on PD-252 (24 distinct augmentation seeds, outer folds fixed): per-seed balanced accuracy and macro-F1 for the fused model and its matched Euclidean RBF ablation (the same pipeline with `--weight 0`), produced by [`scripts/seed_sensitivity.py`](scripts/seed_sensitivity.py) under the pinned environment.

[`results/pd252-default-seed/`](results/pd252-default-seed/) archives the four output CSVs of a PD-252 run of this package at the default seed (rank 3, seed 42, `--allow-rank-deficient`, five folds) under the pinned environment: fused balanced accuracy 0.731 and macro-F1 0.749, the default-seed values quoted in the manuscript's Table 4 and Appendix A, with subject-level out-of-fold predictions and pooled confusion counts.

[`results/robustness/`](results/robustness/) archives the robustness package of the z-standardized full-space variant reported in the manuscript's Appendix A, produced by [`scripts/robustness.py`](scripts/robustness.py) and [`scripts/augmentation_seeds.py`](scripts/augmentation_seeds.py) under the pinned environment: `unified_comparators.csv` / `unified_fold_metrics.csv` (fused, matched Euclidean RBF, Grassmann-only, class-weighted logistic regression, class-weighted linear SVM, SMOTE + logistic regression, SMOTE + Euclidean RBF; both cohorts; PD-252 at the default seed), `oof_predictions_fused.csv` (subject-level out-of-fold predictions for both cohorts), `partition_sensitivity.csv` (20 outer subject partitions), `rank_sensitivity.csv` (rank 2 vs. 3), `augmentation_seed_study.csv` (24 seeds: fused, G-SMOTE + RBF, SMOTE + RBF), `permutation_sensitivity.csv` (feature-family permutation of the Euclidean view, 20 draws per family and fold) and `family_map.csv` (column-to-family assignment). The fusion-weight package adds `w_sweep.csv` ([`scripts/w_sweep.py`](scripts/w_sweep.py): fixed weights 0–1 across the 24 seeds and 20 partitions), `adaptive_seed.csv` / `adaptive_partition.csv` ([`scripts/adaptive_fusion2.py`](scripts/adaptive_fusion2.py): fusion weight selected by repeated 3 × 5 inner cross-validation inside each training fold), `ensemble_fusion.csv` ([`scripts/ensemble_fusion.py`](scripts/ensemble_fusion.py): tuning-free weight ensemble) and `fusion_exploration.csv` / `fusion_exploration2.csv` ([`scripts/fusion_explore.py`](scripts/fusion_explore.py), [`scripts/fusion_explore2.py`](scripts/fusion_explore2.py): exploratory variants — centered, PCA-reduced, rank-one and block-balanced Grassmann views, decision-level fusion, kernel-alignment and nested weight selection, SVM penalties).

```bash
PYTHONPATH=src python scripts/robustness.py --uci <UCI-489 csv> --pd <PD-252 clean csv> --out results/robustness
mkdir -p robustness_out && PYTHONPATH=src python scripts/augmentation_seeds.py   # expects data/pd_speech_features_clean.csv; writes robustness_out/augmentation_seed_study.csv (copy to results/robustness/)
```

The [`results/uci489-rank3/`](results/uci489-rank3/) directory archives the four output CSVs of a UCI-489 run of this package (rank 3, seed 42, `--allow-rank-deficient`, five folds) under the pinned environment in [`requirements-lock.txt`](requirements-lock.txt), including subject-level out-of-fold predictions; it is the original z-standardized run (`--model original`; balanced accuracy 0.850, macro-F1 0.847), retained for the record.


## Datasets

The datasets are public and de-identified but are not bundled with this repository.

| Paper label | Official source | Cohort | Recordings | Modeled variables in the paper | Distributed file |
| --- | --- | ---: | ---: | ---: | --- |
| UCI-489 | [Parkinson Dataset with replicated acoustic features](https://archive.ics.uci.edu/dataset/489/parkinson%2Bdataset%2Bwith%2Breplicated%2Bacoustic%2Bfeatures) | 80 subjects: 40 PD, 40 control | 3 per subject | 45 | `ReplicatedAcousticFeatures-ParkinsonDatabase.csv` |
| PD-252 | [Parkinson's Disease Classification, UCI dataset 470](https://archive.ics.uci.edu/dataset/470/parkinson%2Bs%2Bdisease%2Bclassification) | 252 subjects: 188 PD, 64 control | 3 per subject | 753 | `pd_speech_features.csv` inside `pd_speech_features.rar` |

`PD-252` is the paper's shorthand for the 252-subject cohort in the dataset, with 756 recordings (three per subject) across 252 unique subjects; it is **not** a UCI dataset number. Its official UCI identifier is 470. Both datasets contain engineered features, not raw speech recordings.

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

Download and extract `ReplicatedAcousticFeatures-ParkinsonDatabase.csv` from the [official UCI-489 page](https://archive.ics.uci.edu/dataset/489/parkinson%2Bdataset%2Bwith%2Breplicated%2Bacoustic%2Bfeatures), then run the manuscript's model (the default `--model adaptive`; add `--model original` to reproduce the archived z-standardized run in `results/uci489-rank3/`):

```bash
voxneuro \
  --csv /path/to/ReplicatedAcousticFeatures-ParkinsonDatabase.csv \
  --id-col ID \
  --label-col Status \
  --drop-cols Recording \
  --rank 3 \
  --allow-rank-deficient \
  --output-dir results/uci489-adaptive
```

`--allow-rank-deficient` is required: the official CSV contains byte-identical repeated recordings for subject `CONT-36`, so that subject's matrix has numerical rank 2 and the paper's rank-3 configuration retains its first three left-singular vectors, as in the paper's Equation (4).

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

Then run the manuscript's model (add `--model original` for the archived z-standardized run in `results/pd252-default-seed/`):

```bash
voxneuro \
  --csv /path/to/pd_speech_features_clean.csv \
  --id-col id \
  --label-col class \
  --rank 3 \
  --allow-rank-deficient \
  --output-dir results/pd252-adaptive
```

`--allow-rank-deficient` is required here as well: the official file contains byte-identical repeated recordings for subject id `37`.

The `gender` column is intentionally retained as a modeled variable, matching the paper.

## Input CSV contract

Each row must represent one recording. The package requires:

- one nonmissing subject identifier column;
- one nonmissing binary label column;
- a constant label within each subject;
- numeric feature columns with no missing values;
- optional metadata columns named with `--drop-cols`;
- at least `--rank` recordings, at least `--rank` features, and (unless `--allow-rank-deficient`) numerical rank at least `--rank` for every subject;
- at least `--splits` subjects from each class; the PLS-DA comparator additionally uses an inner stratified split with `min(5, smallest training class count)` folds, so at least three training subjects per class are required.

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
| `--model` | `adaptive` | `adaptive`: rank-normalized, dimension-adaptive model with its comparators (v2.0.0); `original`: z-standardized fused model of the original submission |
| `--normalization` | `rank` | Recording-level normalization for the adaptive model (`rank` or `standard`), fitted on training folds only |
| `--subspace-dims` | `8 16 32` | Supervised-subspace dimensions of the ensemble members |
| `--gate` | `auto` | Supervised-subspace gate: `auto` activates it when the feature count exceeds the number of training recordings; `on`/`off` force it |

Run `voxneuro --help` for the installed command's authoritative option list.

## Implemented methods

| Output identifier | Method |
| --- | --- |
| `Balanced_LogReg` | Class-weighted logistic regression on the Euclidean summary |
| `Balanced_LinearSVM` | Class-weighted linear SVM on the Euclidean summary |
| `SMOTE_LogReg` | Euclidean SMOTE on training summaries, followed by class-weighted logistic regression |
| `Fused_Grassmann_GSMOTE` | Fold-local G-SMOTE followed by a class-weighted, precomputed-kernel SVC on the fused kernel |
| `Adaptive_Subspace_Fusion` | (`--model adaptive`) rank-normalized, dimension-adaptive fused model: supervised-subspace ensemble when the gate is active, otherwise the rank-normalized fused model with the G-SMOTE gate |
| `Euclidean_RBF_GSMOTE` | (`--model adaptive`) full-space Euclidean-kernel ablation (fusion weight 0) under the same normalization and, where active, the same G-SMOTE augmentation |
| `Balanced_RandomForest` | (`--model adaptive`) class-weighted random forest (500 trees, `balanced_subsample` class weights) on the Euclidean summary |
| `PLS_DA` | (`--model adaptive`) PLS discriminant analysis on the standardized Euclidean summary (components chosen from {2, 4, 8, 16} by inner cross-validation, shrinkage LDA with equal priors) |
| `Subspace_q<k>` | (`--model adaptive`, gate active) individual ensemble members with subspace dimension *k* |

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
├── README.md, CITATION.cff, LICENSE
├── pyproject.toml, requirements.txt, requirements-lock.txt
├── paper/                       manuscript PDF (current) and the archived initial version
├── results/
│   ├── adaptive/                manuscript package (comparators, partitions, sweep, ablations, concentration, OOF predictions)
│   ├── robustness/              supplementary analyses of the z-standardized full-space variant (Appendix A)
│   ├── exploratory/             archived development-search outputs
│   ├── pd252-seed-study/, pd252-default-seed/, uci489-rank3/   archived runs of the original variant
├── scripts/
│   ├── adaptive_package.py      regenerates results/adaptive
│   ├── robustness.py, seed_sensitivity.py, augmentation_seeds.py, w_sweep.py, adaptive_fusion2.py, ensemble_fusion.py, fusion_explore*.py
│   └── exploratory/             archived development scripts (see its README)
├── src/voxneuro/
│   ├── __init__.py, cli.py
│   ├── method.py                subject views, kernels, G-SMOTE, original variant
│   └── adaptive.py              rank normalization, dimension-adaptive subspace ensemble, comparators
└── tests/
    ├── test_method.py
    └── test_adaptive.py
```

## Tests

After installing the package:

```bash
python -m pip install pytest
pytest -q
```

The suite covers geodesic orthonormality, chordal-distance symmetry and diagonal behavior, an end-to-end subject-level evaluation smoke test of the original variant, the dimensionality gate, the monotone rank normalizer, the shape and coding of the PLS targets, the gate-inactive and gate-active behavior of the adaptive model, and regression checks for identifier collisions, impossible ranks, and empty ensembles. Passing tests establish these checked behaviors; they are not a complete certification of leakage safety.

## Reproducibility notes

- Outer folds are stratified and generated at the subject level.
- The scaler is fit only to recordings from training subjects.
- Kernel bandwidths are estimated only from training representations (augmented ones where G-SMOTE is active).
- Synthetic subjects are created only inside imbalanced training folds.
- Randomized components receive explicit seeds.
- Subject-level out-of-fold predictions make aggregate metrics auditable.
- Dependency versions are lower-bounded in `requirements.txt`; the exact environment used to generate the archived results is pinned in [`requirements-lock.txt`](requirements-lock.txt) (`pip install -r requirements-lock.txt` on Python 3.11).
- Given a fixed dataset, seed, and dependency set, predicted labels and reported metrics reproduce under the documented environment; decision values may differ at floating-point precision across BLAS implementations and hardware. On imbalanced datasets the full-space models additionally depend on the G-SMOTE random draws, so `--seed` must be held fixed. `evaluate_adaptive` runs under single-threaded linear algebra; the archived package was generated with `OPENBLAS_NUM_THREADS=1` on macOS (Apple silicon) under the pinned environment.
- Fold assignments are recoverable from the archived out-of-fold prediction files (fold identifiers per subject); the prepared datasets are not redistributed (see Datasets).

## Paper, project, and citations

**Paper**

Laiba Khan and Manas Waghe, “VoxNeuro: Leakage-Safe Parkinson’s Screening Support via Multi-View Grassmannian Speech Analysis,” manuscript prepared for submission to *Bioengineering* (MDPI), 2026 (unpublished). This entry and `CITATION.cff` will be updated once the article is published. The current manuscript is archived at [`paper/VoxNeuro_MDPI_Bioengineering.pdf`](paper/VoxNeuro_MDPI_Bioengineering.pdf); the superseded initial version (z-standardized full-space model only) is kept as [`paper/archive_VoxNeuro_MDPI_Bioengineering_initial_v1.pdf`](paper/archive_VoxNeuro_MDPI_Bioengineering_initial_v1.pdf).

```bibtex
@unpublished{khan2026voxneuro,
  author = {Khan, Laiba and Waghe, Manas},
  title  = {VoxNeuro: Leakage-Safe Parkinson's Screening Support
            via Multi-View Grassmannian Speech Analysis},
  note   = {Manuscript prepared for submission to Bioengineering (MDPI); unpublished},
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
