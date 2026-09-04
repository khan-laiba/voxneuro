"""Adaptive-model package: every table behind the manuscript's adaptive-subspace results.

Outputs (written to --out):
  adaptive_default_<cohort>_<normalization>/   evaluate_adaptive() tables at the default partition (seed 42)
  adaptive_comparators.csv                     summary of all comparators, both cohorts, both normalizations
  adaptive_partitions.csv                      20 outer partitions (seeds 0-19), both cohorts, key models
  adaptive_dimension_sweep.csv                 subspace dimension sweep (gate forced on), both cohorts, 20 partitions
  adaptive_weight_ablation.csv                 fusion-weight ablation of the adaptive model (w = 0, 0.5, 1; default + 20 partitions)
  adaptive_gsmote_ablation.csv                 G-SMOTE inside the subspace branch (24 seeds, default partition, PD-252)
  adaptive_concentration.csv                   distance-concentration diagnostics before/after the projection

Usage:
  PYTHONPATH=src python scripts/adaptive_package.py --uci <UCI-489 csv> --pd <PD-252 clean csv> --out results/adaptive
"""
from __future__ import annotations

import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from voxneuro.adaptive import (
    DEFAULT_SUBSPACE_DIMS,
    concentration_statistics,
    evaluate_adaptive,
    fit_normalizer,
    fit_supervised_projection,
)
from voxneuro.method import (
    _fit_fused_model,
    _g_smote,
    _subject_rows,
    _validate_frame,
    build_subject_views,
)

warnings.filterwarnings("ignore")
COHORTS: dict[str, dict] = {}


def load(cohort: str, spec: dict | None = None) -> dict:
    spec = COHORTS[cohort] if spec is None else spec
    frame = pd.read_csv(spec["csv"])
    df, feats = _validate_frame(frame, spec["id"], spec["label"], spec["drop"])
    subj = df[[spec["id"], spec["label"]]].drop_duplicates(spec["id"])
    return dict(
        frame=frame, spec=spec, X=df[feats].to_numpy(float), row_ids=df[spec["id"]].to_numpy(str),
        row_labels=df[spec["label"]].to_numpy(int), sids=subj[spec["id"]].to_numpy(str),
        slab=subj[spec["label"]].to_numpy(int), rmap=_subject_rows(df[spec["id"]].to_numpy(str)), feats=feats,
    )


def _summary_rows(result, **keys):
    rows = []
    for _, r in result.summary_metrics.iterrows():
        rows.append(dict(**keys, method=r["method"], balanced_accuracy=r["balanced_accuracy_mean"],
                         balanced_accuracy_se=r["balanced_accuracy_se"], macro_f1=r["macro_f1_mean"], macro_f1_se=r["macro_f1_se"]))
    return rows


def default_partition(out: Path) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        D = load(cohort)
        for normalization in ("rank", "standard"):
            res = evaluate_adaptive(D["frame"], id_col=D["spec"]["id"], label_col=D["spec"]["label"], drop_cols=D["spec"]["drop"],
                                    normalization=normalization, allow_rank_deficient=True, random_state=42)
            res.save(out / f"adaptive_default_{cohort}_{normalization}")
            rows += _summary_rows(res, cohort=cohort, normalization=normalization, partition_seed=42)
            print("default", cohort, normalization, res.summary_metrics.iloc[0].to_dict(), flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "adaptive_comparators.csv", index=False); return df


def _partition_job(args):
    cohort, spec, ps, normalization, gate = args
    D = load(cohort, spec)
    res = evaluate_adaptive(D["frame"], id_col=D["spec"]["id"], label_col=D["spec"]["label"], drop_cols=D["spec"]["drop"],
                            normalization=normalization, gate=gate, allow_rank_deficient=True, random_state=ps, include_members=True)
    return _summary_rows(res, cohort=cohort, normalization=normalization, gate=gate, partition_seed=ps)


def partitions(out: Path, workers: int) -> pd.DataFrame:
    jobs = [(c, COHORTS[c], ps, n, "auto") for c in COHORTS for ps in range(20) for n in ("rank", "standard")]
    jobs += [("UCI-489", COHORTS["UCI-489"], ps, "rank", "on") for ps in range(20)]      # gate forced on: subspace stage applied to the low-dimensional cohort
    rows = []
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_partition_job, jobs), start=1):
            rows += r
            if k % 10 == 0: print("partitions", k, "/", len(jobs), flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "adaptive_partitions.csv", index=False); return df


def _sweep_job(args):
    """Subspace dimension sweep with the gate forced on: single members q and the default ensemble."""
    cohort, spec, ps = args
    D = load(cohort, spec)
    dims = tuple(q for q in (4, 8, 16, 32, 64) if q <= D["X"].shape[1])
    res = evaluate_adaptive(D["frame"], id_col=D["spec"]["id"], label_col=D["spec"]["label"], drop_cols=D["spec"]["drop"],
                            normalization="rank", gate="on", subspace_dims=dims, allow_rank_deficient=True,
                            random_state=ps, include_members=True)
    rows = _summary_rows(res, cohort=cohort, partition_seed=ps, members=",".join(map(str, dims)))
    res2 = evaluate_adaptive(D["frame"], id_col=D["spec"]["id"], label_col=D["spec"]["label"], drop_cols=D["spec"]["drop"],
                             normalization="rank", gate="on", allow_rank_deficient=True, random_state=ps, include_members=False)
    rows += [r for r in _summary_rows(res2, cohort=cohort, partition_seed=ps, members="8,16,32") if r["method"] == "Adaptive_Subspace_Fusion"]
    return rows


def dimension_sweep(out: Path, workers: int) -> pd.DataFrame:
    jobs = [(c, COHORTS[c], ps) for c in COHORTS for ps in range(20)]
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_sweep_job, jobs): rows += r
    df = pd.DataFrame(rows); df.to_csv(out / "adaptive_dimension_sweep.csv", index=False); return df


def _weight_job(args):
    """Fusion-weight ablation of the adaptive model: w = 0 (Euclidean view only) and w = 1 (Grassmann view only)."""
    cohort, spec, ps, w = args
    D = load(cohort, spec)
    res = evaluate_adaptive(D["frame"], id_col=D["spec"]["id"], label_col=D["spec"]["label"], drop_cols=D["spec"]["drop"],
                            normalization="rank", gate="auto", weight=w, allow_rank_deficient=True, random_state=ps, include_members=False)
    return [r for r in _summary_rows(res, cohort=cohort, partition_seed=ps, weight=w)
            if r["method"] in ("Adaptive_Subspace_Fusion", "Fused_Grassmann_GSMOTE")]


def weight_ablation(out: Path, workers: int) -> pd.DataFrame:
    jobs = [(c, COHORTS[c], ps, w) for c in COHORTS for ps in [42] + list(range(20)) for w in (0.0, 0.5, 1.0)]
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_weight_job, jobs): rows += r
    df = pd.DataFrame(rows); df.to_csv(out / "adaptive_weight_ablation.csv", index=False); return df


def gsmote_ablation(out: Path) -> pd.DataFrame:
    """Subspace branch with and without G-SMOTE (PD-252, default partition, 24 augmentation seeds)."""
    D = load("PD-252"); rows = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(splitter.split(D["sids"], D["slab"]))
    for seed_index in range(24):
        acc = {"with_gsmote": [], "without_gsmote": []}
        for fold, (tr, te) in enumerate(folds, start=1):
            train_ids, test_ids = D["sids"][tr], D["sids"][te]
            train_rows = np.concatenate([D["rmap"][s] for s in train_ids]); test_rows = np.concatenate([D["rmap"][s] for s in test_ids])
            normalizer = fit_normalizer(D["X"][train_rows], "rank")
            Xn = np.empty_like(D["X"]); Xn[train_rows] = normalizer.transform(D["X"][train_rows]); Xn[test_rows] = normalizer.transform(D["X"][test_rows])
            y_test = D["slab"][te]; scores = {"with_gsmote": [], "without_gsmote": []}
            for q in DEFAULT_SUBSPACE_DIMS:
                proj = fit_supervised_projection(Xn[train_rows], D["row_labels"][train_rows], q)
                Z = np.empty((len(Xn), q)); Z[train_rows] = proj.transform(Xn[train_rows]); Z[test_rows] = proj.transform(Xn[test_rows])
                Q_tr, s_tr, y_tr = build_subject_views(Z, D["row_ids"], D["row_labels"], train_ids, 3, True)
                Q_te, s_te, _ = build_subject_views(Z, D["row_ids"], D["row_labels"], test_ids, 3, True)
                for name, (Qa, sa, ya) in (("without_gsmote", (Q_tr, s_tr, y_tr)),
                                            ("with_gsmote", _g_smote(Q_tr, s_tr, y_tr, random_state=(1000 + seed_index) * 1000 + fold, neighbors=5)[:3])):
                    model = _fit_fused_model(Qa, sa, ya, weight=0.5, C=1.0)
                    scale = float(np.std(model.decision_function(Qa, sa))) + 1e-12
                    scores[name].append(model.decision_function(Q_te, s_te) / scale)
            for name in acc:
                pred = (np.mean(scores[name], axis=0) > 0).astype(int)
                acc[name].append((balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")))
        rows.append(dict(seed_index=seed_index, **{f"{n}_ba": np.mean([v[0] for v in a]) for n, a in acc.items()},
                         **{f"{n}_f1": np.mean([v[1] for v in a]) for n, a in acc.items()}))
        print("gsmote ablation seed", seed_index, rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "adaptive_gsmote_ablation.csv", index=False); return df


def concentration(out: Path) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        D = load(cohort)
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D["sids"], D["slab"]), start=1):
            train_ids = D["sids"][tr]; train_rows = np.concatenate([D["rmap"][s] for s in train_ids])
            for normalization in ("standard", "rank"):
                normalizer = fit_normalizer(D["X"][train_rows], normalization)
                Xn = np.empty_like(D["X"]); Xn[train_rows] = normalizer.transform(D["X"][train_rows])
                Q, s, _ = build_subject_views(Xn, D["row_ids"], D["row_labels"], train_ids, 3, True)
                rows.append(dict(cohort=cohort, fold=fold, normalization=normalization, space="full", n_features=D["X"].shape[1],
                                 n_train_recordings=len(train_rows), **concentration_statistics(Q, s)))
                if normalization == "rank":
                    for q in DEFAULT_SUBSPACE_DIMS:
                        proj = fit_supervised_projection(Xn[train_rows], D["row_labels"][train_rows], q)
                        Z = np.empty((len(Xn), q)); Z[train_rows] = proj.transform(Xn[train_rows])
                        Qz, sz, _ = build_subject_views(Z, D["row_ids"], D["row_labels"], train_ids, 3, True)
                        rows.append(dict(cohort=cohort, fold=fold, normalization=normalization, space=f"subspace_q{q}", n_features=D["X"].shape[1],
                                         n_train_recordings=len(train_rows), **concentration_statistics(Qz, sz)))
    df = pd.DataFrame(rows); df.to_csv(out / "adaptive_concentration.csv", index=False); return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", required=True); ap.add_argument("--pd", required=True); ap.add_argument("--out", default="results/adaptive")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: default partitions sweep gsmote concentration")
    args = ap.parse_args()
    COHORTS["UCI-489"] = dict(csv=args.uci, id="ID", label="Status", drop=("Recording",))
    COHORTS["PD-252"] = dict(csv=args.pd, id="id", label="class", drop=())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    steps = args.only or ["default", "concentration", "partitions", "sweep", "weights", "gsmote"]
    if "default" in steps: default_partition(out)
    if "concentration" in steps: concentration(out)
    if "partitions" in steps: partitions(out, args.workers)
    if "sweep" in steps: dimension_sweep(out, args.workers)
    if "weights" in steps: weight_ablation(out, args.workers)
    if "gsmote" in steps: gsmote_ablation(out)
    print("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
