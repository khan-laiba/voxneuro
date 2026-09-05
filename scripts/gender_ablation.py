"""Gender-covariate ablation of the manuscript's model under rank normalization.

The dataset-provided binary gender covariate is mapped by the rank normalizer to two values near +-5.2 and
therefore carries a larger weight in the Euclidean view than a continuous coordinate. This script quantifies
its influence by re-running the adaptive model and its comparators with the covariate removed from the feature
set, at the reference partition (seed 42) and across the 20 alternative partitions (seeds 0-19), and pairing
every run with the matching run that keeps the covariate (identical folds, normalizer refitted per fold).

Output (--out): adaptive_gender_ablation.csv, one row per (cohort, partition_seed, variant, method).

Usage:
  PYTHONPATH=src python scripts/gender_ablation.py --uci <UCI-489 csv> --pd <PD-252 clean csv> --out results/adaptive
"""
from __future__ import annotations

import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from voxneuro.adaptive import evaluate_adaptive

warnings.filterwarnings("ignore")
COHORTS: dict[str, dict] = {}
VARIANTS = ("with_gender", "without_gender")


def _job(args):
    cohort, spec, seed, variant = args
    frame = pd.read_csv(spec["csv"])
    drop = tuple(spec["drop"]) + ((spec["gender"],) if variant == "without_gender" else ())
    with threadpool_limits(limits=1):
        res = evaluate_adaptive(frame, id_col=spec["id"], label_col=spec["label"], drop_cols=drop, normalization="rank",
                                gate="auto", allow_rank_deficient=True, random_state=seed, include_members=False)
    rows = []
    for _, r in res.summary_metrics.iterrows():
        rows.append(dict(cohort=cohort, partition_seed=seed, variant=variant, method=r["method"],
                         balanced_accuracy=r["balanced_accuracy_mean"], balanced_accuracy_se=r["balanced_accuracy_se"],
                         macro_f1=r["macro_f1_mean"], macro_f1_se=r["macro_f1_se"]))
    return rows


def summarize(df: pd.DataFrame, method: str = "Adaptive_Subspace_Fusion") -> None:
    m = df[df.method == method].pivot_table(index=["cohort", "partition_seed"], columns="variant",
                                            values=["balanced_accuracy", "macro_f1"])
    for cohort in sorted(df.cohort.unique()):
        sub = m.loc[cohort]
        for metric in ("balanced_accuracy", "macro_f1"):
            d = sub[(metric, "without_gender")] - sub[(metric, "with_gender")]
            ref, alt = d.loc[42], d.drop(42)
            print(f"{method} {cohort} {metric}: reference {sub[(metric, 'with_gender')].loc[42]:.4f} -> "
                  f"{sub[(metric, 'without_gender')].loc[42]:.4f} (delta {ref:+.4f}); 20 partitions: without "
                  f"{sub[(metric, 'without_gender')].drop(42).mean():.4f} +- {sub[(metric, 'without_gender')].drop(42).std(ddof=1):.4f}, "
                  f"delta {alt.mean():+.4f} +- {alt.std(ddof=1):.4f}, positive {(alt > 0).sum()}/20, ties {(alt == 0).sum()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", required=True); ap.add_argument("--pd", required=True)
    ap.add_argument("--out", default="results/adaptive"); ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    COHORTS["UCI-489"] = dict(csv=args.uci, id="ID", label="Status", drop=("Recording",), gender="Gender")
    COHORTS["PD-252"] = dict(csv=args.pd, id="id", label="class", drop=(), gender="gender")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    jobs = [(c, COHORTS[c], seed, v) for c in COHORTS for seed in [42] + list(range(20)) for v in VARIANTS]
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs), start=1):
            rows += r
            if k % 6 == 0:
                print("gender ablation", k, "/", len(jobs), "%.0fs" % (time.time() - t0), flush=True)
    df = pd.DataFrame(rows).sort_values(["cohort", "partition_seed", "variant", "method"]).reset_index(drop=True)
    df.to_csv(out / "adaptive_gender_ablation.csv", index=False)
    summarize(df)
    print("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
