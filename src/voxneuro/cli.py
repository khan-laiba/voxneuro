"""Command-line interface for the minimal VoxNeuro method package."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .method import evaluate_repeated_measurements


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe subject-level VoxNeuro evaluation on a CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to the repeated-measurement CSV.")
    parser.add_argument("--id-col", required=True, help="Subject identifier column.")
    parser.add_argument("--label-col", required=True, help="Binary label column.")
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=[],
        help="Optional non-feature columns to exclude.",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory.")
    parser.add_argument("--rank", type=int, default=3, help="Fixed Grassmann rank.")
    parser.add_argument("--splits", type=int, default=5, help="Outer subject-level folds.")
    parser.add_argument("--weight", type=float, default=0.5, help="Grassmann-kernel weight.")
    parser.add_argument("--C", type=float, default=1.0, help="SVM penalty.")
    parser.add_argument("--gsmote-neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    frame = pd.read_csv(args.csv)
    result = evaluate_repeated_measurements(
        frame,
        id_col=args.id_col,
        label_col=args.label_col,
        drop_cols=args.drop_cols,
        rank=args.rank,
        n_splits=args.splits,
        weight=args.weight,
        C=args.C,
        g_smote_neighbors=args.gsmote_neighbors,
        random_state=args.seed,
    )
    result.save(Path(args.output_dir))
    print(result.summary_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
