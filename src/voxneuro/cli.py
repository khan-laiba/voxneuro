"""Command-line interface for the minimal VoxNeuro method package."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .adaptive import DEFAULT_SUBSPACE_DIMS, evaluate_adaptive
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
    parser.add_argument(
        "--model",
        choices=["adaptive", "original"],
        default="adaptive",
        help=(
            "'adaptive' (default) runs the rank-normalized, dimension-adaptive "
            "multi-view model with its comparators; 'original' runs the "
            "z-standardized fused model of the original submission."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=["rank", "standard"],
        default="rank",
        help="Recording-level normalization for --model adaptive (fitted on training folds only).",
    )
    parser.add_argument(
        "--subspace-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUBSPACE_DIMS),
        help="Supervised-subspace dimensions of the ensemble members (adaptive model).",
    )
    parser.add_argument(
        "--gate",
        choices=["auto", "on", "off"],
        default="auto",
        help="Supervised-subspace gate: 'auto' activates it when the feature count exceeds the training recordings.",
    )
    parser.add_argument(
        "--allow-rank-deficient",
        action="store_true",
        help=(
            "Accept subjects whose recording matrix has numerical rank below "
            "--rank, retaining the first r left-singular vectors with a "
            "warning. Required to run the paper's rank-3 configuration on the "
            "official UCI files, which contain byte-identical repeated "
            "recordings for some subjects."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    frame = pd.read_csv(args.csv)
    common = dict(
        id_col=args.id_col,
        label_col=args.label_col,
        drop_cols=args.drop_cols,
        rank=args.rank,
        allow_rank_deficient=args.allow_rank_deficient,
        n_splits=args.splits,
        weight=args.weight,
        C=args.C,
        g_smote_neighbors=args.gsmote_neighbors,
        random_state=args.seed,
    )
    if args.model == "adaptive":
        result = evaluate_adaptive(
            frame,
            normalization=args.normalization,
            subspace_dims=tuple(args.subspace_dims),
            gate=args.gate,
            **common,
        )
    else:
        result = evaluate_repeated_measurements(frame, **common)
    result.save(Path(args.output_dir))
    print(result.summary_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
