"""Paired G-SMOTE seed-sensitivity study for PD-252.

Re-evaluates the fused Grassmann-Euclidean model and its matched Euclidean
RBF ablation (the same pipeline with fusion weight 0) across distinct
G-SMOTE augmentation seeds, holding the outer folds, preprocessing, and all
non-augmentation settings fixed. Within each seed and fold both models use
the same augmented training set, so their difference is paired at the seed
level. Results are deterministic for a fixed dataset, seed list, and
dependency set (see requirements-lock.txt).

Usage:
    python scripts/seed_sensitivity.py --csv pd_speech_features_clean.csv \
        --id-col id --label-col class --seeds 24 --output pd252_seed_study.csv
"""
from __future__ import annotations

import argparse
import csv
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from voxneuro import method as M


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--output", default="seed_study.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    frame, feats = M._validate_frame(pd.read_csv(args.csv), args.id_col, args.label_col, ())
    X = frame[feats].to_numpy(float)
    row_ids = frame[args.id_col].to_numpy(str)
    row_labels = frame[args.label_col].to_numpy(int)
    subject_frame = frame[[args.id_col, args.label_col]].drop_duplicates(args.id_col)
    sids = subject_frame[args.id_col].to_numpy(str)
    slab = subject_frame[args.label_col].to_numpy(int)
    rmap = M._subject_rows(row_ids)

    folds = []
    splitter = StratifiedKFold(args.splits, shuffle=True, random_state=args.fold_seed)
    for tr, te in splitter.split(sids, slab):
        tid, eid = sids[tr], sids[te]
        trr = np.concatenate([rmap[s] for s in tid])
        ter = np.concatenate([rmap[s] for s in eid])
        sc = StandardScaler().fit(X[trr])
        Xs = np.empty_like(X)
        Xs[trr] = sc.transform(X[trr])
        Xs[ter] = sc.transform(X[ter])
        folds.append(
            (
                M.build_subject_views(Xs, row_ids, row_labels, tid, args.rank, True),
                M.build_subject_views(Xs, row_ids, row_labels, eid, args.rank, True),
            )
        )

    def evaluate(seed_of_fold):
        scores = {"fused": [], "ablation": []}
        for fold, ((Qtr, strn, ytr), (Qte, ste, yte)) in enumerate(folds, start=1):
            Qa, sa, ya, _ = M._g_smote(Qtr, strn, ytr, random_state=seed_of_fold(fold), neighbors=5)
            dg, de = M.chordal_squared(Qa), M.euclidean_squared(sa)
            gg, ge = M._median_gamma(dg), M._median_gamma(de)
            Kg_te = np.exp(-gg * M.chordal_squared(Qte, Qa))
            Ke_te = np.exp(-ge * M.euclidean_squared(ste, sa))
            for name, w in (("fused", args.weight), ("ablation", 0.0)):
                K = w * np.exp(-gg * dg) + (1 - w) * np.exp(-ge * de)
                clf = SVC(kernel="precomputed", C=args.C, class_weight="balanced").fit(K, ya)
                yp = (clf.decision_function(w * Kg_te + (1 - w) * Ke_te) > 0).astype(int)
                scores[name].append(
                    (balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro"))
                )
        out = {}
        for name, vals in scores.items():
            arr = np.asarray(vals)
            out[name] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
        return out

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "seed_tag",
                "seed_formula",
                "fused_balanced_accuracy",
                "fused_macro_f1",
                "ablation_balanced_accuracy",
                "ablation_macro_f1",
                "delta_balanced_accuracy",
                "delta_macro_f1",
            ]
        )
        for c in range(args.seeds):
            res = evaluate(lambda fold, c=c: (1000 + c) * 1000 + fold)
            fb, ff = res["fused"]
            ab, af = res["ablation"]
            writer.writerow(
                [
                    "s%02d" % c,
                    "(1000+%d)*1000+fold" % c,
                    round(fb, 4),
                    round(ff, 4),
                    round(ab, 4),
                    round(af, 4),
                    round(fb - ab, 4),
                    round(ff - af, 4),
                ]
            )
            print("s%02d done" % c, flush=True)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
