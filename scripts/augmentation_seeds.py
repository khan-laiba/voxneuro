"""Paired comparison of augmentation families across the 24 archived G-SMOTE seeds (PD-252, fixed folds):
ordinary SMOTE + Euclidean RBF versus G-SMOTE + Euclidean RBF (the matched ablation) versus the fused model."""
import sys as _sys
if "--help" in _sys.argv or "-h" in _sys.argv:
    print(__doc__); print("\nRun from the repository root with data/ (UCI datasets) and robustness_out/ present; no arguments are required."); _sys.exit(0)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import voxneuro.method as M
frame, feats = M._validate_frame(pd.read_csv("data/pd_speech_features_clean.csv"), "id", "class", ())
X = frame[feats].to_numpy(float); rid = frame["id"].to_numpy(str); rl = frame["class"].to_numpy(int)
sf = frame[["id", "class"]].drop_duplicates("id"); sids = sf["id"].to_numpy(str); slab = sf["class"].to_numpy(int); rmap = M._subject_rows(rid)
F = []
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(sids, slab), start=1):
    tid, eid = sids[tr], sids[te]; trr = np.concatenate([rmap[s] for s in tid]); ter = np.concatenate([rmap[s] for s in eid])
    sc = StandardScaler().fit(X[trr]); Xs = np.empty_like(X); Xs[trr] = sc.transform(X[trr]); Xs[ter] = sc.transform(X[ter])
    F.append((fold, M.build_subject_views(Xs, rid, rl, tid, 3, True), M.build_subject_views(Xs, rid, rl, eid, 3, True)))
rows = []
for c in range(24):
    seed = lambda fold: (1000 + c) * 1000 + fold
    sc_ = {"fused": [], "gsmote_rbf": [], "smote_rbf": []}
    for fold, (Qtr, s_tr, ytr), (Qte, s_te, yte) in F:
        Qa, sa, ya, _ = M._g_smote(Qtr, s_tr, ytr, random_state=seed(fold), neighbors=5)
        dg, de = M.chordal_squared(Qa), M.euclidean_squared(sa); gg, ge = M._median_gamma(dg), M._median_gamma(de)
        Kg_te = np.exp(-gg * M.chordal_squared(Qte, Qa)); Ke_te = np.exp(-ge * M.euclidean_squared(s_te, sa))
        for name, w in (("fused", 0.5), ("gsmote_rbf", 0.0)):
            clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(w * np.exp(-gg * dg) + (1 - w) * np.exp(-ge * de), ya)
            yp = (clf.decision_function(w * Kg_te + (1 - w) * Ke_te) > 0).astype(int)
            sc_[name].append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
        counts = np.bincount(ytr); s_rs, y_rs = SMOTE(random_state=seed(fold), k_neighbors=min(5, int(counts.min()) - 1)).fit_resample(s_tr, ytr)
        de2 = M.euclidean_squared(s_rs); ge2 = M._median_gamma(de2)
        clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(np.exp(-ge2 * de2), y_rs)
        yp = (clf.decision_function(np.exp(-ge2 * M.euclidean_squared(s_te, s_rs))) > 0).astype(int)
        sc_["smote_rbf"].append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
    row = dict(seed_tag="s%02d" % c, seed_formula="(1000+%d)*1000+fold" % c)
    for name, v in sc_.items():
        a = np.asarray(v); row[name + "_ba"] = a[:, 0].mean(); row[name + "_f1"] = a[:, 1].mean()
    rows.append(row); print(c, {k: round(row[k], 4) for k in row if k.endswith("_f1")}, flush=True)
pd.DataFrame(rows).to_csv("robustness_out/augmentation_seed_study.csv", index=False); print("DONE")
