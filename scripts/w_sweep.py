"""Fixed fusion-weight sweep across the 24 augmentation seeds (PD-252, fixed folds) and 20 outer partitions (both cohorts)."""
import sys as _sys
if "--help" in _sys.argv or "-h" in _sys.argv:
    print(__doc__); print("\nRun from the repository root with data/ (UCI datasets) and robustness_out/ present; no arguments are required."); _sys.exit(0)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
import voxneuro.method as M
GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
def load(name):
    if name == "UCI-489":
        frame, feats = M._validate_frame(pd.read_csv("data/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"), "ID", "Status", ("Recording",)); idc, labc = "ID", "Status"
    else:
        frame, feats = M._validate_frame(pd.read_csv("data/pd_speech_features_clean.csv"), "id", "class", ()); idc, labc = "id", "class"
    X = frame[feats].to_numpy(float); rid = frame[idc].to_numpy(str); rl = frame[labc].to_numpy(int); sf = frame[[idc, labc]].drop_duplicates(idc)
    return dict(X=X, rid=rid, rl=rl, sids=sf[idc].to_numpy(str), slab=sf[labc].to_numpy(int), rmap=M._subject_rows(rid))
def folds(D, seed):
    out = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]; trr = np.concatenate([D["rmap"][s] for s in tid]); ter = np.concatenate([D["rmap"][s] for s in eid])
        sc = StandardScaler().fit(D["X"][trr]); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        out.append((fold, M.build_subject_views(Xs, D["rid"], D["rl"], tid, 3, True), M.build_subject_views(Xs, D["rid"], D["rl"], eid, 3, True)))
    return out
def sweep(F, aug):
    res = {w: [] for w in GRID}
    for fold, (Qtr, s_tr, ytr), (Qte, s_te, yte) in F:
        Qa, sa, ya, _ = M._g_smote(Qtr, s_tr, ytr, random_state=aug(fold), neighbors=5)
        dg, de = M.chordal_squared(Qa), M.euclidean_squared(sa); gg, ge = M._median_gamma(dg), M._median_gamma(de)
        Kg, Ke = np.exp(-gg * dg), np.exp(-ge * de); Kg_te, Ke_te = np.exp(-gg * M.chordal_squared(Qte, Qa)), np.exp(-ge * M.euclidean_squared(s_te, sa))
        for w in GRID:
            clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(w * Kg + (1 - w) * Ke, ya)
            yp = (clf.decision_function(w * Kg_te + (1 - w) * Ke_te) > 0).astype(int)
            res[w].append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
    return {w: (np.mean([v[0] for v in r]), np.mean([v[1] for v in r])) for w, r in res.items()}
rows = []
D = load("PD-252"); F = folds(D, 42)
for c in range(24):
    r = sweep(F, lambda fold, c=c: (1000 + c) * 1000 + fold)
    for w, (ba, f1) in r.items(): rows.append(dict(study="seed", cohort="PD-252", index=c, w=w, ba=ba, f1=f1))
    print("seed", c, flush=True)
for cohort in ("UCI-489", "PD-252"):
    D = load(cohort)
    for ps in range(20):
        r = sweep(folds(D, ps), lambda fold: 42 * 1000 + fold)
        for w, (ba, f1) in r.items(): rows.append(dict(study="partition", cohort=cohort, index=ps, w=w, ba=ba, f1=f1))
    print(cohort, "partitions", flush=True)
pd.DataFrame(rows).to_csv("robustness_out/w_sweep.csv", index=False); print("DONE")
