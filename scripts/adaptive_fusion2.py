"""Adaptive fusion with nested selection (repeated 3 x 5 inner CV on the training subjects, balanced accuracy):
  early-adaptive: w in {0, 0.1, ..., 0.8} for K = w K_G + (1-w) K_E
  late-adaptive : lambda in {0, 0.1, ..., 0.5} for standardized decision-value fusion of per-view SVMs
Compared with fixed w = 0 (Euclidean RBF ablation) and w = 0.5 (paper) on the same held-out folds.
--study partition: 20 outer partitions (both cohorts), default augmentation seeds; --study seed: 24 augmentation seeds (PD-252, fixed folds)."""
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
import voxneuro.method as M
WG = np.round(np.arange(0.0, 0.81, 0.1), 1); LG = np.round(np.arange(0.0, 0.51, 0.1), 1)
def load(name):
    if name == "UCI-489":
        frame, feats = M._validate_frame(pd.read_csv("data/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"), "ID", "Status", ("Recording",)); idc, labc = "ID", "Status"
    else:
        frame, feats = M._validate_frame(pd.read_csv("data/pd_speech_features_clean.csv"), "id", "class", ()); idc, labc = "id", "class"
    X = frame[feats].to_numpy(float); rid = frame[idc].to_numpy(str); rl = frame[labc].to_numpy(int); sf = frame[[idc, labc]].drop_duplicates(idc)
    return dict(X=X, rid=rid, rl=rl, sids=sf[idc].to_numpy(str), slab=sf[labc].to_numpy(int), rmap=M._subject_rows(rid))
def kernels(Qtr, s_tr, ytr, Qte, s_te, seed):
    Qa, sa, ya, _ = M._g_smote(Qtr, s_tr, ytr, random_state=seed, neighbors=5)
    dg, de = M.chordal_squared(Qa), M.euclidean_squared(sa); gg, ge = M._median_gamma(dg), M._median_gamma(de)
    return dict(Kg=np.exp(-gg * dg), Ke=np.exp(-ge * de), Kg_te=np.exp(-gg * M.chordal_squared(Qte, Qa)), Ke_te=np.exp(-ge * M.euclidean_squared(s_te, sa)), ya=ya)
def early_dec(K, w):
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(w * K["Kg"] + (1 - w) * K["Ke"], K["ya"]); return clf.decision_function(w * K["Kg_te"] + (1 - w) * K["Ke_te"])
def late_decs(K):
    fg = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(K["Kg"], K["ya"]); fe = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(K["Ke"], K["ya"])
    sg, se = fg.decision_function(K["Kg"]).std() + 1e-12, fe.decision_function(K["Ke"]).std() + 1e-12
    dg, de = fg.decision_function(K["Kg_te"]) / sg, fe.decision_function(K["Ke_te"]) / se
    return {lam: lam * dg + (1 - lam) * de for lam in LG}
def select(Qtr, s_tr, ytr, base_seed):
    se_, sl_ = {w: [] for w in WG}, {l: [] for l in LG}
    for rep in range(3):
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(np.zeros(len(ytr)), ytr), start=1):
            K = kernels([Qtr[i] for i in a], s_tr[a], ytr[a], [Qtr[i] for i in b], s_tr[b], base_seed * 100 + rep * 10 + k)
            for w in WG: se_[w].append(balanced_accuracy_score(ytr[b], (early_dec(K, w) > 0).astype(int)))
            ld = late_decs(K)
            for l in LG: sl_[l].append(balanced_accuracy_score(ytr[b], (ld[l] > 0).astype(int)))
    me, ml = {w: np.mean(v) for w, v in se_.items()}, {l: np.mean(v) for l, v in sl_.items()}
    return max(WG, key=lambda w: (round(me[w], 6), -w)), max(LG, key=lambda l: (round(ml[l], 6), -l))   # ties -> smaller geometric weight
def evaluate(F, seed_of_fold, tag):
    res = {k: [] for k in ("early_adaptive", "late_adaptive", "w0", "w05")}; chosen = []
    for fold, (Qtr, s_tr, ytr), (Qte, s_te, yte) in F:
        seed = seed_of_fold(fold); w_sel, l_sel = select(Qtr, s_tr, ytr, seed)
        K = kernels(Qtr, s_tr, ytr, Qte, s_te, seed); ld = late_decs(K)
        for name, dec in (("early_adaptive", early_dec(K, w_sel)), ("late_adaptive", ld[l_sel]), ("w0", early_dec(K, 0.0)), ("w05", early_dec(K, 0.5))):
            yp = (dec > 0).astype(int); res[name].append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
        chosen.append((w_sel, l_sel))
    row = dict(tag=tag, chosen=str(chosen))
    for name, v in res.items(): a = np.asarray(v); row[name + "_ba"] = a[:, 0].mean(); row[name + "_f1"] = a[:, 1].mean()
    return row
def folds(D, seed):
    out = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]; trr = np.concatenate([D["rmap"][s] for s in tid]); ter = np.concatenate([D["rmap"][s] for s in eid])
        sc = StandardScaler().fit(D["X"][trr]); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        out.append((fold, M.build_subject_views(Xs, D["rid"], D["rl"], tid, 3, True), M.build_subject_views(Xs, D["rid"], D["rl"], eid, 3, True)))
    return out
ap = argparse.ArgumentParser(); ap.add_argument("--study", choices=["partition", "seed"], required=True); a = ap.parse_args()
rows = []
if a.study == "partition":
    for cohort in ("UCI-489", "PD-252"):
        D = load(cohort)
        for ps in range(20):
            r = evaluate(folds(D, ps), lambda f: 42 * 1000 + f, "%s|partition|%d" % (cohort, ps)); rows.append(r)
            print(r["tag"], {k: round(r[k], 3) for k in r if k.endswith(("_ba", "_f1"))}, r["chosen"], flush=True)
            pd.DataFrame(rows).to_csv("robustness_out/adaptive_partition.csv", index=False)
else:
    D = load("PD-252"); F = folds(D, 42)
    for c in range(24):
        r = evaluate(F, lambda f, c=c: (1000 + c) * 1000 + f, "PD-252|seed|%d" % c); rows.append(r)
        print(r["tag"], {k: round(r[k], 3) for k in r if k.endswith(("_ba", "_f1"))}, r["chosen"], flush=True)
        pd.DataFrame(rows).to_csv("robustness_out/adaptive_seed.csv", index=False)
print("DONE")
