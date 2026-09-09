"""Exploration 2: PCA-reduced Grassmann views, rank-1 direction kernel, late (decision-level) fusion,
block-balanced Grassmann view (PD-252), and C sensitivity. Default partition (seed 42) and default augmentation seeds."""
import sys as _sys
if "--help" in _sys.argv or "-h" in _sys.argv:
    print(__doc__); print("\nRun from the repository root with data/ (UCI datasets) and robustness_out/ present; no arguments are required."); _sys.exit(0)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
import voxneuro.method as M

def load(name):
    if name == "UCI-489":
        frame, feats = M._validate_frame(pd.read_csv("data/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"), "ID", "Status", ("Recording",)); idc, labc = "ID", "Status"
    else:
        frame, feats = M._validate_frame(pd.read_csv("data/pd_speech_features_clean.csv"), "id", "class", ()); idc, labc = "id", "class"
    X = frame[feats].to_numpy(float); rid = frame[idc].to_numpy(str); rl = frame[labc].to_numpy(int); sf = frame[[idc, labc]].drop_duplicates(idc)
    return dict(X=X, rid=rid, rl=rl, sids=sf[idc].to_numpy(str), slab=sf[labc].to_numpy(int), rmap=M._subject_rows(rid), feats=feats)

def family(c):
    c = c.lower()
    if c == "gender": return "Demographics"
    if c.startswith("tqwt"): return "TQWT"
    if "mfcc" in c or "delta" in c or "log_energy" in c: return "MFCC"
    if c.startswith("imf"): return "IMF/EMD"
    if c.startswith(("det_", "app_", "ea", "ed")): return "Wavelet"
    return "Vocal/time-frequency"

def gram_views(Xg, rid, rl, sids, rank):
    """Grassmann view from (possibly transformed) recording matrix Xg; Euclidean view is computed separately."""
    Q = []
    for sid in sids:
        idx = np.flatnonzero(rid == sid); U, s, _ = np.linalg.svd(Xg[idx].T, full_matrices=False); Q.append(U[:, :rank])
    return Q

def fit_kernels(Qtr, s_tr, ytr, Qte, s_te, seed):
    Qa, sa, ya, _ = M._g_smote(Qtr, s_tr, ytr, random_state=seed, neighbors=5)
    dg, de = M.chordal_squared(Qa), M.euclidean_squared(sa); gg, ge = M._median_gamma(dg), M._median_gamma(de)
    return dict(Kg=np.exp(-gg * dg), Ke=np.exp(-ge * de), Kg_te=np.exp(-gg * M.chordal_squared(Qte, Qa)), Ke_te=np.exp(-ge * M.euclidean_squared(s_te, sa)), ya=ya)

def early(K, w, C=1.0):
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(w * K["Kg"] + (1 - w) * K["Ke"], K["ya"])
    return clf.decision_function(w * K["Kg_te"] + (1 - w) * K["Ke_te"])

def late(K, lam, C=1.0):
    fg = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K["Kg"], K["ya"]); fe = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K["Ke"], K["ya"])
    dg_tr, de_tr = fg.decision_function(K["Kg"]), fe.decision_function(K["Ke"])   # training-set scale for standardization
    sg, se = dg_tr.std() + 1e-12, de_tr.std() + 1e-12
    return lam * fg.decision_function(K["Kg_te"]) / sg + (1 - lam) * fe.decision_function(K["Ke_te"]) / se

def score(yte, dec): yp = (dec > 0).astype(int); return balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")

rows = []
for cohort in ("UCI-489", "PD-252"):
    D = load(cohort); fam = np.array([family(c) for c in D["feats"]])
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]; trr = np.concatenate([D["rmap"][s] for s in tid]); ter = np.concatenate([D["rmap"][s] for s in eid])
        sc = StandardScaler().fit(D["X"][trr]); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        seed = 42 * 1000 + fold
        Qtr, s_tr, ytr = M.build_subject_views(Xs, D["rid"], D["rl"], tid, 3, True); Qte, s_te, yte = M.build_subject_views(Xs, D["rid"], D["rl"], eid, 3, True)
        def add(variant, dec): ba, f1 = score(yte, dec); rows.append(dict(cohort=cohort, fold=fold, variant=variant, ba=ba, f1=f1))
        K = fit_kernels(Qtr, s_tr, ytr, Qte, s_te, seed)
        add("early w=0.5 (paper)", early(K, 0.5)); add("Euclidean RBF w=0", early(K, 0.0))
        for C in (0.3, 3.0, 10.0): add("early w=0.5 C=%g" % C, early(K, 0.5, C)); add("Euclidean RBF C=%g" % C, early(K, 0.0, C))
        for lam in (0.2, 0.3, 0.5): add("late lam=%.1f" % lam, late(K, lam))
        # rank-1 direction kernel on full features
        Q1tr = gram_views(Xs, D["rid"], D["rl"], tid, 1); Q1te = gram_views(Xs, D["rid"], D["rl"], eid, 1)
        K1 = fit_kernels(Q1tr, s_tr, ytr, Q1te, s_te, seed)
        add("rank-1 Grassmann only", early(K1, 1.0)); add("rank-1 early w=0.5", early(K1, 0.5)); add("rank-1 early w=0.3", early(K1, 0.3)); add("rank-1 late lam=0.3", late(K1, 0.3))
        # PCA-reduced Grassmann view (PCA fit on training-fold recordings)
        for p in (4, 8, 16, 32):
            pca = PCA(n_components=p, random_state=0).fit(Xs[trr]); Xp = pca.transform(Xs)
            Qptr = gram_views(Xp, D["rid"], D["rl"], tid, 3); Qpte = gram_views(Xp, D["rid"], D["rl"], eid, 3)
            Kp = fit_kernels(Qptr, s_tr, ytr, Qpte, s_te, seed)
            add("PCA%d Grassmann only" % p, early(Kp, 1.0))
            for w in (0.3, 0.5): add("PCA%d early w=%.1f" % (p, w), early(Kp, w))
            add("PCA%d late lam=0.3" % p, late(Kp, 0.3)); add("PCA%d late lam=0.5" % p, late(Kp, 0.5))
        # block-balanced Grassmann view (each family scaled to equal total variance), PD-252 only
        if cohort == "PD-252":
            Xb = Xs.copy()
            for g in set(fam):
                cols = np.flatnonzero(fam == g); Xb[:, cols] /= np.sqrt(len(cols))
            Qbtr = gram_views(Xb, D["rid"], D["rl"], tid, 3); Qbte = gram_views(Xb, D["rid"], D["rl"], eid, 3)
            Kb = fit_kernels(Qbtr, s_tr, ytr, Qbte, s_te, seed)
            add("block-balanced Grassmann only", early(Kb, 1.0)); add("block-balanced early w=0.5", early(Kb, 0.5)); add("block-balanced early w=0.3", early(Kb, 0.3)); add("block-balanced late lam=0.3", late(Kb, 0.3))
        print(cohort, "fold", fold, flush=True)
df = pd.DataFrame(rows); df.to_csv("robustness_out/fusion_exploration2.csv", index=False)
pd.set_option("display.width", 200)
for cohort in ("UCI-489", "PD-252"):
    d = df[df.cohort == cohort].groupby("variant")[["ba", "f1"]].agg(["mean"]); d.columns = ["ba", "f1"]
    print("\n==", cohort); print(d.sort_values("ba", ascending=False).round(3).to_string())
print("DONE")
