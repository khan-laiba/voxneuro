"""Exploration 3 (new methods), default partition (seed 42) + default augmentation seeds, both cohorts.
Candidates: order-statistics view; distribution (MMD / mean-embedding) subject kernel with RKHS-mixture augmentation;
family-wise MKL with centered-alignment weights (PD-252); record-level training + subject aggregation; tree ensembles;
univariate feature selection; simple fusions/stacking of the above with the paper's fused kernel."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
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

def rbf_med(D2):
    v = D2[np.triu_indices(D2.shape[0], 1)] if D2.shape[0] == D2.shape[1] else D2.ravel()
    v = v[np.isfinite(v) & (v > 0)]; return 1.0 / (2.0 * np.median(v) + 1e-12)

def gsmote_pairs(Q_min, s_min, seed, k=5):
    """Replicate the package's G-SMOTE draw sequence to obtain (anchor, partner, t) triples for shared augmentation."""
    d = M.chordal_squared(Q_min); np.fill_diagonal(d, np.inf); kk = min(k, len(Q_min) - 1); nearest = np.argsort(d, axis=1)[:, :kk]
    rng = np.random.default_rng(seed); return rng, nearest

def score(y, dec): yp = (dec > 0).astype(int); return balanced_accuracy_score(y, yp), f1_score(y, yp, average="macro")

def svc_pre(K, y, Kte, C=1.0):
    return SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K, y).decision_function(Kte)

def run(cohort):
    D = load(cohort); fam = np.array([family(c) for c in D["feats"]]); rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]; trr = np.concatenate([D["rmap"][s] for s in tid]); ter = np.concatenate([D["rmap"][s] for s in eid])
        sc = StandardScaler().fit(D["X"][trr]); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        seed = 42 * 1000 + fold
        # ---- subject-level views
        def views(sids):
            Q, S2, S3, R = [], [], [], []
            for sid in sids:
                idx = np.flatnonzero(D["rid"] == sid); Xi = Xs[idx]
                U, _, _ = np.linalg.svd(Xi.T, full_matrices=False); Q.append(U[:, :3])
                S2.append(np.concatenate([Xi.mean(0), Xi.std(0)])); S3.append(np.sort(Xi, axis=0).ravel(order="F")); R.append(Xi)
            y = np.array([int(D["rl"][np.flatnonzero(D["rid"] == s)[0]]) for s in sids])
            return Q, np.vstack(S2), np.vstack(S3), R, y
        Qtr, S2tr, S3tr, Rtr, ytr = views(tid); Qte, S2te, S3te, Rte, yte = views(eid)
        # ---- shared augmentation draws (same anchors/partners/t for every subject-level view)
        counts = np.bincount(ytr); minority = int(np.argmin(counts)); nsyn = int(counts.max() - counts.min())
        mi = np.flatnonzero(ytr == minority); triples = []
        if nsyn > 0:
            rng, nearest = gsmote_pairs([Qtr[i] for i in mi], S2tr[mi], seed)
            for _ in range(nsyn):
                a = int(rng.integers(len(mi))); b = int(rng.choice(nearest[a])); t = float(rng.random()); triples.append((mi[a], mi[b], t))
        def augment_vec(S):  # linear interpolation of a vector view along the shared draws
            if not triples: return S, ytr
            syn = np.vstack([S[a] + t * (S[b] - S[a]) for a, b, t in triples]); return np.vstack([S, syn]), np.concatenate([ytr, np.full(len(triples), minority)])
        def augment_Q(Q):
            if not triples: return list(Q)
            return list(Q) + [M.grassmann_geodesic(Q[a], Q[b], t) for a, b, t in triples]
        Qa = augment_Q(Qtr); S2a, ya = augment_vec(S2tr); S3a, _ = augment_vec(S3tr)
        def add(name, dec): ba, f1 = score(yte, dec); rows.append(dict(cohort=cohort, fold=fold, method=name, ba=ba, f1=f1))
        # ---- baselines: paper fused (w=0.5) and Euclidean RBF (w=0), reproduced here
        dg, de = M.chordal_squared(Qa), M.euclidean_squared(S2a); gg, ge = rbf_med(dg), rbf_med(de)
        Kg, Ke = np.exp(-gg * dg), np.exp(-ge * de); Kg_te, Ke_te = np.exp(-gg * M.chordal_squared(Qte, Qa)), np.exp(-ge * M.euclidean_squared(S2te, S2a))
        add("paper fused w=0.5", svc_pre(0.5 * Kg + 0.5 * Ke, ya, 0.5 * Kg_te + 0.5 * Ke_te)); add("Euclidean RBF (mean+std)", svc_pre(Ke, ya, Ke_te))
        # ---- (a) order-statistics view (sorted triples), RBF
        d3 = M.euclidean_squared(S3a); g3 = rbf_med(d3); K3, K3te = np.exp(-g3 * d3), np.exp(-g3 * M.euclidean_squared(S3te, S3a))
        add("order-stat RBF", svc_pre(K3, ya, K3te)); add("order-stat + Grassmann w=0.3", svc_pre(0.3 * Kg + 0.7 * K3, ya, 0.3 * Kg_te + 0.7 * K3te))
        add("order-stat + mean/std RBF (avg)", svc_pre(0.5 * K3 + 0.5 * Ke, ya, 0.5 * K3te + 0.5 * Ke_te))
        # ---- (b) distribution kernel: mean embedding with base RBF on recordings, RKHS-mixture augmentation
        Rall_tr = np.vstack(Rtr); n_tr = len(Rtr)
        drec = M.euclidean_squared(Rall_tr); grec = rbf_med(drec); Krec = np.exp(-grec * drec)
        blk = np.repeat(np.arange(n_tr), 3)
        E = np.zeros((n_tr, n_tr))
        for i in range(n_tr):
            for j in range(n_tr): E[i, j] = Krec[np.ix_(blk == i, blk == j)].mean()
        # augmentation: mixture embeddings (linear in E)
        if triples:
            A = np.zeros((len(triples), n_tr)); rowsE = []
            for (a, b, t) in triples: v = np.zeros(n_tr); v[a] += 1 - t; v[b] += t; rowsE.append(v)
            Wm = np.vstack(rowsE)                         # synthetic = Wm @ embeddings
            Efull = np.block([[E, E @ Wm.T], [Wm @ E, Wm @ E @ Wm.T]])
        else:
            Wm = np.zeros((0, n_tr)); Efull = E
        Rall_te = np.vstack(Rte); n_te = len(Rte); blk_te = np.repeat(np.arange(n_te), 3)
        Krec_te = np.exp(-grec * M.euclidean_squared(Rall_te, Rall_tr)); Krec_tt = np.exp(-grec * M.euclidean_squared(Rall_te))
        Ete_tr = np.zeros((n_te, n_tr))
        for i in range(n_te):
            for j in range(n_tr): Ete_tr[i, j] = Krec_te[np.ix_(blk_te == i, blk == j)].mean()
        Ete_full = np.hstack([Ete_tr, Ete_tr @ Wm.T]) if triples else Ete_tr
        Ete_diag = np.array([Krec_tt[np.ix_(blk_te == i, blk_te == i)].mean() for i in range(n_te)])
        add("mean-embedding linear kernel", svc_pre(Efull, ya, Ete_full))
        dE = np.diag(Efull); MMD2 = dE[:, None] + dE[None, :] - 2 * Efull; MMD2 = np.maximum(MMD2, 0)
        g2 = rbf_med(MMD2); Kmmd = np.exp(-g2 * MMD2)
        MMD2_te = Ete_diag[:, None] + dE[None, :] - 2 * Ete_full; Kmmd_te = np.exp(-g2 * np.maximum(MMD2_te, 0))
        add("MMD Gaussian kernel", svc_pre(Kmmd, ya, Kmmd_te))
        add("MMD + mean/std RBF (avg)", svc_pre(0.5 * Kmmd + 0.5 * Ke, ya, 0.5 * Kmmd_te + 0.5 * Ke_te))
        add("MMD + Grassmann + mean/std (1/3 each)", svc_pre((Kmmd + Kg + Ke) / 3, ya, (Kmmd_te + Kg_te + Ke_te) / 3))
        # ---- (c) family-wise MKL with centered alignment weights (ALIGNF), on mean+std blocks
        fams = sorted(set(fam)) if cohort == "PD-252" else None
        if fams and len(fams) > 2:
            d2 = S2a.shape[1] // 2; Ks, Kste = [], []
            for g in fams:
                cols = np.flatnonzero(fam == g); cols2 = np.concatenate([cols, d2 + cols])
                dd = M.euclidean_squared(S2a[:, cols2]); gk = rbf_med(dd); Ks.append(np.exp(-gk * dd)); Kste.append(np.exp(-gk * M.euclidean_squared(S2te[:, cols2], S2a[:, cols2])))
            n = len(ya); H = np.eye(n) - np.ones((n, n)) / n; yy = np.outer(2 * ya - 1, 2 * ya - 1)
            Kc = [H @ K @ H for K in Ks]; Mm = np.array([[np.sum(Ka * Kb) for Kb in Kc] for Ka in Kc]); av = np.array([np.sum(Ka * yy) for Ka in Kc])
            L = np.linalg.cholesky(Mm + 1e-9 * np.eye(len(Ks))); v, _ = nnls(L.T, np.linalg.solve(L, av)); beta = v / (v.sum() + 1e-12)
            Kf = sum(b * K for b, K in zip(beta, Ks)); Kfte = sum(b * K for b, K in zip(beta, Kste))
            add("family MKL (ALIGNF)", svc_pre(Kf, ya, Kfte)); add("family MKL + Grassmann w=0.3", svc_pre(0.3 * Kg + 0.7 * Kf, ya, 0.3 * Kg_te + 0.7 * Kfte))
            add("family uniform MKL", svc_pre(sum(Ks) / len(Ks), ya, sum(Kste) / len(Kste)))
            rows.append(dict(cohort=cohort, fold=fold, method="__beta__", ba=np.nan, f1=np.nan, note=str({g: round(b, 3) for g, b in zip(fams, beta)})))
        # ---- (d) record-level training + subject aggregation (RBF SVM on recordings, mean decision per subject)
        yrec = np.repeat(ytr, 3); Krr = Krec; clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Krr, yrec)
        dec_rec = clf.decision_function(Krec_te); dec_sub = np.array([dec_rec[blk_te == i].mean() for i in range(n_te)])
        add("record-level RBF SVM -> mean", dec_sub)
        # stack with the paper's fused model (standardized decision values)
        d_f = svc_pre(0.5 * Kg + 0.5 * Ke, ya, 0.5 * Kg_te + 0.5 * Ke_te); d_f_tr = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(0.5 * Kg + 0.5 * Ke, ya).decision_function(0.5 * Kg + 0.5 * Ke)
        d_r_tr = clf.decision_function(Krr); s_f, s_r = d_f_tr.std() + 1e-12, d_r_tr.std() + 1e-12
        add("stack: fused + record-level (avg)", 0.5 * d_f / s_f + 0.5 * dec_sub / s_r)
        # ---- (e) tree ensembles on mean+std (class-weighted, no augmentation)
        for name, mdl in (("HGB (mean+std)", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)),
                          ("RF (mean+std)", RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=0, n_jobs=4)),
                          ("ET (mean+std)", ExtraTreesClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=0, n_jobs=4))):
            mdl.fit(S2tr, ytr); p = mdl.predict_proba(S2te)[:, 1]; add(name, p - 0.5)
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0).fit(S3tr, ytr); add("HGB (order-stat)", mdl.predict_proba(S3te)[:, 1] - 0.5)
        # ---- (f) univariate feature selection (training subjects only) then RBF on mean+std
        mu_tr = S2tr[:, :S2tr.shape[1] // 2]
        tstat = np.abs((mu_tr[ytr == 1].mean(0) - mu_tr[ytr == 0].mean(0)) / (np.sqrt(mu_tr[ytr == 1].var(0) / (ytr == 1).sum() + mu_tr[ytr == 0].var(0) / (ytr == 0).sum()) + 1e-9))
        order = np.argsort(-tstat); d2 = S2tr.shape[1] // 2
        for k in (25, 50, 100, 200):
            if k >= d2: continue
            cols = np.concatenate([order[:k], d2 + order[:k]]); dd = M.euclidean_squared(S2a[:, cols]); gk = rbf_med(dd)
            add("top-%d feats RBF" % k, svc_pre(np.exp(-gk * dd), ya, np.exp(-gk * M.euclidean_squared(S2te[:, cols], S2a[:, cols]))))
        print(cohort, "fold", fold, "done", flush=True)
    return pd.DataFrame(rows)

df = pd.concat([run("UCI-489"), run("PD-252")]); df.to_csv("robustness_out/newmethods1.csv", index=False)
pd.set_option("display.width", 200)
for cohort in ("UCI-489", "PD-252"):
    d = df[(df.cohort == cohort) & (df.method != "__beta__")].groupby("method")[["ba", "f1"]].mean().round(3)
    print("\n==", cohort); print(d.sort_values("ba", ascending=False).to_string())
b = df[df.method == "__beta__"]
if len(b): print("\nALIGNF family weights per fold:"); print(b[["cohort", "fold", "note"]].to_string(index=False))
print("DONE")
