"""VoxNeuro-2 candidate: rank-normalized, family-wise alignment-weighted multiple-kernel fusion with the
Grassmann view, evaluated with everything (normalizer, family weights, fusion weight) fitted inside training folds.

All hyperparameters that are not prespecified are chosen by inner subject-disjoint CV on the training folds only;
test subjects never influence any fit. Distances are exact vectorized equivalents of voxneuro.method."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
import voxneuro.method as M

# ---------- data ----------
def load(name):
    if name == "UCI-489":
        frame, feats = M._validate_frame(pd.read_csv("data/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"), "ID", "Status", ("Recording",)); idc, labc = "ID", "Status"
    else:
        frame, feats = M._validate_frame(pd.read_csv("data/pd_speech_features_clean.csv"), "id", "class", ()); idc, labc = "id", "class"
    sf = frame[[idc, labc]].drop_duplicates(idc)
    return dict(X=frame[feats].to_numpy(float), rmap=M._subject_rows(frame[idc].to_numpy(str)), feats=feats,
                sids=sf[idc].to_numpy(str), slab=sf[labc].to_numpy(int), sub_label={s: int(l) for s, l in zip(sf[idc].to_numpy(str), sf[labc].to_numpy(int))})

def family(c, cohort):
    c0 = c; c = c.lower()
    if cohort == "PD-252":
        if c == "gender": return "Demographics"
        if c.startswith("tqwt"): return "TQWT"
        if "mfcc" in c or "delta" in c or "log_energy" in c: return "MFCC"
        if c.startswith("imf"): return "IMF/EMD"
        if c.startswith(("det_", "app_", "ea", "ed")): return "Wavelet"
        return "Vocal/time-frequency"
    if c0 == "Gender": return "Sex"
    if c0.startswith("Jitter"): return "Jitter"
    if c0.startswith("Shi"): return "Shimmer"
    if c0.startswith("HNR"): return "HNR"
    if c0.startswith("MFCC"): return "MFCC"
    if c0.startswith("Delta"): return "Delta"
    return "Nonlinear"

# ---------- geometry ----------
def euclidean_squared(A, B=None):
    A = np.asarray(A, float); B = A if B is None else np.asarray(B, float)
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T; np.maximum(D, 0.0, out=D)
    if B is A: np.fill_diagonal(D, 0.0)
    return D

def chordal_squared(Qa, Qb=None):
    A = np.hstack(Qa); r = Qa[0].shape[1]; na = len(Qa)
    if Qb is None: Bm, nb = A, na
    else: Bm = np.hstack(Qb); nb = len(Qb)
    D = r - ((A.T @ Bm) ** 2).reshape(na, r, nb, r).sum(axis=(1, 3)); np.maximum(D, 0.0, out=D)
    if Qb is None: np.fill_diagonal(D, 0.0)
    return D

def rbf_med(D2):
    v = D2[np.triu_indices(D2.shape[0], 1)] if D2.shape[0] == D2.shape[1] else D2.ravel()
    v = v[np.isfinite(v) & (v > 0)]; return 1.0 / (2.0 * np.median(v) + 1e-12)

def alignf(Ks, y, floor=0.0):
    """Non-negative centered kernel-target alignment weights (ALIGNF), mixed with a uniform floor."""
    n = len(y); H = np.eye(n) - 1.0 / n; yy = np.outer(2 * y - 1, 2 * y - 1)
    Kc = [H @ K @ H for K in Ks]; Mm = np.array([[np.sum(a * b) for b in Kc] for a in Kc]); av = np.array([np.sum(a * yy) for a in Kc])
    L = np.linalg.cholesky(Mm + 1e-8 * np.trace(Mm) / len(Ks) * np.eye(len(Ks))); v, _ = nnls(L.T, np.linalg.solve(L, av))
    v = np.ones(len(Ks)) if v.sum() <= 0 else v / v.sum()
    return (1 - floor) * v + floor / len(Ks)

# ---------- one split ----------
FLOORS = (0.0, 0.25, 0.5)
W_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

def split_kernels(D, fam, tr_sids, te_sids, seed, prep):
    trr = np.concatenate([D["rmap"][s] for s in tr_sids]); ter = np.concatenate([D["rmap"][s] for s in te_sids])
    sc = (QuantileTransformer(n_quantiles=min(300, len(trr)), output_distribution="normal", subsample=10**9, random_state=0)
          if prep == "quantile" else StandardScaler()).fit(D["X"][trr])
    Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
    def views(sids):
        Q, S2, R = [], [], []
        for sid in sids:
            Xi = Xs[D["rmap"][sid]]; U, _, _ = np.linalg.svd(Xi.T, full_matrices=False)
            Q.append(U[:, :3]); S2.append(np.concatenate([Xi.mean(0), Xi.std(0)])); R.append(Xi)
        return Q, np.vstack(S2), R, np.array([D["sub_label"][s] for s in sids])
    Qtr, S2tr, Rtr, ytr = views(tr_sids); Qte, S2te, Rte, yte = views(te_sids)
    # prespecified imbalance gate: G-SMOTE along Grassmann geodesics + matched linear interpolation of summaries
    counts = np.bincount(ytr, minlength=2); minority = int(np.argmin(counts)); nsyn = int(counts.max() - counts.min())
    mi = np.flatnonzero(ytr == minority); triples = []
    if nsyn > 0 and len(mi) >= 2:
        d = chordal_squared([Qtr[i] for i in mi]); np.fill_diagonal(d, np.inf)
        nearest = np.argsort(d, axis=1)[:, :min(5, len(mi) - 1)]; rng = np.random.default_rng(seed)
        for _ in range(nsyn):
            a = int(rng.integers(len(mi))); b = int(rng.choice(nearest[a])); triples.append((mi[a], mi[b], float(rng.random())))
    ya = np.concatenate([ytr, np.full(len(triples), minority)]) if triples else ytr
    Qa = list(Qtr) + [M.grassmann_geodesic(Qtr[a], Qtr[b], t) for a, b, t in triples]
    S2a = np.vstack([S2tr] + [S2tr[a] + t * (S2tr[b] - S2tr[a]) for a, b, t in triples]) if triples else S2tr
    d2 = S2tr.shape[1] // 2; K = {}
    dg = chordal_squared(Qa); gg = rbf_med(dg); K["G"] = (np.exp(-gg * dg), np.exp(-gg * chordal_squared(Qte, Qa)))
    de = euclidean_squared(S2a); ge = rbf_med(de); K["E"] = (np.exp(-ge * de), np.exp(-ge * euclidean_squared(S2te, S2a)))
    fams = sorted(set(fam))
    for g in fams:
        cols = np.flatnonzero(fam == g); c2 = np.concatenate([cols, d2 + cols]); dd = euclidean_squared(S2a[:, c2]); gk = rbf_med(dd)
        K["F:" + g] = (np.exp(-gk * dd), np.exp(-gk * euclidean_squared(S2te[:, c2], S2a[:, c2])))
    # permutation-invariant mean-embedding (MMD) kernel over each subject's recording set
    Rall = np.vstack(Rtr); n_tr = len(Rtr); B = np.zeros((n_tr, 3 * n_tr))
    for i in range(n_tr): B[i, 3 * i:3 * i + 3] = 1 / 3
    drec = euclidean_squared(Rall); grec = rbf_med(drec); Krec = np.exp(-grec * drec); E = B @ Krec @ B.T
    if triples:
        Wm = np.zeros((len(triples), n_tr))
        for r, (a, b, t) in enumerate(triples): Wm[r, a] += 1 - t; Wm[r, b] += t
        Efull = np.block([[E, E @ Wm.T], [Wm @ E, Wm @ E @ Wm.T]])
    else:
        Wm = np.zeros((0, n_tr)); Efull = E
    Rte_all = np.vstack(Rte); n_te = len(Rte); Bt = np.zeros((n_te, 3 * n_te))
    for i in range(n_te): Bt[i, 3 * i:3 * i + 3] = 1 / 3
    Krec_te = np.exp(-grec * euclidean_squared(Rte_all, Rall)); Ete = Bt @ Krec_te @ B.T
    Ete_full = np.hstack([Ete, Ete @ Wm.T]) if triples else Ete
    Ete_diag = np.einsum("ij,jk,ik->i", Bt, np.exp(-grec * euclidean_squared(Rte_all)), Bt)
    dE = np.diag(Efull); MMD2 = np.maximum(dE[:, None] + dE[None, :] - 2 * Efull, 0); g2 = rbf_med(MMD2)
    K["MMD"] = (np.exp(-g2 * MMD2), np.exp(-g2 * np.maximum(Ete_diag[:, None] + dE[None, :] - 2 * Ete_full, 0)))
    clf_r = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Krec, np.repeat(ytr, 3))
    rec = (Bt @ clf_r.decision_function(Krec_te), (B @ clf_r.decision_function(Krec)).std() + 1e-12)
    return K, ya, yte, rec, fams

def svc(Ktr, y, Kte):
    return SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Ktr, y).decision_function(Kte)

def split_decisions(D, fam, tr_sids, te_sids, seed, preps=("standard", "quantile")):
    """Decision values on the held-out subjects for every candidate configuration."""
    out = {}
    for prep in preps:
        K, ya, yte, rec, fams = split_kernels(D, fam, tr_sids, te_sids, seed, prep); p = prep[0]
        out["E-RBF/" + p] = svc(K["E"][0], ya, K["E"][1])
        out["fused w=0.5/" + p] = svc(0.5 * K["G"][0] + 0.5 * K["E"][0], ya, 0.5 * K["G"][1] + 0.5 * K["E"][1])
        out["record/" + p] = rec[0]
        keys = ["F:" + g for g in fams] + ["MMD"]; Ktr = [K[k][0] for k in keys]; Kte = [K[k][1] for k in keys]
        for fl in FLOORS:
            beta = alignf(Ktr, ya, fl); KA = sum(b * k for b, k in zip(beta, Ktr)); KAte = sum(b * k for b, k in zip(beta, Kte))
            for w in W_GRID:
                out["MKL fl=%.2f w=%.1f/%s" % (fl, w, p)] = svc((1 - w) * KA + w * K["G"][0], ya, (1 - w) * KAte + w * K["G"][1])
    return out, yte

CAND = ["MKL fl=%.2f w=%.1f" % (fl, w) for fl in FLOORS for w in W_GRID]

def evaluate(D, fam, part_seed, seed_of_fold, preps=("standard", "quantile"), inner_seed=7):
    """One 5-fold subject-disjoint evaluation. Returns per-method (BA, F1) fold means plus nested selections."""
    acc, chosen = {}, []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=part_seed).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]; seed = seed_of_fold(fold)
        out, yte = split_decisions(D, fam, tid, eid, seed, preps)
        ytr = np.array([D["sub_label"][s] for s in tid]); inner = {}
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=inner_seed).split(np.zeros(len(tid)), ytr), start=1):
            o, _ = split_decisions(D, fam, tid[a], tid[b], seed * 100 + k, preps)
            for m, v in o.items(): inner.setdefault(m, np.zeros(len(tid)))[b] = v
        def sc(m):
            yp = (inner[m] > 0).astype(int); return 0.5 * (balanced_accuracy_score(ytr, yp) + f1_score(ytr, yp, average="macro"))
        # nested selection of (floor, w) within each preprocessing, and jointly over preprocessing
        for p in [prep[0] for prep in preps]:
            best = max(CAND, key=lambda c: (sc(c + "/" + p), -float(c.split("w=")[1])))
            out["MKL nested/" + p] = out[best + "/" + p]
        if len(preps) > 1:
            allc = [c + "/" + p for c in CAND for p in [prep[0] for prep in preps]]
            bj = max(allc, key=lambda c: (sc(c), -float(c.split("w=")[1].split("/")[0])))
            out["MKL nested+prep"] = out[bj]
            bb = max(["E-RBF/" + p for p in [prep[0] for prep in preps]] + ["fused w=0.5/" + p for p in [prep[0] for prep in preps]], key=sc)
            out["baseline nested+prep"] = out[bb]
            chosen.append(bj)
        for m, dec in out.items():
            yp = (dec > 0).astype(int); acc.setdefault(m, []).append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
    res = {m: (float(np.mean([v[0] for v in a])), float(np.mean([v[1] for v in a]))) for m, a in acc.items()}
    return res, chosen
