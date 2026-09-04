"""Exploration 4: alignment-weighted multi-view fusion (families + Grassmann + MMD [+ full Euclidean]),
record-level evidence, augmentation bagging, quantile normalization, nested BA-threshold. Default partition, both cohorts."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
import voxneuro.method as M

def load(name):
    if name == "UCI-489":
        frame, feats = M._validate_frame(pd.read_csv("data/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"), "ID", "Status", ("Recording",)); idc, labc = "ID", "Status"
    else:
        frame, feats = M._validate_frame(pd.read_csv("data/pd_speech_features_clean.csv"), "id", "class", ()); idc, labc = "id", "class"
    X = frame[feats].to_numpy(float); rid = frame[idc].to_numpy(str); rl = frame[labc].to_numpy(int); sf = frame[[idc, labc]].drop_duplicates(idc)
    return dict(X=X, rid=rid, rl=rl, sids=sf[idc].to_numpy(str), slab=sf[labc].to_numpy(int), rmap=M._subject_rows(rid), feats=feats,
                sub_label={s: int(l) for s, l in zip(sf[idc].to_numpy(str), sf[labc].to_numpy(int))})

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

def rbf_med(D2):
    v = D2[np.triu_indices(D2.shape[0], 1)] if D2.shape[0] == D2.shape[1] else D2.ravel()
    v = v[np.isfinite(v) & (v > 0)]; return 1.0 / (2.0 * np.median(v) + 1e-12)

def alignf(Ks, y):
    n = len(y); H = np.eye(n) - 1.0 / n; yy = np.outer(2 * y - 1, 2 * y - 1)
    Kc = [H @ K @ H for K in Ks]; Mm = np.array([[np.sum(a * b) for b in Kc] for a in Kc]); av = np.array([np.sum(a * yy) for a in Kc])
    L = np.linalg.cholesky(Mm + 1e-8 * np.trace(Mm) / len(Ks) * np.eye(len(Ks))); v, _ = nnls(L.T, np.linalg.solve(L, av))
    if v.sum() <= 0: v = np.ones(len(Ks))
    return v / v.sum()

def svc_dec(K, y, Kte, C=1.0):
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K, y); return clf.decision_function(Kte), clf.decision_function(K)

def split_eval(D, cohort, fam, tr_sids, te_sids, seed, prep):
    """Fit everything on tr_sids, return decision values on te_sids for each method (dict), plus training-decision scales."""
    trr = np.concatenate([D["rmap"][s] for s in tr_sids]); ter = np.concatenate([D["rmap"][s] for s in te_sids])
    if prep == "quantile":
        sc = QuantileTransformer(n_quantiles=min(200, len(trr)), output_distribution="normal", random_state=0).fit(D["X"][trr])
    else:
        sc = StandardScaler().fit(D["X"][trr])
    Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
    def views(sids):
        Q, S2, R = [], [], []
        for sid in sids:
            Xi = Xs[D["rmap"][sid]]; U, _, _ = np.linalg.svd(Xi.T, full_matrices=False); Q.append(U[:, :3]); S2.append(np.concatenate([Xi.mean(0), Xi.std(0)])); R.append(Xi)
        return Q, np.vstack(S2), R, np.array([D["sub_label"][s] for s in sids])
    Qtr, S2tr, Rtr, ytr = views(tr_sids); Qte, S2te, Rte, yte = views(te_sids)
    counts = np.bincount(ytr, minlength=2); minority = int(np.argmin(counts)); nsyn = int(counts.max() - counts.min()); mi = np.flatnonzero(ytr == minority); triples = []
    if nsyn > 0 and len(mi) >= 2:
        d = M.chordal_squared([Qtr[i] for i in mi]); np.fill_diagonal(d, np.inf); nearest = np.argsort(d, axis=1)[:, :min(5, len(mi) - 1)]; rng = np.random.default_rng(seed)
        for _ in range(nsyn):
            a = int(rng.integers(len(mi))); b = int(rng.choice(nearest[a])); t = float(rng.random()); triples.append((mi[a], mi[b], t))
    ya = np.concatenate([ytr, np.full(len(triples), minority)]) if triples else ytr
    Qa = list(Qtr) + [M.grassmann_geodesic(Qtr[a], Qtr[b], t) for a, b, t in triples]
    S2a = np.vstack([S2tr] + [S2tr[a] + t * (S2tr[b] - S2tr[a]) for a, b, t in triples]) if triples else S2tr
    d2 = S2tr.shape[1] // 2
    # kernels
    Kd = {}
    dg = M.chordal_squared(Qa); gg = rbf_med(dg); Kd["G"] = (np.exp(-gg * dg), np.exp(-gg * M.chordal_squared(Qte, Qa)))
    de = M.euclidean_squared(S2a); ge = rbf_med(de); Kd["E"] = (np.exp(-ge * de), np.exp(-ge * M.euclidean_squared(S2te, S2a)))
    fams = sorted(set(fam))
    for g in fams:
        cols = np.flatnonzero(fam == g); cols2 = np.concatenate([cols, d2 + cols]); dd = M.euclidean_squared(S2a[:, cols2]); gk = rbf_med(dd)
        Kd["F:" + g] = (np.exp(-gk * dd), np.exp(-gk * M.euclidean_squared(S2te[:, cols2], S2a[:, cols2])))
    # MMD distribution kernel (vectorized block means) with RKHS-mixture augmentation
    Rall = np.vstack(Rtr); n_tr = len(Rtr); B = np.zeros((n_tr, 3 * n_tr))
    for i in range(n_tr): B[i, 3 * i:3 * i + 3] = 1 / 3
    drec = M.euclidean_squared(Rall); grec = rbf_med(drec); Krec = np.exp(-grec * drec); E = B @ Krec @ B.T
    if triples:
        Wm = np.zeros((len(triples), n_tr))
        for r, (a, b, t) in enumerate(triples): Wm[r, a] += 1 - t; Wm[r, b] += t
        Efull = np.block([[E, E @ Wm.T], [Wm @ E, Wm @ E @ Wm.T]])
    else:
        Wm = np.zeros((0, n_tr)); Efull = E
    Rte_all = np.vstack(Rte); n_te = len(Rte); Bt = np.zeros((n_te, 3 * n_te))
    for i in range(n_te): Bt[i, 3 * i:3 * i + 3] = 1 / 3
    Krec_te = np.exp(-grec * M.euclidean_squared(Rte_all, Rall)); Ete = Bt @ Krec_te @ B.T; Ete_full = np.hstack([Ete, Ete @ Wm.T]) if triples else Ete
    Ete_diag = np.einsum("ij,jk,ik->i", Bt, np.exp(-grec * M.euclidean_squared(Rte_all)), Bt)
    dE = np.diag(Efull); MMD2 = np.maximum(dE[:, None] + dE[None, :] - 2 * Efull, 0); g2 = rbf_med(MMD2)
    Kd["MMD"] = (np.exp(-g2 * MMD2), np.exp(-g2 * np.maximum(Ete_diag[:, None] + dE[None, :] - 2 * Ete_full, 0)))
    # record-level model (all features) -> mean over the subject's recordings
    yrec = np.repeat(ytr, 3); clf_r = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Krec, yrec)
    dec_rec = clf_r.decision_function(Krec_te); rec_sub = Bt @ dec_rec * 1.0; rec_tr = B @ clf_r.decision_function(Krec)
    out, scale = {}, {}
    def put(name, dec_te, dec_tr): out[name] = dec_te; scale[name] = dec_tr.std() + 1e-12
    put("RBF (mean+std)", *svc_dec(Kd["E"][0], ya, Kd["E"][1]))
    put("paper fused w=0.5", *svc_dec(0.5 * Kd["G"][0] + 0.5 * Kd["E"][0], ya, 0.5 * Kd["G"][1] + 0.5 * Kd["E"][1]))
    put("record-level -> mean", rec_sub, rec_tr)
    def align_model(name, keys):
        Ktr = [Kd[k][0] for k in keys]; Kte = [Kd[k][1] for k in keys]; beta = alignf(Ktr, ya)
        put(name, *svc_dec(sum(b * K for b, K in zip(beta, Ktr)), ya, sum(b * K for b, K in zip(beta, Kte)))); return beta
    fam_keys = ["F:" + g for g in fams]
    align_model("A1 families", fam_keys); align_model("A2 families+G", fam_keys + ["G"]); beta3 = align_model("A3 families+G+MMD", fam_keys + ["G", "MMD"]); align_model("A4 families+G+MMD+E", fam_keys + ["G", "MMD", "E"])
    put("A3 + record stack", 0.5 * out["A3 families+G+MMD"] / scale["A3 families+G+MMD"] + 0.5 * rec_sub / scale["record-level -> mean"], np.ones(2))
    put("A2 + record stack", 0.5 * out["A2 families+G"] / scale["A2 families+G"] + 0.5 * rec_sub / scale["record-level -> mean"], np.ones(2))
    return out, yte, dict(zip(fam_keys + ["G", "MMD"], np.round(beta3, 3)))

def main(cohort):
    D = load(cohort); fam = np.array([family(c, cohort) for c in D["feats"]]); rows = []
    for prep in ("standard", "quantile"):
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D["sids"], D["slab"]), start=1):
            tid, eid = D["sids"][tr], D["sids"][te]; seed = 42 * 1000 + fold
            out, yte, beta = split_eval(D, cohort, fam, tid, eid, seed, prep)
            # augmentation bagging (8 seeds) for the two strongest configurations
            bag = {"A3 families+G+MMD": [], "A3 + record stack": [], "paper fused w=0.5": []}
            for k in range(8):
                o, _, _ = split_eval(D, cohort, fam, tid, eid, seed * 10 + k, prep)
                for m in bag: bag[m].append(o[m])
            for m in bag: out[m + " [bag8]"] = np.mean(bag[m], axis=0)
            # nested BA-threshold (inner 5-fold on training subjects) for baseline and the new methods
            ytr = np.array([D["sub_label"][s] for s in tid]); inner_dec = {m: np.zeros(len(tid)) for m in ("RBF (mean+std)", "A3 families+G+MMD", "A3 + record stack")}
            for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(tid)), ytr), start=1):
                o, _, _ = split_eval(D, cohort, fam, tid[a], tid[b], seed * 100 + k, prep)
                for m in inner_dec: inner_dec[m][b] = o[m]
            for m, dec in inner_dec.items():
                cands = np.unique(dec); best_t, best_ba = 0.0, -1
                for t in cands:
                    ba = balanced_accuracy_score(ytr, (dec > t).astype(int))
                    if ba > best_ba + 1e-12: best_ba, best_t = ba, t
                out[m + " [BA-thr]"] = out[m] - best_t
            for m, dec in out.items():
                yp = (dec > 0).astype(int); rows.append(dict(cohort=cohort, prep=prep, fold=fold, method=m, ba=balanced_accuracy_score(yte, yp), f1=f1_score(yte, yp, average="macro")))
            print(cohort, prep, "fold", fold, "beta", beta, flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.concat([main("UCI-489"), main("PD-252")]); df.to_csv("robustness_out/newmethods2.csv", index=False)
    pd.set_option("display.width", 220)
    for cohort in ("UCI-489", "PD-252"):
        for prep in ("standard", "quantile"):
            d = df[(df.cohort == cohort) & (df.prep == prep)].groupby("method")[["ba", "f1"]].mean().round(3)
            print("\n==", cohort, prep); print(d.sort_values("ba", ascending=False).to_string())
    print("DONE")
