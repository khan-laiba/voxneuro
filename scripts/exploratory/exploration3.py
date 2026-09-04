"""Exploration 5 (Codex-inspired additions), default partition, both cohorts, all augmentation-free unless stated:
uniform-floor ALIGNF, permanent set kernel, reliability/effect-weighted metric, nested BA threshold, balanced subagging, PLS-DA+LDA."""
import warnings; warnings.filterwarnings("ignore")
import sys, itertools, numpy as np, pandas as pd
sys.argv = [sys.argv[0]]
from newmethods2 import load, family, rbf_med, alignf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score
import voxneuro.method as M

def prep(D, tr_sids, te_sids):
    trr = np.concatenate([D["rmap"][s] for s in tr_sids]); ter = np.concatenate([D["rmap"][s] for s in te_sids])
    sc = StandardScaler().fit(D["X"][trr]); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
    def views(sids):
        S2, R, Q = [], [], []
        for sid in sids:
            Xi = Xs[D["rmap"][sid]]; S2.append(np.concatenate([Xi.mean(0), Xi.std(0)])); R.append(Xi); U, _, _ = np.linalg.svd(Xi.T, full_matrices=False); Q.append(U[:, :3])
        return np.vstack(S2), R, Q, np.array([D["sub_label"][s] for s in sids])
    return views(tr_sids), views(te_sids)

def rbf_pair(A, B):
    d = M.euclidean_squared(A); g = rbf_med(d); return np.exp(-g * d), np.exp(-g * M.euclidean_squared(B, A))

def perm_kernel(Rtr, Rte):
    """Mean-embedding and permanent (symmetric tensor) set kernels from a recording RBF; diagonally normalized."""
    A = np.vstack(Rtr); Bm = np.vstack(Rte); n, m = len(Rtr), len(Rte)
    d = M.euclidean_squared(A); g = rbf_med(d); Ktt = np.exp(-g * d); Kst = np.exp(-g * M.euclidean_squared(Bm, A)); Kss = np.exp(-g * M.euclidean_squared(Bm))
    def blocks(K, n1, n2): return K.reshape(n1, 3, n2, 3).transpose(0, 2, 1, 3)   # [i, j, r, s]
    def perm(T):
        out = np.zeros(T.shape[:2])
        for p in itertools.permutations(range(3)): out += T[:, :, 0, p[0]] * T[:, :, 1, p[1]] * T[:, :, 2, p[2]]
        return out / 6.0
    Ttt, Tst, Tss = blocks(Ktt, n, n), blocks(Kst, m, n), blocks(Kss, m, m)
    Ptt, Pst = perm(Ttt), perm(Tst); dtt = np.sqrt(np.diag(Ptt)); dss = np.sqrt(np.array([perm(Tss[i:i+1, i:i+1])[0, 0] for i in range(m)]))
    Ptt_n = Ptt / np.outer(dtt, dtt); Pst_n = Pst / np.outer(dss, dtt)
    Mtt, Mst = Ttt.mean(axis=(2, 3)), Tst.mean(axis=(2, 3))   # mean-embedding (linear) kernel
    return (Ptt_n, Pst_n), (Mtt, Mst)

def reliability_weights(Rtr, S2tr, ytr, rho, eta):
    mu = np.stack([r.mean(0) for r in Rtr]); within = np.stack([r.var(0) for r in Rtr]).mean(0); between = mu.var(0)
    r = between / (between + within + 1e-9)
    m1, m0 = mu[ytr == 1].mean(0), mu[ytr == 0].mean(0); sp = np.sqrt(0.5 * (mu[ytr == 1].var(0) + mu[ytr == 0].var(0))) + 1e-9
    e = np.abs(m1 - m0) / sp
    a = (r + 0.05) ** rho * (e + 0.05) ** eta; a = np.minimum(a, np.percentile(a, 95)); a = a / a.mean()
    return np.concatenate([a, a])   # same weights for mean and dispersion coordinates

def svc(K, y, Kte, C=1.0):
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K, y); return clf.decision_function(Kte), clf.decision_function(K)

def methods(D, fam, tr_sids, te_sids, want=None):
    (S2tr, Rtr, Qtr, ytr), (S2te, Rte, Qte, yte) = prep(D, tr_sids, te_sids); out = {}
    d2 = S2tr.shape[1] // 2; fams = sorted(set(fam))
    Ke, Kete = rbf_pair(S2tr, S2te); out["RBF no-aug"] = svc(Ke, ytr, Kete)[0]
    Kf, Kfte = [], []
    for g in fams:
        cols = np.flatnonzero(fam == g); cols2 = np.concatenate([cols, d2 + cols]); a, b = rbf_pair(S2tr[:, cols2], S2te[:, cols2]); Kf.append(a); Kfte.append(b)
    dg = M.chordal_squared(Qtr); gg = rbf_med(dg); Kg, Kgte = np.exp(-gg * dg), np.exp(-gg * M.chordal_squared(Qte, Qtr))
    (Pt, Pte), (Mt, Mte) = perm_kernel(Rtr, Rte)
    dE = np.diag(Mt); MMD2 = np.maximum(dE[:, None] + dE[None, :] - 2 * Mt, 0); g2 = rbf_med(MMD2); Km = np.exp(-g2 * MMD2)
    Mte_diag = np.array([np.exp(-rbf_med(M.euclidean_squared(np.vstack(Rtr))) * M.euclidean_squared(r)).mean() for r in Rte])
    Kmte = np.exp(-g2 * np.maximum(Mte_diag[:, None] + dE[None, :] - 2 * Mte, 0))
    views_tr = Kf + [Kg, Km]; views_te = Kfte + [Kgte, Kmte]
    beta = alignf(views_tr, ytr)
    for floor in (0.0, 0.25, 0.5):
        b = (1 - floor) * beta + floor / len(beta)
        out["A3 no-aug floor=%.2f" % floor] = svc(sum(x * K for x, K in zip(b, views_tr)), ytr, sum(x * K for x, K in zip(b, views_te)))[0]
    out["perm kernel no-aug"] = svc(Pt, ytr, Pte)[0]
    bp = alignf(views_tr + [Pt], ytr); out["A3+perm no-aug"] = svc(sum(x * K for x, K in zip(bp, views_tr + [Pt])), ytr, sum(x * K for x, K in zip(bp, views_te + [Pte])))[0]
    for rho, eta in ((1, 0), (0, 1), (1, 1), (0.5, 0.5)):
        w = np.sqrt(reliability_weights(Rtr, S2tr, ytr, rho, eta)); a, b = rbf_pair(S2tr * w, S2te * w); out["rel-weighted RBF rho=%g eta=%g" % (rho, eta)] = svc(a, ytr, b)[0]
    for q in (4, 8, 16):
        yt = np.where(ytr == 1, 1.0 / (ytr == 1).sum(), -1.0 / (ytr == 0).sum()); ssc = StandardScaler().fit(S2tr)
        pls = PLSRegression(n_components=q, scale=False).fit(ssc.transform(S2tr), yt); Ztr, Zte = pls.transform(ssc.transform(S2tr)), pls.transform(ssc.transform(S2te))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(Ztr, ytr); out["PLS%d + LDA" % q] = lda.decision_function(Zte)
    # balanced subagging (16 members) for RBF and A3 (only meaningful when imbalanced)
    if abs((ytr == 1).sum() - (ytr == 0).sum()) > 5:
        rng = np.random.default_rng(0); mino = int(np.argmin(np.bincount(ytr))); imin = np.flatnonzero(ytr == mino); imaj = np.flatnonzero(ytr != mino)
        acc_r, acc_a = [], []
        for k in range(16):
            sel = np.concatenate([imin, rng.choice(imaj, size=len(imin), replace=False)])
            a, b = rbf_pair(S2tr[sel], S2te); dec, dtr = svc(a, ytr[sel], b); acc_r.append(np.tanh(dec / (np.median(np.abs(dtr)) + 1e-9)))
            vt = [K[np.ix_(sel, sel)] for K in views_tr]; ve = [K[:, sel] for K in views_te]; bb = alignf(vt, ytr[sel])
            dec, dtr = svc(sum(x * K for x, K in zip(bb, vt)), ytr[sel], sum(x * K for x, K in zip(bb, ve))); acc_a.append(np.tanh(dec / (np.median(np.abs(dtr)) + 1e-9)))
        out["RBF subagging16"] = np.mean(acc_r, axis=0); out["A3 subagging16"] = np.mean(acc_a, axis=0)
    return out, yte, dict(zip(["F:" + g for g in fams] + ["G", "MMD"], np.round(beta, 3)))

def main(cohort):
    D = load(cohort); fam = np.array([family(c, cohort) for c in D["feats"]]); rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]
        out, yte, beta = methods(D, fam, tid, eid)
        # nested BA threshold for two models
        ytr = np.array([D["sub_label"][s] for s in tid]); inner = {"RBF no-aug": np.zeros(len(tid)), "A3 no-aug floor=0.25": np.zeros(len(tid))}
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(tid)), ytr), start=1):
            o, _, _ = methods(D, fam, tid[a], tid[b])
            for m in inner: inner[m][b] = o[m]
        for m, dec in inner.items():
            best_t, best = 0.0, -1
            for t in np.unique(dec):
                ba = balanced_accuracy_score(ytr, (dec > t).astype(int))
                if ba > best + 1e-12: best, best_t = ba, t
            out[m + " [BA-thr]"] = out[m] - best_t
        for m, dec in out.items():
            yp = (dec > 0).astype(int); rows.append(dict(cohort=cohort, fold=fold, method=m, ba=balanced_accuracy_score(yte, yp), f1=f1_score(yte, yp, average="macro")))
        print(cohort, "fold", fold, "beta", beta, flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.concat([main("UCI-489"), main("PD-252")]); df.to_csv("robustness_out/newmethods3.csv", index=False)
    pd.set_option("display.width", 220)
    for cohort in ("UCI-489", "PD-252"):
        d = df[df.cohort == cohort].groupby("method")[["ba", "f1"]].mean().round(3); print("\n==", cohort); print(d.sort_values("ba", ascending=False).to_string())
    print("DONE")
