"""Exploration 6: supervised low-rank projection before the Grassmann step.
Recordings are compressed by a PLS projection fitted on training recordings only; subject subspaces, summaries and
set kernels are then built in that low-dimensional space, where chordal distances are no longer concentrated."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from vox2 import load, family, euclidean_squared, chordal_squared, rbf_med, alignf
import voxneuro.method as M

QS = (2, 4, 8, 16, 32)

def pls_fit(Xtr, ytr_rec, q):
    t = np.where(ytr_rec == 1, 1.0 / max((ytr_rec == 1).sum(), 1), -1.0 / max((ytr_rec == 0).sum(), 1))
    return PLSRegression(n_components=q, scale=False).fit(Xtr, t)

def subj_views(Z, rows, sids, r=3):
    Q, S2, R = [], [], []
    for s in sids:
        Zi = Z[rows[s]]; U, _, _ = np.linalg.svd(Zi.T, full_matrices=False)
        Q.append(U[:, :min(r, U.shape[1])]); S2.append(np.concatenate([Zi.mean(0), Zi.std(0)])); R.append(Zi)
    return Q, np.vstack(S2), R

def svc(Ktr, y, Kte, C=1.0):
    return SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, y).decision_function(Kte)

def kpair(A, B):
    d = euclidean_squared(A); g = rbf_med(d); return np.exp(-g * d), np.exp(-g * euclidean_squared(B, A))

def gpair(Qa, Qb):
    d = chordal_squared(Qa); g = rbf_med(d); return np.exp(-g * d), np.exp(-g * chordal_squared(Qb, Qa))

def setpair(Rtr, Rte):
    A = np.vstack(Rtr); Bm = np.vstack(Rte); n, m = len(Rtr), len(Rte)
    B = np.zeros((n, 3 * n)); Bt = np.zeros((m, 3 * m))
    for i in range(n): B[i, 3 * i:3 * i + 3] = 1 / 3
    for i in range(m): Bt[i, 3 * i:3 * i + 3] = 1 / 3
    d = euclidean_squared(A); g = rbf_med(d); Ktt = np.exp(-g * d); Kst = np.exp(-g * euclidean_squared(Bm, A))
    E = B @ Ktt @ B.T; Ete = Bt @ Kst @ B.T; dte = np.einsum("ij,jk,ik->i", Bt, np.exp(-g * euclidean_squared(Bm)), Bt)
    dE = np.diag(E); MMD2 = np.maximum(dE[:, None] + dE[None, :] - 2 * E, 0); g2 = rbf_med(MMD2)
    return np.exp(-g2 * MMD2), np.exp(-g2 * np.maximum(dte[:, None] + dE[None, :] - 2 * Ete, 0)), (Ktt, Kst, B, Bt)

def run_split(D, fam, tr_sids, te_sids, prep):
    trr = np.concatenate([D["rmap"][s] for s in tr_sids]); ter = np.concatenate([D["rmap"][s] for s in te_sids])
    sc = (QuantileTransformer(n_quantiles=min(300, len(trr)), output_distribution="normal", subsample=10**9, random_state=0)
          if prep == "quantile" else StandardScaler()).fit(D["X"][trr])
    Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
    ytr = np.array([D["sub_label"][s] for s in tr_sids]); yte = np.array([D["sub_label"][s] for s in te_sids])
    yrec = np.repeat(ytr, 3); out = {}
    # uncompressed reference views
    Qtr, S2tr, Rtr = subj_views(Xs, D["rmap"], tr_sids); Qte, S2te, Rte = subj_views(Xs, D["rmap"], te_sids)
    Ke, Kete = kpair(S2tr, S2te); Kg, Kgte = gpair(Qtr, Qte)
    out["E-RBF"] = svc(Ke, ytr, Kete); out["fused w=0.5"] = svc(0.5 * Ke + 0.5 * Kg, ytr, 0.5 * Kete + 0.5 * Kgte)
    for q in QS:
        pls = pls_fit(Xs[trr], yrec, q); Z = np.empty((len(Xs), q)); Z[trr] = pls.transform(Xs[trr]); Z[ter] = pls.transform(Xs[ter])
        Qa, S2a, Ra = subj_views(Z, D["rmap"], tr_sids); Qb, S2b, Rb = subj_views(Z, D["rmap"], te_sids)
        Kz, Kzte = kpair(S2a, S2b); out["PLS%d E-RBF" % q] = svc(Kz, ytr, Kzte)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(S2a[:, :q], ytr)
        out["PLS%d mean-LDA" % q] = lda.decision_function(S2b[:, :q])
        if q >= 3:
            Kgz, Kgzte = gpair(Qa, Qb); out["PLS%d G-only" % q] = svc(Kgz, ytr, Kgzte)
            for w in (0.3, 0.5, 0.7):
                out["PLS%d fused w=%.1f" % (q, w)] = svc((1 - w) * Kz + w * Kgz, ytr, (1 - w) * Kzte + w * Kgzte)
        Km, Kmte, (Ktt, Kst, B, Bt) = setpair(Ra, Rb)
        out["PLS%d MMD" % q] = svc(Km, ytr, Kmte)
        rec = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Ktt, yrec)
        out["PLS%d record->mean" % q] = Bt @ rec.decision_function(Kst)
        if q >= 3:
            Ks = [Kz, Kgz, Km]; Kt = [Kzte, Kgzte, Kmte]; b = alignf(Ks, ytr, 0.25)
            out["PLS%d MKL(E,G,MMD)" % q] = svc(sum(x * k for x, k in zip(b, Ks)), ytr, sum(x * k for x, k in zip(b, Kt)))
    return out, yte

def main():
    rows = []
    for cohort in ("UCI-489", "PD-252"):
        D = load(cohort); fam = np.array([family(c, cohort) for c in D["feats"]])
        for prep in ("standard", "quantile"):
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(D["sids"], D["slab"]), start=1):
                out, yte = run_split(D, fam, D["sids"][tr], D["sids"][te], prep)
                for m, dec in out.items():
                    yp = (dec > 0).astype(int)
                    rows.append(dict(cohort=cohort, prep=prep, fold=fold, method=m, ba=balanced_accuracy_score(yte, yp), f1=f1_score(yte, yp, average="macro")))
                print(cohort, prep, "fold", fold, flush=True)
    df = pd.DataFrame(rows); df.to_csv("robustness_out/newmethods4.csv", index=False)
    for cohort in ("UCI-489", "PD-252"):
        for prep in ("standard", "quantile"):
            d = df[(df.cohort == cohort) & (df.prep == prep)].groupby("method")[["ba", "f1"]].mean().round(3)
            print("\n==", cohort, prep); print(d.sort_values("ba", ascending=False).head(18).to_string())
    print("DONE")

if __name__ == "__main__": main()
