"""VoxNeuro-2 candidate: rank-normalized, supervised-subspace multi-view kernel fusion.

Pipeline inside every training fold (nothing is fitted on held-out subjects):
  1. rank normalization of recordings (quantile -> normal), fitted on training recordings;
  2. a supervised PLS projection of recordings to q components, fitted on training recordings with
     class-balanced targets, giving a low-dimensional space in which subspace distances are informative;
  3. subject views in that space: mean-dispersion summary (Euclidean), rank-3 SVD basis (Grassmann),
     the recording set itself (mean-embedding / MMD);
  4. the prespecified imbalance gate (G-SMOTE along Grassmann geodesics with matched summary interpolation);
  5. median-heuristic RBF kernels, convex fusion, class-weighted SVC (C = 1).
q and the fusion weight are chosen by inner subject-disjoint 5-fold CV on the training subjects only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from family_mkl import load, family, euclidean_squared, chordal_squared, rbf_med, alignf
import voxneuro.method as M

QS = (4, 8, 16, 32)
WS = (0.0, 0.3, 0.5, 0.7)
RANK = 3

def _norm(Xtr_rows, X, prep):
    sc = (QuantileTransformer(n_quantiles=min(300, len(Xtr_rows)), output_distribution="normal", subsample=10**9, random_state=0)
          if prep == "quantile" else StandardScaler()).fit(X[Xtr_rows])
    return sc

def _views(Z, rmap, sids, rank=RANK):
    Q, S2, R = [], [], []
    for s in sids:
        Zi = Z[rmap[s]]; U, _, _ = np.linalg.svd(Zi.T, full_matrices=False)
        Q.append(U[:, :min(rank, U.shape[1])]); S2.append(np.concatenate([Zi.mean(0), Zi.std(0)])); R.append(Zi)
    return Q, np.vstack(S2), R

def _gsmote(Q, S2, y, seed, k=5):
    """Prespecified imbalance gate: returns augmented (Q, S2, y). Identity when the fold is balanced."""
    counts = np.bincount(y, minlength=2); nsyn = int(counts.max() - counts.min())
    mi = np.flatnonzero(y == int(np.argmin(counts)))
    if nsyn == 0 or len(mi) < 2: return list(Q), S2, y
    d = chordal_squared([Q[i] for i in mi]); np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :min(k, len(mi) - 1)]; rng = np.random.default_rng(seed); tri = []
    for _ in range(nsyn):
        a = int(rng.integers(len(mi))); b = int(rng.choice(nn[a])); tri.append((mi[a], mi[b], float(rng.random())))
    Qa = list(Q) + [M.grassmann_geodesic(Q[a], Q[b], t) for a, b, t in tri]
    S2a = np.vstack([S2] + [S2[a] + t * (S2[b] - S2[a]) for a, b, t in tri])
    return Qa, S2a, np.concatenate([y, np.full(len(tri), int(np.argmin(counts)))])

def _setk(Rtr, Rte):
    A = np.vstack(Rtr); Bm = np.vstack(Rte); n, m = len(Rtr), len(Rte)
    B = np.zeros((n, 3 * n)); Bt = np.zeros((m, 3 * m))
    for i in range(n): B[i, 3 * i:3 * i + 3] = 1 / 3
    for i in range(m): Bt[i, 3 * i:3 * i + 3] = 1 / 3
    d = euclidean_squared(A); g = rbf_med(d); Ktt = np.exp(-g * d); Kst = np.exp(-g * euclidean_squared(Bm, A))
    E = B @ Ktt @ B.T; Ete = Bt @ Kst @ B.T; dte = np.einsum("ij,jk,ik->i", Bt, np.exp(-g * euclidean_squared(Bm)), Bt)
    dE = np.diag(E); MMD2 = np.maximum(dE[:, None] + dE[None, :] - 2 * E, 0); g2 = rbf_med(MMD2)
    return np.exp(-g2 * MMD2), np.exp(-g2 * np.maximum(dte[:, None] + dE[None, :] - 2 * Ete, 0)), Ktt, Kst, B, Bt

def _svc(K, y, Kte):
    return SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(K, y).decision_function(Kte)

def split_decisions(D, fam, tr_sids, te_sids, seed, preps=("standard", "quantile")):
    """Decision values on held-out subjects for every candidate configuration of the new pipeline plus baselines."""
    trr = np.concatenate([D["rmap"][s] for s in tr_sids]); ter = np.concatenate([D["rmap"][s] for s in te_sids])
    ytr = np.array([D["sub_label"][s] for s in tr_sids]); yte = np.array([D["sub_label"][s] for s in te_sids])
    out = {}
    for prep in preps:
        p = prep[0]; sc = _norm(trr, D["X"], prep)
        Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        # ---- baselines in the original feature space (paper protocol) ----
        Q0, S0, R0 = _views(Xs, D["rmap"], tr_sids); Q1, S1, R1 = _views(Xs, D["rmap"], te_sids)
        Qa, Sa, ya = _gsmote(Q0, S0, ytr, seed)
        de = euclidean_squared(Sa); ge = rbf_med(de); Ke, Kete = np.exp(-ge * de), np.exp(-ge * euclidean_squared(S1, Sa))
        dg = chordal_squared(Qa); gg = rbf_med(dg); Kg, Kgte = np.exp(-gg * dg), np.exp(-gg * chordal_squared(Q1, Qa))
        out["E-RBF/" + p] = _svc(Ke, ya, Kete)
        out["fused w=0.5/" + p] = _svc(0.5 * Ke + 0.5 * Kg, ya, 0.5 * Kete + 0.5 * Kgte)
        # ---- supervised subspace pipeline ----
        yrec = np.repeat(ytr, 3)
        t = np.where(yrec == 1, 1.0 / max((yrec == 1).sum(), 1), -1.0 / max((yrec == 0).sum(), 1))
        for q in QS:
            pls = PLSRegression(n_components=q, scale=False).fit(Xs[trr], t)
            Z = np.empty((len(Xs), q)); Z[trr] = pls.transform(Xs[trr]); Z[ter] = pls.transform(Xs[ter])
            Qz, Sz, Rz = _views(Z, D["rmap"], tr_sids); Qw, Sw, Rw = _views(Z, D["rmap"], te_sids)
            Qz2, Sz2, y2 = _gsmote(Qz, Sz, ytr, seed)
            dz = euclidean_squared(Sz2); gz = rbf_med(dz); Kz, Kzte = np.exp(-gz * dz), np.exp(-gz * euclidean_squared(Sw, Sz2))
            dgz = chordal_squared(Qz2); ggz = rbf_med(dgz); Kgz, Kgzte = np.exp(-ggz * dgz), np.exp(-ggz * chordal_squared(Qw, Qz2))
            for w in WS:
                out["V2 q=%d w=%.1f/%s" % (q, w, p)] = _svc((1 - w) * Kz + w * Kgz, y2, (1 - w) * Kzte + w * Kgzte)
            Km, Kmte, Ktt, Kst, B, Bt = _setk(Rz, Rw)
            out["V2 q=%d MMD/%s" % (q, p)] = _svc(Km, ytr, Kmte)
            out["V2 q=%d record/%s" % (q, p)] = Bt @ SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Ktt, yrec).decision_function(Kst)
            Ks = [Kz[:len(ytr), :len(ytr)], Kgz[:len(ytr), :len(ytr)], Km]; Kt = [Kzte[:, :len(ytr)], Kgzte[:, :len(ytr)], Kmte]
            b = alignf(Ks, ytr, 0.25)
            out["V2 q=%d MKL/%s" % (q, p)] = _svc(sum(x * k for x, k in zip(b, Ks)), ytr, sum(x * k for x, k in zip(b, Kt)))
            # no-augmentation ablation of the fused configuration
            dz0 = euclidean_squared(Sz); gz0 = rbf_med(dz0); Kz0, Kz0te = np.exp(-gz0 * dz0), np.exp(-gz0 * euclidean_squared(Sw, Sz))
            dg0 = chordal_squared(Qz); gg0 = rbf_med(dg0); Kg0, Kg0te = np.exp(-gg0 * dg0), np.exp(-gg0 * chordal_squared(Qw, Qz))
            for w in (0.0, 0.3, 0.5):
                out["V2n q=%d w=%.1f/%s" % (q, w, p)] = _svc((1 - w) * Kz0 + w * Kg0, ytr, (1 - w) * Kz0te + w * Kg0te)
    return out, yte

GRID = ["V2 q=%d w=%.1f" % (q, w) for q in QS for w in WS]
GRIDN = ["V2n q=%d w=%.1f" % (q, w) for q in QS for w in (0.0, 0.3, 0.5)]

def evaluate(D, fam, part_seed, seed_of_fold, preps=("standard", "quantile"), inner_seed=7):
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
        ps = [x[0] for x in preps]
        for grid, tag in ((GRID, "V2 nested"), (GRIDN, "V2n nested")):
            for p in ps:
                best = max(grid, key=lambda c: (sc(c + "/" + p), -float(c.split("q=")[1].split(" ")[0]), -float(c.split("w=")[1])))
                out[tag + "/" + p] = out[best + "/" + p]
            if len(ps) > 1:
                allc = [c + "/" + p for c in grid for p in ps]
                bj = max(allc, key=lambda c: (sc(c), -float(c.split("q=")[1].split(" ")[0]), -float(c.split("w=")[1].split("/")[0])))
                out[tag + "+prep"] = out[bj]
                if tag == "V2 nested": chosen.append(bj)
        if len(ps) > 1:
            bb = max([b + "/" + p for b in ("E-RBF", "fused w=0.5") for p in ps], key=sc); out["baseline nested+prep"] = out[bb]
        for m, dec in out.items():
            yp = (dec > 0).astype(int); acc.setdefault(m, []).append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
    return {m: (float(np.mean([v[0] for v in a])), float(np.mean([v[1] for v in a]))) for m, a in acc.items()}, chosen
