"""Prespecified exploration of principled fusion variants (selection on training data only).
Views: U = uncentered rank-3 Grassmann (paper), C = centered rank-2 Grassmann (within-subject deviations).
Fusion: sum K = w K_G + (1-w) K_E over a grid of w; product K = K_G * K_E.
Weight selection: (i) inner 5-fold CV on the training subjects (BA), (ii) kernel-target alignment on the training kernel.
Outer protocol identical to the paper (partition seed 42, default augmentation seed rule)."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
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
    X = frame[feats].to_numpy(float); rid = frame[idc].to_numpy(str); rl = frame[labc].to_numpy(int)
    sf = frame[[idc, labc]].drop_duplicates(idc)
    return dict(X=X, rid=rid, rl=rl, sids=sf[idc].to_numpy(str), slab=sf[labc].to_numpy(int), rmap=M._subject_rows(rid))

def views(Xs, rid, rl, sids, centered):
    """Grassmann + Euclidean views; centered -> rank-2 span of within-subject deviations."""
    Q, S, Y = [], [], []
    for sid in sids:
        idx = np.flatnonzero(rid == sid); Xi = Xs[idx]
        mean = Xi.mean(axis=0); std = Xi.std(axis=0, ddof=0)
        Mm = (Xi - mean).T if centered else Xi.T
        U, s, _ = np.linalg.svd(Mm, full_matrices=False)
        Q.append(U[:, :2] if centered else U[:, :3]); S.append(np.concatenate([mean, std])); Y.append(int(rl[idx[0]]))
    return Q, np.vstack(S), np.asarray(Y)

def fit_eval(Qtr, s_tr, ytr, Qte, s_te, aug_seed, w_list, product=False):
    Qa, sa, ya, _ = M._g_smote(Qtr, s_tr, ytr, random_state=aug_seed, neighbors=5)
    dg, de = M.chordal_squared(Qa), M.euclidean_squared(sa); gg, ge = M._median_gamma(dg), M._median_gamma(de)
    Kg, Ke = np.exp(-gg * dg), np.exp(-ge * de)
    Kg_te, Ke_te = np.exp(-gg * M.chordal_squared(Qte, Qa)), np.exp(-ge * M.euclidean_squared(s_te, sa))
    out = {}
    for w in w_list:
        K, Kt = (Kg * Ke, Kg_te * Ke_te) if product else (w * Kg + (1 - w) * Ke, w * Kg_te + (1 - w) * Ke_te)
        clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(K, ya)
        out[w] = (clf.decision_function(Kt) > 0).astype(int)
        if product: break
    return out, (Kg, Ke, ya)

def alignment_w(Kg, Ke, ya):
    yy = np.outer(2 * ya - 1, 2 * ya - 1); best, bw = -1, 0.5
    for w in GRID:
        K = w * Kg + (1 - w) * Ke; a = (K * yy).sum() / (np.linalg.norm(K) * np.linalg.norm(yy))
        if a > best: best, bw = a, w
    return bw

def inner_cv_w(Qtr, s_tr, ytr, aug_seed):
    inner = StratifiedKFold(5, shuffle=True, random_state=7); score = {w: [] for w in GRID}
    for k, (a, b) in enumerate(inner.split(np.zeros(len(ytr)), ytr), start=1):
        preds, _ = fit_eval([Qtr[i] for i in a], s_tr[a], ytr[a], [Qtr[i] for i in b], s_tr[b], aug_seed * 10 + k, GRID)
        for w in GRID: score[w].append(balanced_accuracy_score(ytr[b], preds[w]))
    means = {w: np.mean(v) for w, v in score.items()}
    return max(GRID, key=lambda w: (means[w], -abs(w - 0.5)))   # ties -> closest to 0.5

def run(cohort, part_seed=42):
    D = load(cohort); rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=part_seed).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]
        trr = np.concatenate([D["rmap"][s] for s in tid]); ter = np.concatenate([D["rmap"][s] for s in eid])
        sc = StandardScaler().fit(D["X"][trr]); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        aug = 42 * 1000 + fold
        for view in ("U", "C"):
            Qtr, s_tr, ytr = views(Xs, D["rid"], D["rl"], tid, view == "C"); Qte, s_te, yte = views(Xs, D["rid"], D["rl"], eid, view == "C")
            preds, (Kg, Ke, ya) = fit_eval(Qtr, s_tr, ytr, Qte, s_te, aug, GRID)
            for w in GRID: rows.append(dict(cohort=cohort, fold=fold, view=view, method="fixed", w=w, ba=balanced_accuracy_score(yte, preds[w]), f1=f1_score(yte, preds[w], average="macro")))
            pp, _ = fit_eval(Qtr, s_tr, ytr, Qte, s_te, aug, [None], product=True); yp = pp[None]
            rows.append(dict(cohort=cohort, fold=fold, view=view, method="product", w=np.nan, ba=balanced_accuracy_score(yte, yp), f1=f1_score(yte, yp, average="macro")))
            wa = alignment_w(Kg, Ke, ya); rows.append(dict(cohort=cohort, fold=fold, view=view, method="alignment", w=wa, ba=balanced_accuracy_score(yte, preds[wa]), f1=f1_score(yte, preds[wa], average="macro")))
            wn = inner_cv_w(Qtr, s_tr, ytr, aug); rows.append(dict(cohort=cohort, fold=fold, view=view, method="nested", w=wn, ba=balanced_accuracy_score(yte, preds[wn]), f1=f1_score(yte, preds[wn], average="macro")))
        print(cohort, "fold", fold, "done", flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.concat([run("UCI-489"), run("PD-252")]); df.to_csv("robustness_out/fusion_exploration.csv", index=False)
    pd.set_option("display.width", 220)
    for cohort in ("UCI-489", "PD-252"):
        d = df[df.cohort == cohort]
        print("\n==", cohort, "fixed-w sweep (outer means; diagnostic only)")
        piv = d[d.method == "fixed"].groupby(["view", "w"])[["ba", "f1"]].mean().round(3).unstack(0); print(piv.to_string())
        for meth in ("product", "alignment", "nested"):
            for view in ("U", "C"):
                s = d[(d.method == meth) & (d.view == view)]
                print("%-9s view %s: BA %.3f ± %.3f  F1 %.3f ± %.3f  chosen w: %s" % (meth, view, s.ba.mean(), s.ba.std(ddof=1) / np.sqrt(5), s.f1.mean(), s.f1.std(ddof=1) / np.sqrt(5), list(s.w.round(1))))
    print("DONE")
