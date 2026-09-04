"""Unified-candidate sweep: ensembles of complementary subject-level learners (full-space fused, supervised-subspace
fused at three scales, recording-level SVM) under one normalizer. 24 seeds (PD) + 20 partitions (both cohorts)."""
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from family_mkl import load, euclidean_squared, chordal_squared, rbf_med
from subspace_dev import _norm, _views, _gsmote, _svc, _setk

def fused_dec(Q0, S0, Q1, S1, y, w):
    de = euclidean_squared(S0); ge = rbf_med(de); Ke, Kete = np.exp(-ge * de), np.exp(-ge * euclidean_squared(S1, S0))
    dg = chordal_squared(Q0); gg = rbf_med(dg); Kg, Kgte = np.exp(-gg * dg), np.exp(-gg * chordal_squared(Q1, Q0))
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit((1 - w) * Ke + w * Kg, y)
    return clf.decision_function((1 - w) * Kete + w * Kgte), clf.decision_function((1 - w) * Ke + w * Kg).std() + 1e-12

def split(D, tr_sids, te_sids, seed):
    trr = np.concatenate([D["rmap"][s] for s in tr_sids]); ter = np.concatenate([D["rmap"][s] for s in te_sids])
    ytr = np.array([D["sub_label"][s] for s in tr_sids]); yte = np.array([D["sub_label"][s] for s in te_sids]); dec, sc_ = {}, {}
    for prep in ("standard", "quantile"):
        p = prep[0]; sc = _norm(trr, D["X"], prep); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        Q0, S0, R0 = _views(Xs, D["rmap"], tr_sids); Q1, S1, R1 = _views(Xs, D["rmap"], te_sids)
        Qa, Sa, ya = _gsmote(Q0, S0, ytr, seed)
        dec["fused/" + p], sc_["fused/" + p] = fused_dec(Qa, Sa, Q1, S1, ya, 0.5)
        if p == "s": dec["E-RBF/s"], sc_["E-RBF/s"] = fused_dec(Qa, Sa, Q1, S1, ya, 0.0)
        Km, Kmte, Ktt, Kst, B, Bt = _setk(R0, R1)
        rec = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Ktt, np.repeat(ytr, 3))
        dec["record/" + p] = Bt @ rec.decision_function(Kst); sc_["record/" + p] = (B @ rec.decision_function(Ktt)).std() + 1e-12
        if p == "q":
            yrec = np.repeat(ytr, 3); t = np.where(yrec == 1, 1.0 / (yrec == 1).sum(), -1.0 / (yrec == 0).sum())
            for q in (8, 16, 32):
                pls = PLSRegression(n_components=q, scale=False).fit(Xs[trr], t); Z = np.empty((len(Xs), q)); Z[trr] = pls.transform(Xs[trr]); Z[ter] = pls.transform(Xs[ter])
                Qz, Sz, _ = _views(Z, D["rmap"], tr_sids); Qw, Sw, _ = _views(Z, D["rmap"], te_sids)
                dec["V2 q=%d" % q], sc_["V2 q=%d" % q] = fused_dec(Qz, Sz, Qw, Sw, ytr, 0.5)
    z = lambda m: dec[m] / sc_[m]
    out = dict(dec)
    out["V2 ens"] = np.mean([z("V2 q=%d" % q) for q in (8, 16, 32)], axis=0)
    out["U1: V2{8,16,32}+record/q"] = np.mean([z("V2 q=%d" % q) for q in (8, 16, 32)] + [z("record/q")], axis=0)
    out["U2: V2{8,16,32}+fused/q+record/q"] = np.mean([z("V2 q=%d" % q) for q in (8, 16, 32)] + [z("fused/q"), z("record/q")], axis=0)
    out["U3: V2ens+record/q (equal)"] = 0.5 * out["V2 ens"] / (out["V2 ens"].std() + 1e-12) + 0.5 * z("record/q")
    out["U4: V2{8,16,32}+fused/q"] = np.mean([z("V2 q=%d" % q) for q in (8, 16, 32)] + [z("fused/q")], axis=0)
    out["U5: fused/q+record/q"] = 0.5 * z("fused/q") + 0.5 * z("record/q")
    out["U6: V2{8,16,32}+record/s"] = np.mean([z("V2 q=%d" % q) for q in (8, 16, 32)] + [z("record/s")], axis=0)
    out["U7: fused/s+record/s"] = 0.5 * z("fused/s") + 0.5 * z("record/s")
    return out, yte

def evaluate(D, part_seed, base):
    acc = {}
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=part_seed).split(D["sids"], D["slab"]), start=1):
        out, yte = split(D, D["sids"][tr], D["sids"][te], base + fold)
        for m, d in out.items():
            yp = (d > 0).astype(int); acc.setdefault(m, []).append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
    return {m: (float(np.mean([v[0] for v in a])), float(np.mean([v[1] for v in a]))) for m, a in acc.items()}

JOBS = [("seed", "PD-252", c, 42, (1000 + c) * 1000) for c in range(24)]
for cohort in ("UCI-489", "PD-252"): JOBS += [("partition", cohort, ps, ps, 42 * 1000) for ps in range(20)]
_C = {}
def work(job):
    study, cohort, idx, part_seed, base = job
    if cohort not in _C: _C[cohort] = load(cohort)
    res = evaluate(_C[cohort], part_seed, base)
    return dict(study=study, cohort=cohort, index=idx, **{m + "_ba": v[0] for m, v in res.items()}, **{m + "_f1": v[1] for m, v in res.items()})

if __name__ == "__main__":
    rows = []
    with Pool(5, maxtasksperchild=8) as p:
        for r in p.imap_unordered(work, JOBS):
            rows.append(r); pd.DataFrame(rows).to_csv("robustness_out/unified_sweep.csv", index=False); print(r["study"], r["cohort"], r["index"], flush=True)
    df = pd.DataFrame(rows)
    for study, cohort in (("seed", "PD-252"), ("partition", "PD-252"), ("partition", "UCI-489")):
        s = df[(df.study == study) & (df.cohort == cohort)]; print("\n== %s %s n=%d" % (study, cohort, len(s)))
        ms = sorted({c[:-3] for c in s.columns if c.endswith("_ba")}, key=lambda m: -s[m + "_ba"].mean())
        for m in ms:
            ba, f1 = s[m + "_ba"], s[m + "_f1"]; pb, pf = s["fused/s_ba"], s["fused/s_f1"]
            print("%-36s BA %.3f±%.3f F1 %.3f±%.3f | vs paper fused BA%+.3f (%2d) F1%+.3f (%2d)" % (m, ba.mean(), ba.std(), f1.mean(), f1.std(), (ba - pb).mean(), (ba > pb).sum(), (f1 - pf).mean(), (f1 > pf).sum()))
    print("DONE")
