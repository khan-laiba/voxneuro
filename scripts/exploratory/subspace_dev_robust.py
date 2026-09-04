"""Robustness harness for the unified VoxNeuro-2 candidate (rank normalization + optional supervised subspace,
fused Grassmann-Euclidean kernel, w = 0.5 prespecified), with the subspace dimension chosen by inner CV among
{8, 16, all}. 24 augmentation seeds (PD-252) + 20 outer partitions on both cohorts; 5 worker processes."""
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from family_mkl import load, family, euclidean_squared, chordal_squared, rbf_med
from subspace_dev import _norm, _views, _gsmote, _svc

QS = (8, 16, 32)

def fused_pair(Q0, S0, Q1, S1, y, seed, aug, w):
    Qa, Sa, ya = _gsmote(Q0, S0, y, seed) if aug else (list(Q0), S0, y)
    de = euclidean_squared(Sa); ge = rbf_med(de); Ke, Kete = np.exp(-ge * de), np.exp(-ge * euclidean_squared(S1, Sa))
    dg = chordal_squared(Qa); gg = rbf_med(dg); Kg, Kgte = np.exp(-gg * dg), np.exp(-gg * chordal_squared(Q1, Qa))
    return {ww: _svc((1 - ww) * Ke + ww * Kg, ya, (1 - ww) * Kete + ww * Kgte) for ww in w}

def split_decisions(D, tr_sids, te_sids, seed, full=True):
    trr = np.concatenate([D["rmap"][s] for s in tr_sids]); ter = np.concatenate([D["rmap"][s] for s in te_sids])
    ytr = np.array([D["sub_label"][s] for s in tr_sids]); yte = np.array([D["sub_label"][s] for s in te_sids]); out = {}
    if full:   # standard-prep baselines (paper protocol)
        sc = _norm(trr, D["X"], "standard"); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        Q0, S0, _ = _views(Xs, D["rmap"], tr_sids); Q1, S1, _ = _views(Xs, D["rmap"], te_sids)
        r = fused_pair(Q0, S0, Q1, S1, ytr, seed, True, (0.0, 0.5)); out["E-RBF/s"] = r[0.0]; out["fused w=0.5/s"] = r[0.5]
    sc = _norm(trr, D["X"], "quantile"); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
    Q0, S0, _ = _views(Xs, D["rmap"], tr_sids); Q1, S1, _ = _views(Xs, D["rmap"], te_sids)
    r = fused_pair(Q0, S0, Q1, S1, ytr, seed, True, (0.0, 0.5)); out["E-RBF/q"] = r[0.0]; out["fused w=0.5/q"] = r[0.5]
    if full:
        r = fused_pair(Q0, S0, Q1, S1, ytr, seed, False, (0.0, 0.5)); out["E-RBF/q no-aug"] = r[0.0]; out["fused w=0.5/q no-aug"] = r[0.5]
    yrec = np.repeat(ytr, 3); t = np.where(yrec == 1, 1.0 / max((yrec == 1).sum(), 1), -1.0 / max((yrec == 0).sum(), 1))
    for q in QS:
        pls = PLSRegression(n_components=q, scale=False).fit(Xs[trr], t)
        Z = np.empty((len(Xs), q)); Z[trr] = pls.transform(Xs[trr]); Z[ter] = pls.transform(Xs[ter])
        Qz, Sz, _ = _views(Z, D["rmap"], tr_sids); Qw, Sw, _ = _views(Z, D["rmap"], te_sids)
        r = fused_pair(Qz, Sz, Qw, Sw, ytr, seed, False, (0.3, 0.5) if full else (0.5,))
        for ww, dec in r.items(): out["V2 q=%d w=%.1f" % (q, ww)] = dec
    if full:
        for ww in (0.3, 0.5): out["V2 ens w=%.1f" % ww] = np.mean([out["V2 q=%d w=%.1f" % (q, ww)] for q in QS], axis=0)
    return out, yte

MENU = ["V2 q=8 w=0.5", "V2 q=16 w=0.5", "V2 q=32 w=0.5", "fused w=0.5/q"]

def evaluate(D, part_seed, seed_of_fold, inner_seed=7):
    acc, chosen = {}, []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=part_seed).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]; seed = seed_of_fold(fold)
        out, yte = split_decisions(D, tid, eid, seed, full=True)
        ytr = np.array([D["sub_label"][s] for s in tid]); inner = {m: np.zeros(len(tid)) for m in MENU}
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=inner_seed).split(np.zeros(len(tid)), ytr), start=1):
            o, _ = split_decisions(D, tid[a], tid[b], seed * 100 + k, full=False)
            for m in MENU: inner[m][b] = o[m]
        def sc(m):
            yp = (inner[m] > 0).astype(int); return 0.5 * (balanced_accuracy_score(ytr, yp) + f1_score(ytr, yp, average="macro"))
        best4 = max(MENU, key=lambda m: (sc(m), MENU.index(m) == 3, -MENU.index(m)))      # ties -> 'all', then smaller q
        best3 = max([MENU[0], MENU[1], MENU[3]], key=lambda m: (sc(m), MENU.index(m) == 3, -MENU.index(m)))
        out["V2 unified nested{8,16,32,all}"] = out[best4]; out["V2 unified nested{8,16,all}"] = out[best3]; chosen.append(best3)
        for m, dec in out.items():
            yp = (dec > 0).astype(int); acc.setdefault(m, []).append((balanced_accuracy_score(yte, yp), f1_score(yte, yp, average="macro")))
    return {m: (float(np.mean([v[0] for v in a])), float(np.mean([v[1] for v in a]))) for m, a in acc.items()}, chosen

JOBS = [("seed", "PD-252", c, 42, (1000 + c) * 1000) for c in range(24)]
for cohort in ("UCI-489", "PD-252"): JOBS += [("partition", cohort, ps, ps, 42 * 1000) for ps in range(20)]
_C = {}
def work(job):
    study, cohort, idx, part_seed, base = job
    if cohort not in _C: _C[cohort] = load(cohort)
    t0 = time.time(); res, chosen = evaluate(_C[cohort], part_seed, lambda f: base + f)
    return dict(study=study, cohort=cohort, index=idx, secs=round(time.time() - t0, 1), chosen="|".join(chosen),
                **{m + "_ba": v[0] for m, v in res.items()}, **{m + "_f1": v[1] for m, v in res.items()})

if __name__ == "__main__":
    rows = []
    with Pool(5, maxtasksperchild=8) as p:
        for r in p.imap_unordered(work, JOBS):
            rows.append(r); pd.DataFrame(rows).to_csv("robustness_out/vox3_robust.csv", index=False)
            print(r["study"], r["cohort"], r["index"], "%ss" % r["secs"], {k[:-3]: round(r[k], 3) for k in ("fused w=0.5/s_ba", "fused w=0.5/q_ba", "V2 ens w=0.5_ba", "V2 unified nested{8,16,all}_ba")}, r["chosen"], flush=True)
    print("DONE", len(rows))
