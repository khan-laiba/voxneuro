"""Robustness harness for the VoxNeuro-2 candidate: 24 augmentation seeds (PD-252, fixed folds) and
20 outer subject partitions on both cohorts, with every selection nested inside training folds."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
from vox2 import load, family, evaluate

JOBS = []
for c in range(24): JOBS.append(("seed", "PD-252", c, 42, (1000 + c) * 1000))
for cohort in ("UCI-489", "PD-252"):
    for ps in range(20): JOBS.append(("partition", cohort, ps, ps, 42 * 1000))

_C = {}
def work(job):
    study, cohort, idx, part_seed, base = job
    if cohort not in _C:
        D = load(cohort); _C[cohort] = (D, np.array([family(c, cohort) for c in D["feats"]]))
    D, fam = _C[cohort]
    t0 = time.time(); res, chosen = evaluate(D, fam, part_seed, lambda f: base + f)
    return dict(study=study, cohort=cohort, index=idx, secs=round(time.time() - t0, 1), chosen="|".join(chosen),
                **{m + "_ba": v[0] for m, v in res.items()}, **{m + "_f1": v[1] for m, v in res.items()})

if __name__ == "__main__":
    rows = []
    with Pool(5, maxtasksperchild=8) as p:
        for r in p.imap_unordered(work, JOBS):
            rows.append(r); pd.DataFrame(rows).to_csv("robustness_out/vox2_robust.csv", index=False)
            print(r["study"], r["cohort"], r["index"], "%ss" % r["secs"],
                  {k[:-3]: round(r[k], 3) for k in ("E-RBF/s_ba", "fused w=0.5/s_ba", "MKL nested/q_ba", "MKL nested+prep_ba")}, flush=True)
    print("DONE", len(rows))
