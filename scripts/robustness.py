"""Robustness analyses for the VoxNeuro manuscript (all under the paper's fixed protocol).

Outputs (CSV, in --out):
  unified_comparators.csv   per-cohort summary of every comparator (mean +- fold SE, pooled confusions)
  unified_fold_metrics.csv  per-fold metrics for the same
  oof_predictions_fused.csv fused-model out-of-fold predictions (both cohorts)
  partition_sensitivity.csv 20 outer subject partitions (StratifiedKFold seeds 0..19): fused / RBF-only / Grassmann-only
  rank_sensitivity.csv      rank r = 2 versus r = 3 at the default partition
  permutation_sensitivity.csv  feature-family permutation of the Euclidean view (PD-252, 20 draws per family and fold)
  family_map.csv            column -> feature family assignment used for the permutation analysis
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import voxneuro.method as M

AUG = lambda fold: 42 * 1000 + fold      # package default G-SMOTE seed rule
SM = lambda fold: 42 + fold              # package default SMOTE seed rule

def load(name, uci_csv, pd_csv):
    if name == "UCI-489":
        frame, feats = M._validate_frame(pd.read_csv(uci_csv), "ID", "Status", ("Recording",)); idc, labc = "ID", "Status"
    else:
        frame, feats = M._validate_frame(pd.read_csv(pd_csv), "id", "class", ()); idc, labc = "id", "class"
    X = frame[feats].to_numpy(float); rid = frame[idc].to_numpy(str); rl = frame[labc].to_numpy(int)
    sf = frame[[idc, labc]].drop_duplicates(idc)
    return dict(X=X, rid=rid, rl=rl, sids=sf[idc].to_numpy(str), slab=sf[labc].to_numpy(int), rmap=M._subject_rows(rid), feats=feats)

def make_folds(D, part_seed, rank):
    out = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=part_seed).split(D["sids"], D["slab"]), start=1):
        tid, eid = D["sids"][tr], D["sids"][te]
        trr = np.concatenate([D["rmap"][s] for s in tid]); ter = np.concatenate([D["rmap"][s] for s in eid])
        sc = StandardScaler().fit(D["X"][trr]); Xs = np.empty_like(D["X"]); Xs[trr] = sc.transform(D["X"][trr]); Xs[ter] = sc.transform(D["X"][ter])
        Qtr, s_tr, ytr = M.build_subject_views(Xs, D["rid"], D["rl"], tid, rank, True)
        Qte, s_te, yte = M.build_subject_views(Xs, D["rid"], D["rl"], eid, rank, True)
        out.append(dict(fold=fold, tid=tid, eid=eid, Qtr=Qtr, s_tr=s_tr, ytr=ytr, Qte=Qte, s_te=s_te, yte=yte))
    return out

def conf(y, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel(); return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

def rec(fold, y, yp, **extra):
    return dict(fold=fold, ba=balanced_accuracy_score(y, yp), f1=f1_score(y, yp, average="macro"), **conf(y, yp), **extra)

def kernel_models(F, weights=(0.5, 0.0, 1.0), want_preds=False, want_fit=False):
    res = {w: [] for w in weights}; preds = []; fits = []
    for f in F:
        Qa, sa, ya, nsyn = M._g_smote(f["Qtr"], f["s_tr"], f["ytr"], random_state=AUG(f["fold"]), neighbors=5)
        dg, de = M.chordal_squared(Qa), M.euclidean_squared(sa); gg, ge = M._median_gamma(dg), M._median_gamma(de)
        Kg_te = np.exp(-gg * M.chordal_squared(f["Qte"], Qa)); Ke_te = np.exp(-ge * M.euclidean_squared(f["s_te"], sa))
        for w in weights:
            clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(w * np.exp(-gg * dg) + (1 - w) * np.exp(-ge * de), ya)
            dec = clf.decision_function(w * Kg_te + (1 - w) * Ke_te); yp = (dec > 0).astype(int)
            res[w].append(rec(f["fold"], f["yte"], yp, n_synthetic=nsyn))
            if want_preds and w == 0.5:
                preds.append(pd.DataFrame(dict(fold=f["fold"], subject_id=f["eid"], true_label=f["yte"], predicted_label=yp, decision_score=dec)))
            if want_fit and w == 0.5:
                fits.append(dict(clf=clf, sa=sa, gg=gg, ge=ge, Kg_te=Kg_te, Qa=Qa))
    return res, preds, fits

def euclid_models(F):
    names = ["balanced_logreg", "balanced_linear_svm", "smote_logreg", "smote_rbf"]; res = {n: [] for n in names}
    for f in F:
        s_tr, ytr, s_te, yte = f["s_tr"], f["ytr"], f["s_te"], f["yte"]
        lr = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42).fit(s_tr, ytr)
        lin = LinearSVC(class_weight="balanced", C=1.0, max_iter=20000, random_state=42).fit(s_tr, ytr)
        counts = np.bincount(ytr); mn = int(counts.min())
        if mn >= 2 and counts[0] != counts[1]:
            s_rs, y_rs = SMOTE(random_state=SM(f["fold"]), k_neighbors=min(5, mn - 1)).fit_resample(s_tr, ytr)
        else:
            s_rs, y_rs = s_tr, ytr
        lr2 = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42).fit(s_rs, y_rs)
        de = M.euclidean_squared(s_rs); ge = M._median_gamma(de)
        rbf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(np.exp(-ge * de), y_rs)
        for n, model, n_syn in ((names[0], lr, 0), (names[1], lin, 0), (names[2], lr2, len(y_rs) - len(ytr))):
            # the class-weighted logistic regression and linear SVM are fitted on the original summaries (no synthetic rows)
            res[n].append(rec(f["fold"], yte, (model.decision_function(s_te) > 0).astype(int), n_synthetic=n_syn))
        yp = (rbf.decision_function(np.exp(-ge * M.euclidean_squared(s_te, s_rs))) > 0).astype(int)
        res[names[3]].append(rec(f["fold"], yte, yp, n_synthetic=len(y_rs) - len(ytr)))
    return res

def summarize(rows):
    df = pd.DataFrame(rows); se = lambda c: float(df[c].std(ddof=1) / np.sqrt(len(df)))
    return dict(ba_mean=df.ba.mean(), ba_se=se("ba"), f1_mean=df.f1.mean(), f1_se=se("f1"),
                TN=int(df.TN.sum()), FP=int(df.FP.sum()), FN=int(df.FN.sum()), TP=int(df.TP.sum()), n_synthetic_total=int(df.n_synthetic.sum()))

def family(col):
    c = col.lower()
    if c == "gender": return "Demographics"
    if c.startswith("tqwt"): return "TQWT"
    if "mfcc" in c or "delta" in c or "log_energy" in c: return "MFCC"
    if c.startswith("imf"): return "IMF/EMD"
    if c.startswith(("det_", "app_", "ea", "ed")): return "Wavelet"
    return "Vocal/time-frequency"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--uci", default="data/ReplicatedAcousticFeatures-ParkinsonDatabase.csv")
    ap.add_argument("--pd", default="data/pd_speech_features_clean.csv"); ap.add_argument("--out", default="robustness_out")
    ap.add_argument("--partitions", type=int, default=20); ap.add_argument("--draws", type=int, default=20); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    KM = {0.5: "fused_w0.5", 0.0: "euclidean_rbf_w0", 1.0: "grassmann_only_w1"}
    unified, foldrows, oof = [], [], []
    part_rows, rank_rows = [], []
    for cohort in ("UCI-489", "PD-252"):
        D = load(cohort, a.uci, a.pd)
        # ---- unified comparators at the paper's partition (seed 42), rank 3, default augmentation seeds
        F = make_folds(D, 42, 3)
        kres, preds, fits = kernel_models(F, want_preds=True, want_fit=(cohort == "PD-252"))
        eres = euclid_models(F)
        for w, rows in kres.items():
            unified.append(dict(cohort=cohort, model=KM[w], **summarize(rows))); foldrows += [dict(cohort=cohort, model=KM[w], **r) for r in rows]
        for n, rows in eres.items():
            unified.append(dict(cohort=cohort, model=n, **summarize(rows))); foldrows += [dict(cohort=cohort, model=n, **r) for r in rows]
        oof.append(pd.concat(preds).assign(cohort=cohort))
        print(cohort, "unified done", flush=True)
        # ---- partition sensitivity
        for ps in range(a.partitions):
            Fp = make_folds(D, ps, 3); kr, _, _ = kernel_models(Fp)
            row = dict(cohort=cohort, partition_seed=ps)
            for w, rows in kr.items():
                s = summarize(rows); row[KM[w] + "_ba"] = s["ba_mean"]; row[KM[w] + "_f1"] = s["f1_mean"]
            er = euclid_models(Fp)
            for n in ("smote_rbf", "balanced_logreg"):
                s = summarize(er[n]); row[n + "_ba"] = s["ba_mean"]; row[n + "_f1"] = s["f1_mean"]
            part_rows.append(row)
        print(cohort, "partitions done", flush=True)
        # ---- rank sensitivity (r = 2 vs r = 3) at the default partition
        for r in (2, 3):
            Fr = make_folds(D, 42, r); kr, _, _ = kernel_models(Fr)
            for w, rows in kr.items():
                s = summarize(rows); rank_rows.append(dict(cohort=cohort, rank=r, model=KM[w], ba_mean=s["ba_mean"], ba_se=s["ba_se"], f1_mean=s["f1_mean"], f1_se=s["f1_se"]))
        print(cohort, "rank done", flush=True)
        # ---- permutation sensitivity (PD-252 only; families of the source dataset)
        if cohort == "PD-252":
            fam = np.array([family(c) for c in D["feats"]]); d = len(D["feats"])
            pd.DataFrame(dict(column=D["feats"], family=fam)).to_csv(os.path.join(a.out, "family_map.csv"), index=False)
            print("family counts:", dict(pd.Series(fam).value_counts()), flush=True)
            prow = []
            for f, fit in zip(F, fits):
                base_dec = fit["clf"].decision_function(0.5 * fit["Kg_te"] + 0.5 * np.exp(-fit["ge"] * M.euclidean_squared(f["s_te"], fit["sa"])))
                base = f1_score(f["yte"], (base_dec > 0).astype(int), average="macro")
                rng = np.random.default_rng(1000 + f["fold"])
                for g in sorted(set(fam)):
                    cols = np.concatenate([np.flatnonzero(fam == g), d + np.flatnonzero(fam == g)])   # mean and dispersion blocks
                    drops = []
                    for _ in range(a.draws):
                        s_perm = f["s_te"].copy(); s_perm[:, cols] = s_perm[rng.permutation(len(s_perm))][:, cols]
                        dec = fit["clf"].decision_function(0.5 * fit["Kg_te"] + 0.5 * np.exp(-fit["ge"] * M.euclidean_squared(s_perm, fit["sa"])))
                        drops.append(base - f1_score(f["yte"], (dec > 0).astype(int), average="macro"))
                    prow.append(dict(fold=f["fold"], family=g, n_columns=int((fam == g).sum()), baseline_f1=base, mean_decrease=float(np.mean(drops)), sd_over_draws=float(np.std(drops, ddof=1)), draws=a.draws))
            pd.DataFrame(prow).to_csv(os.path.join(a.out, "permutation_sensitivity.csv"), index=False)
            print("permutation done", flush=True)
    pd.DataFrame(unified).to_csv(os.path.join(a.out, "unified_comparators.csv"), index=False)
    pd.DataFrame(foldrows).to_csv(os.path.join(a.out, "unified_fold_metrics.csv"), index=False)
    pd.concat(oof).to_csv(os.path.join(a.out, "oof_predictions_fused.csv"), index=False)
    pd.DataFrame(part_rows).to_csv(os.path.join(a.out, "partition_sensitivity.csv"), index=False)
    pd.DataFrame(rank_rows).to_csv(os.path.join(a.out, "rank_sensitivity.csv"), index=False)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
