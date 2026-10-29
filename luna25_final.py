"""LUNA25 step 4 - final, robust malignancy results + the deployable cancer-risk model.

    python luna25_final.py                 (after 3 scratch seeds + transfer are trained)

1. scores the VALIDATION and TEST folds with every trained malignancy model (TTA + MC dropout)
2. seed robustness: mean +- SD of test AUC over the scratch seeds
3. seed ENSEMBLE (mean probability) with patient-bootstrap CI
4. FUSION: logistic regression on [ensemble logit, predicted diameter, age] fitted on VALIDATION only
5. paired patient-bootstrap tests: ensemble vs size-only, fusion vs ensemble, fusion vs size-only,
   transfer vs scratch ensemble
6. uncertainty, measured properly for an imbalanced task: errors at the validation-chosen
   threshold, and AUC after referring the most uncertain 10/20% of nodules to the radiologist
7. checks whether the same nodule is followed over several screening rounds (paper-2 idea)
8. writes work/runs/l25_risk/risk_model.json, used by the doctor app for a per-nodule cancer risk
Outputs: work/luna25_final/{final_metrics.json, final_table.csv, val_predictions.csv, test_predictions.csv}
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import config as C
from luna25_common import L25_DIR, find_csv, load_table
from metrics import expected_calibration_error
from predictor import Predictor

OUT = os.path.join(C.WORK_DIR, "luna25_final")
RISK_DIR = os.path.join(C.RUNS_DIR, "l25_risk")
SCRATCH = ["l25_scratch", "l25_scratch_s1", "l25_scratch_s2"]
TRANSFER = ["l25_transfer"]
N_BOOT = 2000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def boot_idx(groups, n=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    pats = np.unique(groups)
    where = {g: np.where(groups == g)[0] for g in pats}
    for _ in range(n):
        yield np.concatenate([where[g] for g in rng.choice(pats, len(pats))])


def auc_ci(y, p, g):
    v = [roc_auc_score(y[s], p[s]) for s in boot_idx(g) if 0 < y[s].sum() < len(s)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired(y, pa, pb, g):
    """AUC(b) - AUC(a) with patient-bootstrap CI and two-sided p."""
    d = np.array([roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s])
                  for s in boot_idx(g) if 0 < y[s].sum() < len(s)])
    return {"delta": float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def score_split(run, split, meta, arr):
    path = os.path.join(OUT, f"scores_{run}_{split}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    pred = Predictor(os.path.join(C.RUNS_DIR, run, "best.pt"), apply_calibration=False)
    res = {"prob": [], "unc": []}
    for s in range(0, len(meta), 128):
        r = pred.score(np.asarray(arr[meta.row.to_numpy()[s:s + 128]]))
        res["prob"].append(r["prob_raw"]); res["unc"].append(r["uncertainty"])
    out = pd.DataFrame({"row": meta.row.to_numpy(), "prob": np.concatenate(res["prob"]),
                        "unc": np.concatenate(res["unc"])})
    out.to_csv(path, index=False)
    print(f"  scored {run} on {split}")
    return out


def operating_point(yv, pv, yt, pt):
    fpr, tpr, th = roc_curve(yv, pv)
    thr = float(th[np.argmax(tpr - fpr)])
    tp = ((pt >= thr) & (yt == 1)).sum(); fn = ((pt < thr) & (yt == 1)).sum()
    tn = ((pt < thr) & (yt == 0)).sum(); fp = ((pt >= thr) & (yt == 0)).sum()
    return thr, float(tp / max(1, tp + fn)), float(tn / max(1, tn + fp))


def uncertainty_block(y, p, unc, thr):
    err = ((p >= thr) != (y == 1)).astype(int)
    out = {"auroc_errors_at_threshold": float(roc_auc_score(err, unc)) if 0 < err.sum() < len(err) else None}
    order = np.argsort(-unc)
    for f in (0.1, 0.2):
        keep = np.ones(len(p), bool); keep[order[: int(round(f * len(p)))]] = False
        rnd = np.random.RandomState(0).permutation(len(p))
        keep_r = np.ones(len(p), bool); keep_r[rnd[: int(round(f * len(p)))]] = False
        out[f"auc_after_referring_{int(f * 100)}pct_most_uncertain"] = float(roc_auc_score(y[keep], p[keep]))
        out[f"auc_after_referring_{int(f * 100)}pct_random"] = float(roc_auc_score(y[keep_r], p[keep_r]))
    return out


def temporal_check():
    """Does the CSV follow the same nodule across screening rounds (T0/T1/T2)?"""
    csv = find_csv(C.find_luna25(), ask=False)
    if not csv:
        return {"checked": False}
    raw = pd.read_csv(csv)
    cols = list(raw.columns)
    norm = {c: "".join(ch for ch in c.lower() if ch.isalnum()) for c in cols}
    pcol = next((c for c in cols if norm[c] in ("patientid", "patient")), None)
    dcol = next((c for c in cols if norm[c] in ("studydate", "date")), None)
    scol = next((c for c in cols if norm[c] in ("seriesinstanceuid", "seriesuid")), None)
    res = {"checked": True, "columns": cols}
    if pcol and scol:
        k = raw.groupby(pcol)[scol].nunique()
        res["patients_with_2plus_scans"] = int((k >= 2).sum())
        res["patients_with_3plus_scans"] = int((k >= 3).sum())
    if pcol and dcol:
        res["distinct_study_dates"] = sorted(map(str, raw[dcol].dropna().unique()))[:10]
        links = {}
        for c in cols:
            if c in (pcol, dcol, scol) or "id" not in norm[c]:
                continue
            g = raw.groupby([pcol, c])[dcol].nunique()
            links[c] = {"ids": int(len(g)), "followed_over_2plus_dates": int((g >= 2).sum())}
        res["id_columns_linking_rounds"] = links
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(os.path.join(L25_DIR, "meta.csv"), dtype={"seriesuid": str, "patient": str})
    arr = np.load(os.path.join(L25_DIR, "patches.npy"), mmap_mode="r")
    ext = pd.read_csv(os.path.join(L25_DIR, "external", "predictions.csv"), dtype={"seriesuid": str, "patient": str})
    meta = meta.merge(ext[["row", "diameter_pred_mm"]], on="row", how="left")
    va, te = meta[meta.role == "val"].reset_index(drop=True), meta[meta.role == "test"].reset_index(drop=True)
    yv, yt, gt = va.label.to_numpy(), te.label.to_numpy(), te.patient.to_numpy()
    print(f"val {len(va)} ({yv.sum()} malignant) | test {len(te)} ({yt.sum()} malignant, {len(np.unique(gt))} patients)")

    runs = [r for r in SCRATCH + TRANSFER if os.path.exists(os.path.join(C.RUNS_DIR, r, "best.pt"))]
    print("models:", runs)
    P = {}
    for r in runs:
        P[r] = {"val": score_split(r, "val", va, arr), "test": score_split(r, "test", te, arr)}

    res = {"n_val": int(len(va)), "n_test": int(len(te)), "n_test_malignant": int(yt.sum()),
           "n_test_patients": int(len(np.unique(gt))), "models": {}}
    rows = []

    def add(name, pv, pt, unc=None, note=""):
        thr, sens, spec = operating_point(yv, pv, yt, pt)
        m = {"val_auc": float(roc_auc_score(yv, pv)), "test_auc": float(roc_auc_score(yt, pt)),
             "test_auc_ci95": auc_ci(yt, pt, gt), "test_ap": float(average_precision_score(yt, pt)),
             "threshold_from_val": thr, "test_sensitivity": sens, "test_specificity": spec,
             "test_ece": expected_calibration_error(pt, yt)[0]}
        if unc is not None:
            m["uncertainty"] = uncertainty_block(yt, pt, unc, thr)
        res["models"][name] = m
        rows.append({"model": name, "val_AUC": round(m["val_auc"], 4), "test_AUC": round(m["test_auc"], 4),
                     "AUC_95CI": "{:.3f}-{:.3f}".format(*m["test_auc_ci95"]), "AP": round(m["test_ap"], 4),
                     "sens": round(sens, 3), "spec": round(spec, 3), "ECE": round(m["test_ece"], 4), "note": note})
        return m

    def calibrated(raw_v, raw_t):            # Platt on validation logits
        lr = LogisticRegression(C=1e4).fit(logit(raw_v)[:, None], yv)
        return lr.predict_proba(logit(raw_v)[:, None])[:, 1], lr.predict_proba(logit(raw_t)[:, None])[:, 1], lr

    # individual models
    for r in runs:
        pv, pt, _ = calibrated(P[r]["val"].prob.to_numpy(), P[r]["test"].prob.to_numpy())
        add(r, pv, pt, P[r]["test"].unc.to_numpy(), "transfer" if r in TRANSFER else "scratch seed")
    seeds = [r for r in SCRATCH if r in P]
    aucs = [res["models"][r]["test_auc"] for r in seeds]
    res["scratch_seed_test_auc_mean_sd"] = [float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0]

    # size-only baseline (predicted diameter from the LUNA16 model) - calibrated on validation
    dv, dt = va.diameter_pred_mm.to_numpy(), te.diameter_pred_mm.to_numpy()
    lr_s = LogisticRegression(C=1e4).fit(dv[:, None], yv)
    size_v, size_t = lr_s.predict_proba(dv[:, None])[:, 1], lr_s.predict_proba(dt[:, None])[:, 1]
    add("size_only", size_v, size_t, None, "predicted diameter only")

    # seed ensemble
    ens_raw_v = np.mean([P[r]["val"].prob for r in seeds], 0)
    ens_raw_t = np.mean([P[r]["test"].prob for r in seeds], 0)
    ens_unc_t = np.mean([P[r]["test"].unc for r in seeds], 0) + np.std([P[r]["test"].prob for r in seeds], 0)
    ens_v, ens_t, lr_e = calibrated(ens_raw_v, ens_raw_t)
    add("ensemble", ens_v, ens_t, ens_unc_t, f"mean of {len(seeds)} scratch seeds")

    # fusion with size and age (fitted on validation only)
    has_age = "age" in va and va.age.notna().all() and te.age.notna().all()
    feats = lambda d, raw: np.column_stack([logit(raw), d.diameter_pred_mm.to_numpy()] +
                                           ([d.age.to_numpy(float)] if has_age else []))
    Xv, Xt = feats(va, ens_raw_v), feats(te, ens_raw_t)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    lr_f = LogisticRegression(C=1.0).fit((Xv - mu) / sd, yv)
    fus_v, fus_t = lr_f.predict_proba((Xv - mu) / sd)[:, 1], lr_f.predict_proba((Xt - mu) / sd)[:, 1]
    add("fusion", fus_v, fus_t, ens_unc_t, "ensemble + size" + (" + age" if has_age else ""))
    # app variant without age (age may be missing in the doctor app)
    lr_f2 = LogisticRegression(C=1.0).fit(((Xv - mu) / sd)[:, :2], yv)

    # paired tests
    res["paired_tests"] = {
        "ensemble_vs_size_only": paired(yt, size_t, ens_t, gt),
        "fusion_vs_ensemble": paired(yt, ens_t, fus_t, gt),
        "fusion_vs_size_only": paired(yt, size_t, fus_t, gt),
    }
    if "l25_transfer" in P:
        pt_tr = calibrated(P["l25_transfer"]["val"].prob.to_numpy(), P["l25_transfer"]["test"].prob.to_numpy())[1]
        res["paired_tests"]["transfer_vs_ensemble"] = paired(yt, ens_t, pt_tr, gt)
    # by nodule size (does the model help most where size is ambiguous?)
    bins = pd.cut(te.diameter_pred_mm, [0, 6, 8, 15, 100], right=False, labels=["<6", "6-8", "8-15", ">=15"])
    res["auc_by_size"] = {}
    for b in bins.cat.categories:
        m = (bins == b).to_numpy()
        if 0 < yt[m].sum() < m.sum():
            res["auc_by_size"][str(b)] = {"n": int(m.sum()), "malignant": int(yt[m].sum()),
                                          "ensemble_auc": float(roc_auc_score(yt[m], ens_t[m])),
                                          "size_only_auc": float(roc_auc_score(yt[m], dt[m]))}
    res["temporal_check"] = temporal_check()

    # deployable risk model for the doctor app
    os.makedirs(RISK_DIR, exist_ok=True)
    risk = {"members": [os.path.join(r, "best.pt") for r in seeds],
            "feature_mean": mu.tolist(), "feature_sd": sd.tolist(), "uses_age": bool(has_age),
            "with_age": {"coef": lr_f.coef_[0].tolist(), "intercept": float(lr_f.intercept_[0])} if has_age else None,
            "without_age": {"coef": lr_f2.coef_[0].tolist(), "intercept": float(lr_f2.intercept_[0])},
            "test_auc": res["models"]["fusion"]["test_auc"], "trained_on": "LUNA25 (NLST) folds 2-4, fitted on fold 1",
            "note": "Research prototype. Malignancy risk for screening-detected nodules; not a diagnosis."}
    json.dump(risk, open(os.path.join(RISK_DIR, "risk_model.json"), "w"), indent=2)

    va.assign(ensemble=ens_v, fusion=fus_v).to_csv(os.path.join(OUT, "val_predictions.csv"), index=False)
    te.assign(ensemble=ens_t, fusion=fus_t, size_only=size_t, uncertainty=ens_unc_t).to_csv(
        os.path.join(OUT, "test_predictions.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "final_metrics.json"), "w"), indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "final_table.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n", df.to_string(index=False))
    print("\nscratch seeds test AUC: {:.4f} +- {:.4f}".format(*res["scratch_seed_test_auc_mean_sd"]))
    for k, v in res["paired_tests"].items():
        print(f"{k:28s} delta {v['delta']:+.4f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p={v['p']:.3f}")
    print("\nAUC by predicted size:", json.dumps(res["auc_by_size"], indent=1))
    print("\nuncertainty (ensemble):", json.dumps(res["models"]["ensemble"].get("uncertainty"), indent=1))
    print("\ntemporal check:", json.dumps(res["temporal_check"], indent=1))
    print("\n->", OUT, "\n-> risk model for the doctor app:", os.path.join(RISK_DIR, "risk_model.json"))


if __name__ == "__main__":
    main()
