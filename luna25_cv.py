"""Step 31 - 5-fold patient-level cross-validation for paper 2 (uses EVERY LUNA25 nodule as test once).

Fold k: test = fold k, validation = fold k+1, training = the other three folds (no patient overlap).
Fold 0 is the original split and reuses the models of steps 13/15 (l25_scratch, l25_temporal_s0,
l25_temporal_noprior_s0); folds 1-4 are trained by 31_CROSS_VALIDATION_OVERNIGHT.bat:
    l25_cv{k}                    single-scan malignancy model (seed 0, from scratch)
    l25_cv{k}_temporal_s0        TG-MSGF (current + prior scan), encoder initialised from l25_cv{k}
    l25_cv{k}_temporal_noprior_s0  same network with the prior replaced by the current scan (control)

    python luna25_cv.py report    -> work/luna25_cv/{cv_metrics.json, cv_table.csv, cv_*_predictions.csv}

Every model is Platt-calibrated on its own validation fold, then the five test folds are pooled
(each nodule is predicted by a model that never saw its patient). AUC 95% CIs and paired tests use a
patient-level bootstrap over the pooled predictions; per-fold AUCs are reported as mean +- SD.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
from luna25_common import L25_DIR
from model import build_model
from predictor import Predictor

OUT = os.path.join(C.WORK_DIR, "luna25_cv")
FOLDS = [0, 1, 2, 3, 4]


def single_run(k):
    return "l25_scratch" if k == 0 else f"l25_cv{k}"


def temporal_run(k, no_prior):
    pre = "l25_temporal" if k == 0 else f"l25_cv{k}_temporal"
    return f"{pre}{'_noprior' if no_prior else ''}_s0"


def roles(fold, k):
    fold = np.asarray(fold)
    return np.where(fold == k, "test", np.where(fold == (k + 1) % 5, "val", "train"))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def boot(groups, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    pats = np.unique(groups)
    w = {g: np.where(groups == g)[0] for g in pats}
    for _ in range(n):
        yield np.concatenate([w[g] for g in rng.choice(pats, len(pats))])


def auc_ci(y, p, g):
    v = [roc_auc_score(y[s], p[s]) for s in boot(g) if 0 < y[s].sum() < len(s)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired(y, pa, pb, g):
    d = np.array([roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]) for s in boot(g) if 0 < y[s].sum() < len(s)])
    return {"delta": float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def score_single(run, rows, arr, tag):
    """Raw malignancy probability of `run` for the given patch rows (cached)."""
    path = os.path.join(OUT, f"scores_{run}_{tag}.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        if set(rows) <= set(d.row):
            return d.set_index("row").prob
    pred = Predictor(os.path.join(C.RUNS_DIR, run, "best.pt"), apply_calibration=False)
    rows = np.asarray(sorted(set(rows)))
    out = []
    for s in range(0, len(rows), 128):
        out.append(pred.score(np.asarray(arr[rows[s:s + 128]]))["prob_raw"])
    d = pd.DataFrame({"row": rows, "prob": np.concatenate(out)})
    d.to_csv(path, index=False)
    print(f"  scored {run} on {len(rows)} nodules ({tag})")
    return d.set_index("row").prob


def score_temporal(run, pairs, device, tag):
    import luna25_temporal as T
    path = os.path.join(OUT, f"scores_{run}_{tag}.npy")
    if os.path.exists(path):
        p = np.load(path)
        if len(p) == len(pairs):
            return p
    ck = torch.load(os.path.join(C.RUNS_DIR, run, "best.pt"), map_location=device, weights_only=False)
    m = T.TGMSGF(build_model(ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    p, _ = T.predict(m, pairs.reset_index(drop=True), device, ck.get("no_prior", False))
    np.save(path, p)
    print(f"  scored {run} on {len(pairs)} pairs ({tag})")
    return p


def platt(pv, yv, pt):
    lr = LogisticRegression(C=1e4).fit(logit(pv)[:, None], yv)
    return lr.predict_proba(logit(pt)[:, None])[:, 1]


def report():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(os.path.join(L25_DIR, "meta.csv"), dtype={"seriesuid": str, "patient": str})
    done = set(open(os.path.join(L25_DIR, "done_scans.txt")).read().split())
    meta = meta[meta.seriesuid.isin(done)].reset_index(drop=True)
    ext = pd.read_csv(os.path.join(L25_DIR, "external", "predictions.csv"))[["row", "diameter_pred_mm"]]
    meta = meta.merge(ext, on="row", how="left")
    arr = np.load(os.path.join(L25_DIR, "patches.npy"), mmap_mode="r")
    pairs = pd.read_csv(os.path.join(C.WORK_DIR, "luna25_temporal", "pairs.csv"), dtype={"patient": str})
    pfold = meta.drop_duplicates("patient").set_index("patient").fold
    pairs["fold"] = pairs.patient.astype(str).map(pfold)

    nod_rows, pair_rows, per_fold, used = [], [], [], []
    for k in FOLDS:
        if not os.path.exists(os.path.join(C.RUNS_DIR, single_run(k), "best.pt")):
            print(f"fold {k}: single-scan model {single_run(k)} not trained yet - fold skipped")
            continue
        used.append(k)
        r = roles(meta.fold, k)
        va, te = meta[r == "val"].reset_index(drop=True), meta[r == "test"].reset_index(drop=True)
        sc = pd.concat([score_single(single_run(k), va.row, arr, f"fold{(k + 1) % 5}"),
                        score_single(single_run(k), te.row, arr, f"fold{k}")])
        sc = sc[~sc.index.duplicated()]
        yv = va.label.to_numpy()
        # ---- nodule level: 3D model vs size only (all nodules of the test fold)
        pv, pt = sc.reindex(va.row).to_numpy(), sc.reindex(te.row).to_numpy()
        size_lr = LogisticRegression(C=1e4).fit(va.diameter_pred_mm.to_numpy()[:, None], yv)
        d = te[["row", "patient", "label", "diameter_pred_mm"]].copy()
        d["fold"] = k
        d["model_3d"] = platt(pv, yv, pt)
        d["size_only"] = size_lr.predict_proba(te.diameter_pred_mm.to_numpy()[:, None])[:, 1]
        nod_rows.append(d)
        fr = {"fold": k, "test_nodules": int(len(te)), "test_malignant": int(d.label.sum()),
              "auc_3d": float(roc_auc_score(d.label, d.model_3d)), "auc_size": float(roc_auc_score(d.label, d.size_only))}
        # ---- pairs (nodules with a prior scan)
        pr = roles(pairs.fold, k)
        pva, pte = pairs[pr == "val"].reset_index(drop=True), pairs[pr == "test"].reset_index(drop=True)
        for p_ in (pva, pte):
            p_["single_cur"] = sc.reindex(p_.row_cur).to_numpy()
            p_["single_prior"] = sc.reindex(p_.row_prior).to_numpy()
        ypv = pva.label.to_numpy()
        q = pte[["patient", "label", "row_cur", "dt_years", "diam_cur", "diam_prior"]].copy()
        q["fold"] = k
        q["single"] = platt(pva.single_cur, ypv, pte.single_cur)
        feats = lambda x: np.column_stack([logit(x.single_cur), x.diam_cur, (x.diam_cur - x.diam_prior) / x.dt_years.clip(lower=0.25),
                                           logit(x.single_cur) - logit(x.single_prior)])
        Xv, Xt = feats(pva), feats(pte)
        mu, sd = np.nanmean(Xv, 0), np.nanstd(Xv, 0) + 1e-6
        g_lr = LogisticRegression(C=1.0).fit(np.nan_to_num((Xv - mu) / sd), ypv)
        q["growth"] = g_lr.predict_proba(np.nan_to_num((Xt - mu) / sd))[:, 1]
        tg = {}
        for no_prior, col in ((False, "tg_msgf"), (True, "tg_noprior")):
            run = temporal_run(k, no_prior)
            if os.path.exists(os.path.join(C.RUNS_DIR, run, "best.pt")):
                tv = score_temporal(run, pva, device, f"fold{(k + 1) % 5}")
                tt = score_temporal(run, pte, device, f"fold{k}")
                tg[col] = (tv, tt)
                q[col] = platt(tv, ypv, tt)
            else:
                q[col] = np.nan
                print(f"fold {k}: {run} not trained yet")
        if "tg_msgf" in tg:
            fs = lambda x, t: np.column_stack([logit(x.single_cur), logit(t), (x.diam_cur - x.diam_prior) / x.dt_years.clip(lower=0.25),
                                               logit(x.single_cur) - logit(x.single_prior)])
            Xv2, Xt2 = fs(pva, tg["tg_msgf"][0]), fs(pte, tg["tg_msgf"][1])
            mu2, sd2 = np.nanmean(Xv2, 0), np.nanstd(Xv2, 0) + 1e-6
            s_lr = LogisticRegression(C=1.0).fit(np.nan_to_num((Xv2 - mu2) / sd2), ypv)
            q["stacked"] = s_lr.predict_proba(np.nan_to_num((Xt2 - mu2) / sd2))[:, 1]
        else:
            q["stacked"] = np.nan
        pair_rows.append(q)
        yq = q.label.to_numpy()
        fr.update({"test_pairs": int(len(q)), "test_pairs_malignant": int(yq.sum())})
        if 0 < yq.sum() < len(yq):
            for col in ("single", "growth", "tg_msgf", "tg_noprior", "stacked"):
                if q[col].notna().all():
                    fr[f"pairs_auc_{col}"] = float(roc_auc_score(yq, q[col]))
        per_fold.append(fr)
        print(f"fold {k}: {fr}")

    if not per_fold:
        raise SystemExit("No fold has a trained model yet - run 31_CROSS_VALIDATION_OVERNIGHT.bat first.")
    N = pd.concat(nod_rows, ignore_index=True)
    Pp = pd.concat(pair_rows, ignore_index=True)
    N.to_csv(os.path.join(OUT, "cv_nodule_predictions.csv"), index=False)
    Pp.to_csv(os.path.join(OUT, "cv_pair_predictions.csv"), index=False)
    res = {"folds_used": used, "per_fold": per_fold, "nodule_level": {}, "pair_level": {}, "paired_tests": {}}
    rows = []

    def summarise(df, cols, level, names):
        y, g = df.label.to_numpy(), df.patient.astype(str).to_numpy()
        for c in cols:
            if c not in df or df[c].isna().any():
                continue
            p = df[c].to_numpy()
            fold_aucs = [roc_auc_score(d.label, d[c]) for _, d in df.groupby("fold") if 0 < d.label.sum() < len(d)]
            m = {"pooled_auc": float(roc_auc_score(y, p)), "pooled_auc_ci95": auc_ci(y, p, g),
                 "ap": float(average_precision_score(y, p)),
                 "fold_auc_mean": float(np.mean(fold_aucs)), "fold_auc_sd": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0}
            res[level][c] = m
            rows.append({"level": level, "model": names[c], "n": int(len(df)), "malignant": int(y.sum()),
                         "pooled_AUC": round(m["pooled_auc"], 4), "AUC_95CI": "{:.3f}-{:.3f}".format(*m["pooled_auc_ci95"]),
                         "fold_AUC_mean_sd": f"{m['fold_auc_mean']:.3f} +- {m['fold_auc_sd']:.3f}", "AP": round(m["ap"], 4)})

    summarise(N, ["model_3d", "size_only"], "nodule_level", {"model_3d": "3D model (single seed)", "size_only": "size only"})
    names = {"single": "single scan", "growth": "growth formula", "tg_msgf": "TG-MSGF", "tg_noprior": "TG-MSGF no-prior control",
             "stacked": "stacked: single + TG-MSGF + growth"}
    summarise(Pp, list(names), "pair_level", names)
    yN, gN = N.label.to_numpy(), N.patient.astype(str).to_numpy()
    res["paired_tests"]["3D model vs size only (all nodules)"] = paired(yN, N.size_only.to_numpy(), N.model_3d.to_numpy(), gN)
    yP, gP = Pp.label.to_numpy(), Pp.patient.astype(str).to_numpy()
    full = lambda c: c in Pp and Pp[c].notna().all()
    for a, b in (("single", "growth"), ("single", "tg_msgf"), ("single", "stacked"), ("tg_noprior", "tg_msgf"), ("growth", "stacked")):
        if full(a) and full(b):
            res["paired_tests"][f"{names[b]} vs {names[a]}"] = paired(yP, Pp[a].to_numpy(), Pp[b].to_numpy(), gP)
    json.dump(res, open(os.path.join(OUT, "cv_metrics.json"), "w"), indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cv_table.csv"), index=False)
    print(f"\nfolds used: {used}  |  nodules {len(N)} ({int(yN.sum())} malignant)  |  pairs {len(Pp)} ({int(yP.sum())} malignant)\n")
    print(df.to_string(index=False))
    print()
    for k, v in res["paired_tests"].items():
        print(f"{k:55s} delta {v['delta']:+.4f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p={v['p']:.3f}")
    print("->", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        print(__doc__)
