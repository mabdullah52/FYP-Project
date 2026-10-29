"""LUNA25 step 2 (part A) - external validation of the LUNA16 nodule model on NLST/LUNA25.

LUNA25 provides nodule locations but no non-nodule candidates, so the external test measures
NODULE-LEVEL SENSITIVITY at fixed operating points: the probability thresholds that gave
1/8 ... 8 false positives per scan on the LUNA16 VALIDATION subset (chosen before seeing LUNA25).
The same thresholds are applied to the LUNA16 test subset for a direct internal-vs-external comparison.

    python luna25_external.py           -> WORK_DIR/luna25/external/{metrics.json, predictions.csv}
"""
import json
import os

import numpy as np
import pandas as pd

import config as C
from luna25_common import L25_DIR
from luna_io import index_scans
from metrics import per_scan_hits
from predictor import Predictor

RATES = C.FROC_FP_RATES


def thresholds_from_validation(run):
    """Score thresholds at which the LUNA16 validation subset has r FPs per scan."""
    v = pd.read_csv(os.path.join(C.RUNS_DIR, run, f"eval_subset{C.DEFAULT_VAL_SUBSET}", "predictions.csv"))
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if os.path.exists(C.EXCLUDED_CSV) else None
    uids = sorted(u for u, (_, s) in index_scans().items() if s == C.DEFAULT_VAL_SUBSET)
    fps = np.sort(np.concatenate([h[1] for h in per_scan_hits(v, ann, excl, uids)]))[::-1]
    return {r: float(fps[min(len(fps) - 1, max(0, int(round(r * len(uids))) - 1))]) for r in RATES}


def internal_sensitivity(run, thr):
    t = pd.read_csv(os.path.join(C.RUNS_DIR, run, f"eval_subset{C.DEFAULT_TEST_SUBSET}", "predictions.csv"))
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if os.path.exists(C.EXCLUDED_CSV) else None
    uids = sorted(u for u, (_, s) in index_scans().items() if s == C.DEFAULT_TEST_SUBSET)
    hits = per_scan_hits(t, ann, excl, uids)
    nod = np.concatenate([h[0] for h in hits])
    fp = np.concatenate([h[1] for h in hits])
    return {str(r): {"sensitivity": float((nod >= v).mean()), "fp_per_scan": float((fp >= v).sum() / len(uids))}
            for r, v in thr.items()}


def boot_ci(hit, groups, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    g = pd.Series(hit).groupby(groups)
    s, c = g.sum().to_numpy(), g.count().to_numpy()
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(s), len(s))
        vals.append(s[i].sum() / c[i].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    run = C.FINAL_RUN
    out_dir = os.path.join(L25_DIR, "external")
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(L25_DIR, "meta.csv"), dtype={"seriesuid": str, "patient": str})
    arr = np.load(os.path.join(L25_DIR, "patches.npy"), mmap_mode="r")
    pred = Predictor(os.path.join(C.RUNS_DIR, run, "best.pt"))       # TTA + MC dropout + Platt (as deployed)
    print(f"Scoring {len(meta)} LUNA25 nodules with the LUNA16 model '{run}' ...")
    probs, unc, diam = [], [], []
    for s in range(0, len(meta), 256):
        r = pred.score(np.asarray(arr[meta.row.to_numpy()[s:s + 256]]))
        probs.append(r["prob"]); unc.append(r["uncertainty"]); diam.append(r["diameter_mm"])
        if (s // 256) % 5 == 0:
            print(f"  {min(s + 256, len(meta))}/{len(meta)}")
    meta["nodule_prob"], meta["uncertainty"], meta["diameter_pred_mm"] = map(np.concatenate, (probs, unc, diam))
    meta.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    thr = thresholds_from_validation(run)
    internal = internal_sensitivity(run, thr)
    res = {"model": run, "n_nodules": int(len(meta)), "n_patients": int(meta.patient.nunique()),
           "thresholds_from_luna16_validation": {str(k): v for k, v in thr.items()}, "operating_points": {}}
    size_bins = pd.cut(meta.diameter_pred_mm, [0, 6, 8, 15, 100], right=False, labels=["<6", "6-8", "8-15", ">=15"])
    for r, v in thr.items():
        hit = (meta.nodule_prob >= v).to_numpy()
        res["operating_points"][str(r)] = {
            "luna25_sensitivity": float(hit.mean()), "luna25_ci95": boot_ci(hit, meta.patient.to_numpy()),
            "luna25_sens_malignant": float(hit[meta.label == 1].mean()),
            "luna25_sens_benign": float(hit[meta.label == 0].mean()),
            "luna25_sens_by_size": {str(b): float(hit[(size_bins == b).to_numpy()].mean())
                                    for b in size_bins.cat.categories if (size_bins == b).any()},
            "luna16_test_sensitivity": internal[str(r)]["sensitivity"],
            "luna16_test_fp_per_scan": internal[str(r)]["fp_per_scan"],
        }
    res["mean_luna25_sensitivity_over_7_points"] = float(np.mean([v["luna25_sensitivity"] for v in res["operating_points"].values()]))
    res["mean_luna16_test_sensitivity_over_7_points"] = float(np.mean([v["luna16_test_sensitivity"] for v in res["operating_points"].values()]))
    res["predicted_diameter_mm_median_iqr"] = [float(np.percentile(meta.diameter_pred_mm, q)) for q in (50, 25, 75)]
    json.dump(res, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(f"\n{'FP/scan (LUNA16 val)':>22} {'LUNA16 test sens':>17} {'LUNA25 sens (95% CI)':>26} {'malignant':>10} {'benign':>8}")
    for r, o in res["operating_points"].items():
        print(f"{r:>22} {o['luna16_test_sensitivity']:17.3f} {o['luna25_sensitivity']:10.3f} "
              f"({o['luna25_ci95'][0]:.3f}-{o['luna25_ci95'][1]:.3f}) {o['luna25_sens_malignant']:10.3f} {o['luna25_sens_benign']:8.3f}")
    print(f"\nmean sensitivity over the 7 points: LUNA16 test {res['mean_luna16_test_sensitivity_over_7_points']:.3f}"
          f" | LUNA25 external {res['mean_luna25_sensitivity_over_7_points']:.3f}")
    print("->", out_dir)


if __name__ == "__main__":
    main()
