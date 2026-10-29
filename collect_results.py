"""Collect every metrics.json under WORK_DIR/runs into one table for the paper.

    python collect_results.py      -> WORK_DIR/results_table.csv (+ printed)
"""
import glob
import json
import os

import pandas as pd

import config as C
from rescore import rescore_all

rescore_all()   # make every CPM use the same (current) annotations_excluded.csv

rows = []
for f in sorted(glob.glob(os.path.join(C.RUNS_DIR, "*", "eval_subset*", "metrics.json"))):
    m = json.load(open(f))
    s = m["froc"]["sensitivity_at_fp"]
    rows.append({
        "run": m["run"], "eval": os.path.basename(os.path.dirname(f)), "tta": m.get("tta"), "mc": m.get("mc"),
        "CPM": round(m["froc"]["cpm"], 4),
        "CPM_95CI": "{:.3f}-{:.3f}".format(*m["froc"]["cpm_ci95"]),
        **{f"S@{k}": round(v, 3) for k, v in s.items()},
        "AUC": round(m["auc"], 4) if m.get("auc") else None,
        "AP": round(m["ap"], 4) if m.get("ap") else None,
        "ECE": round(m["ece"], 4) if m.get("ece") is not None else None,
        "ECE_raw": round(m.get("ece_uncalibrated", m.get("ece") or 0), 4) if m.get("ece") is not None else None,
        "diam_MAE_mm": round(m["diameter_mae_mm"], 2) if m.get("diameter_mae_mm") else None,
        "unc_AUC_errors": round(m["uncertainty_auc_for_errors"], 3) if m.get("uncertainty_auc_for_errors") else None,
    })
df = pd.DataFrame(rows)
out = os.path.join(C.WORK_DIR, "results_table.csv")
df.to_csv(out, index=False)
with pd.option_context("display.width", 250, "display.max_columns", 50):
    print(df)
print("->", out)
