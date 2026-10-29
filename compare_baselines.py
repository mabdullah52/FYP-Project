"""Step 34 report - MSGF-Net vs standard architectures on the LUNA16 test subset 9 (same split,
training budget and inference: 8-flip TTA + MC dropout). Paired scan-level bootstrap for CPM.

    python compare_baselines.py   -> work/baseline_table.csv
"""
import json
import os

import pandas as pd

import config as C
from model import build_model

RUNS = [("MSGF-Net (ours)", C.FINAL_RUN), ("3D ResNet-18", "base_r3d18"),
        ("2.5D ResNet-18", "base_resnet25d"), ("Plain 3D CNN", "base_cnn3d")]
SUB = f"eval_subset{C.DEFAULT_TEST_SUBSET}"


def main():
    rows, ours = [], os.path.join(C.RUNS_DIR, C.FINAL_RUN, SUB)
    for name, run in RUNS:
        f = os.path.join(C.RUNS_DIR, run, SUB, "metrics.json")
        if not os.path.exists(f):
            print(f"{name}: no test evaluation yet ({f})"); continue
        m = json.load(open(f))
        fr = m.get("froc", m)
        cfg = json.load(open(os.path.join(C.RUNS_DIR, run, "config.json")))
        n_par = sum(p.numel() for p in build_model(cfg).parameters()) / 1e6
        s = fr["sensitivity_at_fp"]
        r = {"model": name, "params_M": round(n_par, 2), "CPM": round(fr["cpm"], 4),
             "CPM_95CI": "{:.3f}-{:.3f}".format(*fr["cpm_ci95"]),
             **{f"sens@{k}": round(v, 3) for k, v in s.items()},
             "AUC": round(m.get("auc", float("nan")), 4), "ECE_raw": round(m.get("ece_uncalibrated", float("nan")), 4),
             "diam_MAE_mm": round(m["diameter_mae_mm"], 2) if m.get("diameter_mae_mm") is not None else None,
             "unc_AUROC_errors": round(m["uncertainty_auc_for_errors"], 3) if m.get("uncertainty_auc_for_errors") else None}
        if run != C.FINAL_RUN and os.path.exists(os.path.join(ours, "predictions.csv")):
            from stats_compare import paired_cpm
            d = paired_cpm(os.path.join(C.RUNS_DIR, run, SUB), ours)
            r["MSGF_minus_this_CPM"] = "{:+.3f} [{:+.3f}, {:+.3f}] p={:.3f}".format(d["delta"], *d["ci95"], d["p"])
        rows.append(r)
    df = pd.DataFrame(rows)
    out = os.path.join(C.WORK_DIR, "baseline_table.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
