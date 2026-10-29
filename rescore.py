"""Recompute CPM / FROC for every saved evaluation from its predictions.csv, using the
CURRENT annotations_excluded.csv (no GPU, a few seconds per run). Run it after adding
annotations_excluded.csv, or any time - collect_results.py calls it automatically.

    python rescore.py
"""
import glob
import json
import os

import pandas as pd

import config as C
from luna_io import index_scans
from metrics import cpm_with_ci


def rescore_all(n_boot=1000, verbose=True):
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if os.path.exists(C.EXCLUDED_CSV) else None
    scan_index = index_scans()
    for pred_path in sorted(glob.glob(os.path.join(C.RUNS_DIR, "*", "eval_subset*", "predictions.csv"))):
        d = os.path.dirname(pred_path)
        mpath = os.path.join(d, "metrics.json")
        m = json.load(open(mpath)) if os.path.exists(mpath) else {}
        if m.get("froc", {}).get("excluded_used") == (excl is not None) and m.get("rescored"):
            continue
        df = pd.read_csv(pred_path)
        subset = int(os.path.basename(d)[len("eval_subset"):].split("_")[0])
        uids = [u for u, (_, s) in scan_index.items() if s == subset]
        froc, _ = cpm_with_ci(df, ann[ann.seriesuid.isin(uids)], excl, uids, n_boot=n_boot)
        froc["excluded_used"] = excl is not None
        m.update({"froc": froc, "rescored": True})
        m.setdefault("run", os.path.basename(os.path.dirname(d)))
        m.setdefault("subset", subset)
        json.dump(m, open(mpath, "w"), indent=2)
        if verbose:
            print(f"{os.path.relpath(d, C.RUNS_DIR):45s} CPM {froc['cpm']:.4f}  "
                  f"[{froc['cpm_ci95'][0]:.3f}-{froc['cpm_ci95'][1]:.3f}]  excluded file used: {excl is not None}")


if __name__ == "__main__":
    rescore_all()
