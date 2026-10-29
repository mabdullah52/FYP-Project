"""Average the predictions of several runs on the same evaluation (e.g. round 1 + round 2).

    python ensemble.py --runs msgf_f9 msgf_f9_r2 --eval eval_subset9 --name ens_r1r2
Writes WORK_DIR/runs/<name>/<eval>/predictions.csv (+ metrics.json via rescore).
Choose whether to REPORT the ensemble using the validation subset (eval_subset8...), not the test.
"""
import argparse
import json
import os

import pandas as pd

import config as C
from rescore import rescore_all

KEY = ["seriesuid", "coordX", "coordY", "coordZ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval", required=True, help="folder name, e.g. eval_subset9 or eval_subset8_plain")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    dfs = []
    for r in a.runs:
        p = os.path.join(C.RUNS_DIR, r, a.eval, "predictions.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - run evaluate.py for {r} first")
        d = pd.read_csv(p)
        for k in ("coordX", "coordY", "coordZ"):
            d[k] = d[k].round(3)
        dfs.append(d.set_index(KEY))
    base = dfs[0][["class"]].copy()
    for col in ("probability", "uncertainty", "diameter_pred_mm"):
        if all(col in d for d in dfs):
            base[col] = sum(d[col] for d in dfs) / len(dfs)
    out = os.path.join(C.RUNS_DIR, a.name, a.eval)
    os.makedirs(out, exist_ok=True)
    base.reset_index().to_csv(os.path.join(out, "predictions.csv"), index=False)
    json.dump({"run": a.name, "members": a.runs, "ensemble": True}, open(os.path.join(out, "metrics.json"), "w"))
    rescore_all()


if __name__ == "__main__":
    main()
