"""Step 2b (optional, recommended) - model-driven hard-negative mining for round 2.

Scores EVERY candidate of the training subsets with the round-1 model and keeps, per
scan, the TOP_K negatives the model is most wrongly confident about. These are
stored as a second patch file and mixed into round-2 fine-tuning (train.py --hard-meta).
Validation and test subsets are never touched.

    python mine_hard_negatives.py --run msgf_f9 --top-k 20
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as C
from evaluate import score_subsets
from luna_io import index_scans, iter_scan_patches
from predictor import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2), help="extraction threads")
    ap.add_argument("--max-scans", type=int, default=None)
    args = ap.parse_args()

    pred = Predictor(os.path.join(C.RUNS_DIR, args.run, "best.pt"), tta=False, mc=1)
    held_out = {pred.cfg["test_subset"], pred.cfg["val_subset"]}
    train_subsets = [s for s in range(10) if s not in held_out]
    scan_index = index_scans()
    cands = pd.read_csv(C.CANDIDATES_CSV)
    print(f"Scoring all candidates of training subsets {train_subsets} ...")
    df, _ = score_subsets(pred, scan_index, cands, train_subsets, args.workers, args.max_scans, log_every=50)

    already = pd.read_csv(os.path.join(C.PATCH_DIR, "meta.csv"))
    key = lambda d: d.seriesuid + "_" + d.coordX.round(2).astype(str) + "_" + d.coordZ.round(2).astype(str)
    neg = df[(df["class"] == 0) & ~key(df).isin(set(key(already)))]
    hard = neg.sort_values("probability", ascending=False).groupby("seriesuid").head(args.top_k)
    hard = hard[hard.probability > 0.05].reset_index(drop=True)     # only genuinely confusing ones
    print(f"hard negatives kept: {len(hard)}  (median model prob {hard.probability.median():.3f})")

    fname = f"patches_hard_{args.run}.npy"
    s = C.STORE_SIZE
    arr = np.lib.format.open_memmap(os.path.join(C.PATCH_DIR, fname), mode="w+", dtype=np.int16,
                                    shape=(len(hard), s, s, s))
    hard["row"] = np.arange(len(hard))
    groups = {u: g for u, g in hard.groupby("seriesuid")}
    jobs = [(u, scan_index[u][0], g[["coordX", "coordY", "coordZ"]].to_numpy()) for u, g in groups.items()]
    for uid, patches, err in iter_scan_patches(jobs, args.workers):
        if err is not None:
            raise RuntimeError(f"could not read scan {uid}: {err}")
        arr[groups[uid].row.to_numpy()] = patches
    arr.flush()
    out = hard.rename(columns={"class": "label"})
    out["subset"] = out.seriesuid.map(lambda u: scan_index[u][1])
    out["diameter_mm"] = np.nan
    out["kind"] = "hard_neg"
    out["file"] = fname
    path = os.path.join(C.PATCH_DIR, f"hard_{args.run}.csv")
    out[["seriesuid", "coordX", "coordY", "coordZ", "label", "subset", "diameter_mm", "kind", "row", "file"]].to_csv(path, index=False)
    print("->", path)


if __name__ == "__main__":
    main()
