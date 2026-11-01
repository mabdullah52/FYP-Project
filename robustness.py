"""Step 35 - robustness of MSGF-Net to NOISY and INCOMPLETE CT data (LUNA16 test subset 9).

Every one of the 88 held-out test scans is read once and corrupted in several ways before the
usual candidate patches are cut and scored with the final model (work/runs/msgf_f9):

    clean        the original scan (reference)
    noise25/50/100   Gaussian noise of 25 / 50 / 100 HU added to every voxel
                 (simulates lower radiation dose; typical low-dose CT noise is about 20-60 HU)
    slices2/3    only every 2nd / 3rd slice kept (50% / 67% of slices missing,
                 i.e. 2x / 3x thicker slices, as in older or incomplete scans)
    noise50+slices2   both at once

Scoring uses one pass with 10 MC-dropout samples (no flip TTA) so it runs on a laptop CPU.
Results -> work/robustness/ : robustness_table.csv, robustness.png, per-condition predictions.
Resumable: finished scans are skipped when the script is started again.

    python robustness.py                 # all conditions
    python robustness.py --max-scans 3   # quick test
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

import config as C
from luna_io import Scan, index_scans
from metrics import cpm_with_ci
from predictor import Predictor

CONDITIONS = {                      # name: (noise sigma in HU, keep every k-th slice)
    "clean": (0, 1),
    "noise25": (25, 1),
    "noise50": (50, 1),
    "noise100": (100, 1),
    "slices2": (0, 2),
    "slices3": (0, 3),
    "noise50+slices2": (50, 2),
}
LABELS = {"clean": "Original scans", "noise25": "Noise 25 HU", "noise50": "Noise 50 HU", "noise100": "Noise 100 HU",
          "slices2": "50% of slices missing (2x thicker)", "slices3": "67% of slices missing (3x thicker)",
          "noise50+slices2": "Noise 50 HU + 50% slices missing"}
OUT = os.path.join(C.WORK_DIR, "robustness")


def corrupt(scan, sigma, keep_every, seed):
    """Return a Scan-like object whose array/geometry carry the corruption."""
    import copy
    s = copy.copy(scan)
    arr = np.asarray(scan.array)
    if keep_every > 1:
        arr = arr[::keep_every]
        s.spacing = scan.spacing.copy(); s.spacing[2] *= keep_every
    if sigma > 0:
        rng = np.random.default_rng(seed)
        noisy = arr.astype(np.float32) + rng.normal(0, sigma, arr.shape).astype(np.float32)
        arr = noisy.clip(*C.HU_CLIP).astype(np.int16)
    elif keep_every > 1:
        arr = np.ascontiguousarray(arr)
    s.array = arr
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="msgf_f9")
    ap.add_argument("--subset", type=int, default=9)
    ap.add_argument("--mc", type=int, default=10)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()
    conds = [c for c in args.conditions.split(",") if c in CONDITIONS]
    os.makedirs(OUT, exist_ok=True)

    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    pred = Predictor(os.path.join(C.RUNS_DIR, args.run, "best.pt"), tta=False, mc=args.mc)
    print(f"device: {pred.device}   conditions: {conds}")

    scan_index = index_scans()
    cands = pd.read_csv(C.CANDIDATES_CSV)
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if os.path.exists(C.EXCLUDED_CSV) else None
    if excl is None:
        print("NOTE: annotations_excluded.csv not found -> CPM slightly lower than the official protocol")
    uids = sorted(u for u, (_, s) in scan_index.items() if s == args.subset)
    if not uids:
        raise SystemExit(f"No scans of subset{args.subset} found under {C.LUNA16_ROOT}")
    if args.max_scans:
        uids = uids[: args.max_scans]
    print(f"{len(uids)} scans of subset {args.subset}")

    done_path = os.path.join(OUT, "done_scans.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); todo = [u for u in uids if u not in done]
    with ThreadPoolExecutor(args.threads) as pool:
        for k, uid in enumerate(todo, 1):
            g = cands[cands.seriesuid == uid]
            centers = g[["coordX", "coordY", "coordZ"]].to_numpy()
            scan = Scan(scan_index[uid][0])
            for ci, c in enumerate(conds):
                sigma, keep = CONDITIONS[c]
                s = corrupt(scan, sigma, keep, seed=zlib.crc32(uid.encode()) + ci)
                patches = s.extract_many(centers, executor=pool) if len(centers) else np.zeros((0, 48, 48, 48), np.int16)
                r = pred.score(patches) if len(patches) else {"prob": [], "uncertainty": [], "diameter_mm": []}
                out = g[["seriesuid", "coordX", "coordY", "coordZ", "class"]].copy()
                out["probability"] = r["prob"]; out["uncertainty"] = r["uncertainty"]; out["diameter_pred_mm"] = r["diameter_mm"]
                f = os.path.join(OUT, f"pred_{c}.csv")
                out.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
            with open(done_path, "a") as fh:
                fh.write(uid + "\n")
            el = time.time() - t0
            print(f"  {k}/{len(todo)} scans, {el / 60:.1f} min, about {el / k * (len(todo) - k) / 60:.0f} min left", flush=True)

    # ---------------- metrics
    rows = []
    for c in conds:
        df = pd.read_csv(os.path.join(OUT, f"pred_{c}.csv")).drop_duplicates(["seriesuid", "coordX", "coordY", "coordZ"])
        df = df[df.seriesuid.isin(uids)]
        froc, _ = cpm_with_ci(df, ann[ann.seriesuid.isin(uids)], excl, uids, n_boot=args.n_boot)
        pos = df[df["class"] == 1]
        rows.append({"condition": c, "description": LABELS[c], "CPM": round(froc["cpm"], 4),
                     "CPM_95CI": f"{froc['cpm_ci95'][0]:.3f}-{froc['cpm_ci95'][1]:.3f}",
                     "sens_at_1FP": round(froc["sensitivity_at_fp"]["1"], 3), "sens_at_4FP": round(froc["sensitivity_at_fp"]["4"], 3),
                     "sens_at_8FP": round(froc["sensitivity_at_fp"]["8"], 3),
                     "mean_uncertainty_nodules": round(float(pos.uncertainty.mean()), 4),
                     "mean_uncertainty_all": round(float(df.uncertainty.mean()), 4), "n_scans": froc["n_scans"]})
        print(f"{c:18s} CPM {froc['cpm']:.4f}  ({froc['cpm_ci95'][0]:.3f}-{froc['cpm_ci95'][1]:.3f})")
    tab = pd.DataFrame(rows)
    if "clean" in set(tab.condition):
        base = float(tab.loc[tab.condition == "clean", "CPM"].iloc[0])
        tab["CPM_change_vs_clean"] = (tab.CPM - base).round(4)
    tab.to_csv(os.path.join(OUT, "robustness_table.csv"), index=False)

    # missing clinical data (already computed by luna25_final.py): cancer risk with vs without age
    fm = os.path.join(C.WORK_DIR, "luna25_final", "final_metrics.json")
    if os.path.exists(fm):
        m = json.load(open(fm))["models"]
        json.dump({"cancer_risk_auc_without_age": m["ensemble"]["test_auc"], "cancer_risk_auc_with_size_and_age": m["fusion"]["test_auc"]},
                  open(os.path.join(OUT, "missing_clinical_data.json"), "w"), indent=2)

    plot(tab)
    print(tab.to_string(index=False))
    print("->", OUT)


def plot(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    get = lambda c: float(tab.loc[tab.condition == c, "CPM"].iloc[0]) if c in set(tab.condition) else None
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=200, sharey=True)
    for ax, xs, cs, title, xlab in [
        (axes[0], [0, 25, 50, 100], ["clean", "noise25", "noise50", "noise100"], "Added noise", "Noise standard deviation (HU)"),
        (axes[1], [0, 50, 67], ["clean", "slices2", "slices3"], "Missing slices", "Slices removed (%)")]:
        pts = [(x, get(c)) for x, c in zip(xs, cs) if get(c) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#2a78d6", lw=2, ms=6, markeredgecolor="white")
            for x, y in pts:
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_title(title, loc="left", fontsize=11); ax.set_xlabel(xlab)
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("CPM (LUNA16 subset 9)"); axes[0].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "robustness.png")); plt.close(fig)


if __name__ == "__main__":
    main()
