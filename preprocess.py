"""Step 1 - audit and preprocess LUNA16.

    python preprocess.py --audit          # only check the data and print a report
    python preprocess.py                  # audit + extract training patches (resumable)

What it produces in WORK_DIR/patches/:
    patches.npy   int16 memmap (N, 48, 48, 48)  HU values, 1 mm isotropic
    meta.csv      one row per patch: seriesuid, subset, coords, label, diameter_mm, kind
    audit.json    dataset statistics (use these numbers in the paper's Dataset section)

Sampling (per scan, patient-level, all 10 subsets so any fold can be used later):
    * every positive candidate (class = 1), with its nodule diameter from annotations.csv
    * NEAR_NEG_PER_SCAN "near-miss" negatives 5-25 mm from a nodule (vessels/pleura
      next to nodules - the hardest false positives)
    * NEG_PER_SCAN random negatives
The held-out test subset is later scored on ALL of its candidates (evaluate.py), so
the sub-sampling here never touches the reported test numbers.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk

import config as C
from luna_io import index_scans, iter_scan_patches


def load_tables(scan_index):
    cands = pd.read_csv(C.CANDIDATES_CSV)
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    n0 = len(cands)
    cands = cands[cands.seriesuid.isin(scan_index)].reset_index(drop=True)
    ann = ann[ann.seriesuid.isin(scan_index)].reset_index(drop=True)
    cands["subset"] = cands.seriesuid.map(lambda s: scan_index[s][1])
    ann["subset"] = ann.seriesuid.map(lambda s: scan_index[s][1])
    return cands, ann, n0


def attach_diameters(cands, ann):
    """For every candidate: distance to nearest annotated nodule and that nodule's diameter."""
    cands["nn_dist_mm"] = np.inf
    cands["diameter_mm"] = np.nan
    for uid, g in ann.groupby("seriesuid"):
        m = cands.seriesuid == uid
        cxyz = cands.loc[m, ["coordX", "coordY", "coordZ"]].to_numpy()
        axyz = g[["coordX", "coordY", "coordZ"]].to_numpy()
        d = np.linalg.norm(cxyz[:, None, :] - axyz[None, :, :], axis=-1)
        j = d.argmin(1)
        cands.loc[m, "nn_dist_mm"] = d[np.arange(len(j)), j]
        diam = g.diameter_mm.to_numpy()[j]
        inside = d[np.arange(len(j)), j] <= np.maximum(diam / 2.0, 3.0)
        cands.loc[m, "diameter_mm"] = np.where(inside, diam, np.nan)
    # positives with no matching annotation (rare in V2): keep label, diameter unknown
    cands.loc[cands["class"] == 0, "diameter_mm"] = np.nan
    return cands


def select_samples(cands, rng):
    rows = []
    for uid, g in cands.groupby("seriesuid"):
        pos = g[g["class"] == 1]
        neg = g[g["class"] == 0]
        lo, hi = C.NEAR_NEG_RANGE_MM
        near = neg[(neg.nn_dist_mm >= lo) & (neg.nn_dist_mm <= hi)]
        near = near.sample(min(len(near), C.NEAR_NEG_PER_SCAN), random_state=rng)
        rest = neg.drop(near.index)
        rand = rest.sample(min(len(rest), C.NEG_PER_SCAN), random_state=rng)
        rows += [pos.assign(kind="pos"), near.assign(kind="near_neg"), rand.assign(kind="rand_neg")]
    sel = pd.concat(rows).sort_values(["seriesuid"]).reset_index(drop=True)
    sel = sel.rename(columns={"class": "label"})
    return sel


def audit(scan_index, cands, ann, n_cands_csv, read_headers=True):
    rep = {
        "scans_found": len(scan_index),
        "scans_per_subset": pd.Series([v[1] for v in scan_index.values()]).value_counts().sort_index().to_dict(),
        "candidates_in_csv": int(n_cands_csv),
        "candidates_for_found_scans": int(len(cands)),
        "positive_candidates": int((cands["class"] == 1).sum()),
        "negative_candidates": int((cands["class"] == 0).sum()),
        "annotated_nodules": int(len(ann)),
        "scans_with_nodules": int(ann.seriesuid.nunique()),
        "nodule_diameter_mm": ann.diameter_mm.describe().round(2).to_dict(),
        "positives_without_diameter_match": int(cands[(cands["class"] == 1)].diameter_mm.isna().sum()),
    }
    rep["imbalance_neg_per_pos"] = round(rep["negative_candidates"] / max(1, rep["positive_candidates"]), 1)
    missing = set(pd.read_csv(C.ANNOTATIONS_CSV).seriesuid) - set(scan_index)
    rep["annotated_scans_missing_on_disk"] = len(missing)
    if read_headers:
        sp, sh = [], []
        for mhd, _ in scan_index.values():
            r = sitk.ImageFileReader()
            r.SetFileName(mhd)
            r.ReadImageInformation()
            sp.append(r.GetSpacing())
            sh.append(r.GetSize())
        sp, sh = np.array(sp), np.array(sh)
        rep["spacing_xy_mm"] = [float(sp[:, 0].min()), float(sp[:, 0].max())]
        rep["spacing_z_mm"] = [float(sp[:, 2].min()), float(sp[:, 2].max())]
        rep["slices_per_scan"] = [int(sh[:, 2].min()), int(sh[:, 2].max())]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="only print the data report")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2),
                    help="threads for patch extraction (one process only)")
    args = ap.parse_args()

    os.makedirs(C.PATCH_DIR, exist_ok=True)
    t0 = time.time()
    scan_index = index_scans()
    cands, ann, n0 = load_tables(scan_index)
    cands = attach_diameters(cands, ann)
    rep = audit(scan_index, cands, ann, n0)
    print(json.dumps(rep, indent=2))
    with open(os.path.join(C.PATCH_DIR, "audit.json"), "w") as f:
        json.dump(rep, f, indent=2)
    if args.audit:
        return

    rng = np.random.RandomState(C.SEED)
    meta_path = os.path.join(C.PATCH_DIR, "meta.csv")
    arr_path = os.path.join(C.PATCH_DIR, "patches.npy")
    done_path = os.path.join(C.PATCH_DIR, "done_scans.txt")

    if os.path.exists(meta_path) and os.path.exists(arr_path):
        sel = pd.read_csv(meta_path)
        arr = np.load(arr_path, mmap_mode="r+")
        print(f"Resuming: {len(sel)} patches planned")
    else:
        sel = select_samples(cands, rng)
        sel["row"] = np.arange(len(sel))
        sel.to_csv(meta_path, index=False)
        s = C.STORE_SIZE
        arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.int16, shape=(len(sel), s, s, s))
        if os.path.exists(done_path):
            os.remove(done_path)
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()

    print(f"Patches: {len(sel)}  (pos={int((sel.label == 1).sum())}, near_neg={int((sel.kind == 'near_neg').sum())}, "
          f"rand_neg={int((sel.kind == 'rand_neg').sum())}); size on disk ~{arr.nbytes / 1e9:.1f} GB")

    jobs = []
    for uid, g in sel.groupby("seriesuid"):
        if uid in done:
            continue
        jobs.append((uid, scan_index[uid][0], g[["coordX", "coordY", "coordZ"]].to_numpy()))
    rows_of = {uid: g.row.to_numpy() for uid, g in sel.groupby("seriesuid")}

    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"RAM available: {vm.available / 1e9:.1f} GB of {vm.total / 1e9:.1f} GB")
    except ImportError:
        pass
    failed, consecutive = [], 0
    with open(done_path, "a") as fdone:
        for k, (uid, patches, err) in enumerate(iter_scan_patches(jobs, args.workers), 1):
            if err is not None:
                failed.append((uid, err))
                consecutive += 1
                print("  FAILED", uid, err[:200])
                if consecutive >= 5:
                    print("\nSTOPPED: 5 scans in a row failed. If the error says memory/allocate, close other "
                          "programs (or restart the PC) and run again - finished scans are kept.")
                    break
                continue
            consecutive = 0
            arr[rows_of[uid]] = patches
            fdone.write(uid + "\n")
            fdone.flush()
            if k % 20 == 0 or k == len(jobs):
                arr.flush()
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} scans  {el / 60:.1f} min elapsed, ~{el / k * (len(jobs) - k) / 60:.1f} min left")
    arr.flush()
    if failed:
        print(f"{len(failed)} scans failed (re-download them and re-run; finished scans are skipped).")
    # sanity check on a few patches
    pos_rows = sel[sel.label == 1].row.to_numpy()[:50]
    if len(pos_rows):
        centre = np.asarray(arr[pos_rows])[:, 20:28, 20:28, 20:28].mean()
        border = np.asarray(arr[pos_rows])[:, :4, :4, :4].mean()
        print(f"Sanity: mean HU at centre of positive patches {centre:.0f} vs corner {border:.0f} "
              "(centre should be clearly higher - nodules are denser than lung)")
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {C.PATCH_DIR}")


if __name__ == "__main__":
    main()
