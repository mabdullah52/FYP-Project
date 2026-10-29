"""LUNA25 step 1 - audit + patient-level folds + 48 mm patches around every nodule (resumable).

    python luna25_prepare.py
Output in WORK_DIR/luna25/: patches.npy (int16, N x 48^3, 1 mm), meta.csv, audit.json
Split (patient level, stratified by malignancy, 5 folds, seed 2026):
    fold 0 = TEST, fold 1 = VALIDATION, folds 2-4 = TRAINING
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C
from luna25_common import L25_DIR, find_csv, images_root, index_images, load_table
from luna_io import iter_scan_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(L25_DIR, exist_ok=True)

    root = images_root()
    if not root:
        raise SystemExit("LUNA25 images not found - run 10_LUNA25_CHECK_AND_STATS.bat first")
    print("LUNA25 images:", root)
    imgs = index_images(root)
    csv = find_csv(root)
    if not csv:
        raise SystemExit("LUNA25 annotation CSV not found.")
    print("LUNA25 annotations:", csv)
    df = load_table(csv)
    n_all = len(df)
    df["on_disk"] = df.seriesuid.isin(imgs)
    audit = {
        "images_found": len(imgs), "csv_rows": n_all, "nodules_with_image": int(df.on_disk.sum()),
        "malignant": int(df.label.sum()), "benign": int((df.label == 0).sum()),
        "patients": int(df.patient.nunique()), "scans_in_csv": int(df.seriesuid.nunique()),
        "malignant_with_image": int(df[df.on_disk].label.sum()),
        "columns_used": [c for c in df.columns if c != "on_disk"],
    }
    if "age" in df:
        audit["age_mean_sd"] = [round(float(df.age.mean()), 1), round(float(df.age.std()), 1)]
    df = df[df.on_disk].drop(columns="on_disk").reset_index(drop=True)

    # patient-level, malignancy-stratified folds (all nodules of a patient in one fold)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df.label, groups=df.patient)):
        df.loc[te, "fold"] = k
    df["role"] = df.fold.map({0: "test", 1: "val"}).fillna("train")
    for r in ("train", "val", "test"):
        d = df[df.role == r]
        audit[f"{r}_nodules_malignant_patients"] = [int(len(d)), int(d.label.sum()), int(d.patient.nunique())]
    assert not (set(df[df.role == "test"].patient) & set(df[df.role != "test"].patient)), "patient leakage!"
    print(json.dumps(audit, indent=2))

    meta_path = os.path.join(L25_DIR, "meta.csv")
    arr_path = os.path.join(L25_DIR, "patches.npy")
    done_path = os.path.join(L25_DIR, "done_scans.txt")
    if os.path.exists(meta_path) and os.path.exists(arr_path):
        meta = pd.read_csv(meta_path, dtype={"seriesuid": str, "patient": str})
        arr = np.load(arr_path, mmap_mode="r+")
        print(f"Resuming: {len(meta)} patches planned")
    else:
        meta = df.copy()
        meta["row"] = np.arange(len(meta))
        meta["diameter_mm"] = np.nan          # LUNA25 has no diameters; the diameter head is not trained here
        meta["file"] = "patches.npy"
        meta.to_csv(meta_path, index=False)
        s = C.STORE_SIZE
        arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.int16, shape=(len(meta), s, s, s))
        if os.path.exists(done_path):
            os.remove(done_path)
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    groups = {u: g for u, g in meta.groupby("seriesuid")}
    jobs = [(u, imgs[u], g[["coordX", "coordY", "coordZ"]].to_numpy(float)) for u, g in groups.items() if u not in done]
    print(f"Extracting {len(jobs)} scans ({len(meta)} nodules, ~{arr.nbytes / 1e9:.1f} GB) ...")
    failed, consecutive = [], 0
    with open(done_path, "a") as fdone:
        for k, (uid, patches, err) in enumerate(iter_scan_patches(jobs, args.workers), 1):
            if err:
                failed.append(uid); consecutive += 1
                print("  FAILED", uid, err[:150])
                if consecutive >= 5:
                    print("STOPPED: 5 scans in a row failed - check memory/disk, then run again (it resumes).")
                    break
                continue
            consecutive = 0
            arr[groups[uid].row.to_numpy()] = patches
            fdone.write(uid + "\n"); fdone.flush()
            if k % 50 == 0 or k == len(jobs):
                arr.flush()
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} scans  {el / 60:.1f} min, ~{el / k * (len(jobs) - k) / 60:.1f} min left")
    arr.flush()
    # sanity: nodules must be denser than lung at the patch centre
    c = C.STORE_SIZE // 2
    rows = meta.row.to_numpy()[: min(len(meta), 2000)]
    centre = np.array([np.asarray(arr[r])[c - 1:c + 1, c - 1:c + 1, c - 1:c + 1].mean() for r in rows])
    audit["sanity_centre_HU_median"] = float(np.median(centre))
    audit["failed_scans"] = len(failed)
    json.dump(audit, open(os.path.join(L25_DIR, "audit.json"), "w"), indent=2)
    print(f"Sanity: median HU at nodule centres = {np.median(centre):.0f} (lung is about -850; nodules are much higher)")
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {L25_DIR}")


if __name__ == "__main__":
    main()
