"""Inspect the extracted LUNA25 folder and print everything needed to plan the next step.
Read-only: it never changes or moves your files.

    python luna25_audit.py                    (finds a folder named luna25 / LUNA 25 automatically)
    python luna25_audit.py "D:\\studentss\\LUNA25"
"""
import collections
import glob
import os
import sys

import config as C


def find_root():
    return C.find_luna25(sys.argv[1] if len(sys.argv) > 1 else None)


def main():
    root = find_root()
    if not root or not os.path.isdir(root):
        print("LUNA25 folder NOT FOUND automatically.")
        print("Open File Explorer, go into your LUNA25 folder, click the address bar, copy the path,")
        print("then paste it here (right-click pastes) and press Enter:", flush=True)
        try:
            root = input().strip().strip('"')
        except EOFError:
            root = ""
        if not os.path.isdir(root):
            print(f"Not a folder: {root!r}")
            return
    C.remember_luna25(root)          # every later LUNA25 step reuses this path
    print("LUNA25 folder:", os.path.abspath(root))
    ext = collections.Counter()
    dirs = collections.Counter()
    images, csvs, total = [], [], 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            e = ".nii.gz" if f.lower().endswith(".nii.gz") else os.path.splitext(f)[1].lower()
            ext[e] += 1
            dirs[os.path.relpath(dp, root).split(os.sep)[0]] += 1
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
            if e in (".mha", ".mhd", ".nii.gz", ".nii", ".nrrd", ".npy", ".dcm"):
                images.append(p)
            if e == ".csv":
                csvs.append(p)
    print(f"total size: {total / 1e9:.1f} GB")
    print("file types:", dict(ext.most_common(12)))
    print("top-level folders (files):", dict(dirs.most_common(12)))

    import pandas as pd
    for c in csvs[:6]:
        try:
            df = pd.read_csv(c)
        except Exception as e:
            print(f"\nCSV {c}: unreadable ({e})")
            continue
        print(f"\nCSV {os.path.relpath(c, root)}: {len(df)} rows")
        print("columns:", list(df.columns))
        print(df.head(3).to_string(max_colwidth=40))
        for col in df.columns:
            if any(k in col.lower() for k in ("label", "malig", "cancer", "class")):
                print(f"value counts of '{col}':", df[col].value_counts().head(10).to_dict())
        for col in df.columns:
            if any(k in col.lower() for k in ("patient", "series")):
                print(f"unique '{col}': {df[col].nunique()}")

    if images:
        import SimpleITK as sitk
        print(f"\nimage files: {len(images)}  e.g. {os.path.relpath(images[0], root)}")
        for p in images[:3]:
            try:
                if p.endswith(".npy"):
                    import numpy as np
                    a = np.load(p, mmap_mode="r")
                    print(f"  {os.path.basename(p)}: numpy {a.shape} {a.dtype}")
                    continue
                r = sitk.ImageFileReader()
                r.SetFileName(p)
                r.ReadImageInformation()
                print(f"  {os.path.basename(p)}: size {r.GetSize()} spacing {tuple(round(s, 3) for s in r.GetSpacing())} "
                      f"origin {tuple(round(o, 1) for o in r.GetOrigin())} direction {tuple(round(d, 2) for d in r.GetDirection())}")
            except Exception as e:
                print(f"  {os.path.basename(p)}: could not read ({e})")
    print("\nSend this whole output back.")


if __name__ == "__main__":
    main()
