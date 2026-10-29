"""Shared helpers for the LUNA25 steps: locate images + annotation CSV, normalise columns.

LUNA25 (NLST low-dose CT): 4,069 scans (.mha) and LUNA25_Public_Training_Development_Data.csv
with one row per nodule: world coordinates in mm (x, y, z) and label 1 = malignant, 0 = benign.
"""
import glob
import os

import pandas as pd

import config as C

L25_DIR = os.path.join(C.WORK_DIR, "luna25")
CSV_NAME = "LUNA25_Public_Training_Development_Data.csv"

# accepted column names (lower-case, without _ and spaces) -> our name
ALIASES = {
    "seriesuid": ["seriesinstanceuid", "seriesuid", "series", "scanid"],
    "coordX": ["coordx", "worldx", "x"],
    "coordY": ["coordy", "worldy", "y"],
    "coordZ": ["coordz", "worldz", "z"],
    "label": ["label", "malignancy", "malignant", "cancer", "class"],
    "patient": ["patientid", "patient", "pid"],
    "annotation": ["annotationid", "noduleid", "lesionid", "id"],
    "age": ["ageatstudydate", "age"],
    "gender": ["gender", "sex"],
    "study_date": ["studydate", "date"],
}


def _n(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def images_root():
    return C.find_luna25()


def index_images(root):
    """{SeriesInstanceUID: path} for every .mha / .mhd / .nii.gz under root (recursive)."""
    out = {}
    for ext in ("*.mha", "*.mhd", "*.nii.gz"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            stem = os.path.basename(p)
            for e in (".mha", ".mhd", ".nii.gz"):
                if stem.endswith(e):
                    stem = stem[: -len(e)]
            out[stem] = p
    return out


def find_csv(images_dir=None, ask=True):
    saved = os.path.join(C.WORK_DIR, "luna25_csv_path.txt")
    for p in (os.environ.get("LUNA25_CSV"), open(saved).read().strip() if os.path.exists(saved) else None):
        if p and os.path.isfile(p):
            return p
    starts = []
    if images_dir:
        d = os.path.abspath(images_dir)
        for _ in range(4):
            starts.append(d)
            d = os.path.dirname(d)
    l16 = os.path.abspath(C.LUNA16_ROOT)
    home = os.path.expanduser("~")
    starts += [os.path.dirname(l16), l16, os.path.abspath("."), os.path.abspath("..")] + \
              [os.path.join(home, x) for x in ("Downloads", "Desktop", "Documents")] + list(C.SEARCH_PLACES)
    exact, loose = [], []
    for s in starts:
        if not os.path.isdir(s):
            continue
        for dp, dns, fs in os.walk(s):
            if dp[len(s):].count(os.sep) >= 3:
                dns[:] = []
            dns[:] = [x for x in dns if x.lower() not in ("lungenv", "windows", "program files", "appdata", "$recycle.bin")
                      and not x.lower().startswith("luna25_images")]
            for f in fs:
                if f.lower() == CSV_NAME.lower():
                    exact.append(os.path.join(dp, f))
                elif f.lower().endswith(".csv") and "luna25" in _n(f):
                    loose.append(os.path.join(dp, f))
        if exact:
            break
    found = (exact or loose or [None])[0]
    if found is None and ask:
        print(f"Could not find {CSV_NAME}.")
        print("Download it from https://zenodo.org/records/14673658 , or paste the full path of the CSV here and press Enter:",
              flush=True)
        try:
            p = input().strip().strip('"')
        except EOFError:
            p = ""
        found = p if os.path.isfile(p) else None
    if found:
        os.makedirs(C.WORK_DIR, exist_ok=True)
        open(saved, "w").write(os.path.abspath(found))
    return found


def load_table(csv_path):
    df = pd.read_csv(csv_path)
    norm = {_n(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for ours, cands in ALIASES.items():
        for c in cands:
            if c in norm:
                out[ours] = df[norm[c]]
                break
    missing = [k for k in ("seriesuid", "coordX", "coordY", "coordZ", "label") if k not in out]
    if missing:
        raise SystemExit(f"CSV {csv_path} is missing columns {missing}. Columns found: {list(df.columns)}")
    if "patient" not in out:
        out["patient"] = out["seriesuid"]           # fall back to scan-level grouping
    if "annotation" not in out:
        out["annotation"] = [f"n{i}" for i in range(len(out))]
    out["seriesuid"] = out["seriesuid"].astype(str)
    out["patient"] = out["patient"].astype(str)
    out["label"] = out["label"].astype(int)
    return out
