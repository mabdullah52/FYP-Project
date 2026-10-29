"""Central configuration for the LUNA16 pipeline.

Edit the paths in PATHS (or override them on the command line) and leave the rest
alone for the first full run. Every script imports from here so that preprocessing,
training, evaluation and the apps all use exactly the same settings.
"""
import glob
import os

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
# Where is your LUNA16 folder? Leave empty ("") to let the code FIND it automatically
# (it looks in D:\students, D:\studentss, D:\ and the code folder for a folder whose
# name starts with "luna" and that contains subset0 ... subset9).
# If it can't find it, paste the full path between the quotes, e.g. r"D:\students\luna16"
MY_LUNA16_FOLDER = r""

SEARCH_PLACES = [r"D:\students", r"D:\studentss", r"D:\Students", "D:\\", "..", "."]


def _has_subsets(p):
    return bool(glob.glob(os.path.join(p, "subset*")))


def _find_luna16():
    if os.environ.get("LUNA16_ROOT"):
        return os.environ["LUNA16_ROOT"]
    if MY_LUNA16_FOLDER:
        return MY_LUNA16_FOLDER
    for place in SEARCH_PLACES:
        if not os.path.isdir(place):
            continue
        if _has_subsets(place) and os.path.basename(os.path.abspath(place)).lower().startswith("luna"):
            return os.path.abspath(place)
        for name in sorted(os.listdir(place)):
            p = os.path.join(place, name)
            if not (name.lower().startswith("luna") and os.path.isdir(p)):
                continue
            if _has_subsets(p):
                return os.path.abspath(p)
            inner = glob.glob(os.path.join(p, "*", "subset*"))       # e.g. luna16\LUNA16\subset0
            if inner:
                return os.path.abspath(os.path.dirname(inner[0]))
    return "./LUNA16"


def _find_csv(root, *names):
    """CSV files may sit next to the subsets or in a sub-folder (e.g. CSVFILES, annotations)."""
    for n in names:
        for pattern in (n, os.path.join("*", n), os.path.join("*", "*", n),
                        os.path.join("..", n), os.path.join("..", "*", n)):
            hits = glob.glob(os.path.join(root, pattern))
            if hits:
                return os.path.abspath(hits[0])
    return os.path.join(root, names[0])


# Where is your LUNA25 folder? Leave empty to find it automatically (any folder whose name
# contains "luna25", ignoring spaces / dashes / underscores / capitals, on any drive).
MY_LUNA25_FOLDER = r""

_SKIP = {"windows", "program files", "program files (x86)", "programdata", "$recycle.bin", "appdata",
         "lungenv", "node_modules", "system volume information", "__pycache__", "work", "app_data"}


def _norm(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _looks_like_luna25(path):
    """True if the folder holds CT images or a LUNA25 csv somewhere in its first 2 levels."""
    for dp, dns, fs in os.walk(path):
        if dp[len(path):].count(os.sep) >= 2:
            dns[:] = []
        if any(f.lower().endswith((".mha", ".mhd", ".nii.gz", ".npy", ".csv", ".zip")) for f in fs):
            return True
    return False


def find_luna25(extra=None, max_depth=3):
    """Return the LUNA25 folder, searching: argument / env / config, then next to LUNA16, the code
    folder and its parents, D:\\students*, every drive root, Downloads / Desktop / Documents."""
    saved = os.path.join(WORK_DIR, "luna25_path.txt")
    saved = open(saved).read().strip() if os.path.exists(saved) else None
    for p in (extra, os.environ.get("LUNA25_ROOT"), MY_LUNA25_FOLDER, saved):
        if p and os.path.isdir(p.strip().strip('"')):
            return os.path.abspath(p.strip().strip('"'))
    here = os.path.abspath(".")
    l16 = os.path.abspath(_find_luna16())
    starts = [os.path.dirname(l16), os.path.dirname(os.path.dirname(l16)), here, os.path.dirname(here),
              os.path.dirname(os.path.dirname(here))] + list(SEARCH_PLACES)
    home = os.path.expanduser("~")
    starts += [os.path.join(home, d) for d in ("Downloads", "Desktop", "Documents")] + [home]
    if os.name == "nt":
        import string
        starts += [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
    else:
        starts += ["/"]
    seen, hits = set(), []
    for s in starts:
        if not s or not os.path.isdir(s):
            continue
        s = os.path.abspath(s)
        for dp, dns, _ in os.walk(s):
            depth = dp[len(s):].count(os.sep)
            dns[:] = [d for d in dns if d.lower() not in _SKIP and not d.startswith(".")]
            for d in dns:
                full = os.path.join(dp, d)
                if "luna25" in _norm(d) and full not in seen:
                    seen.add(full)
                    hits.append(full)
            if depth >= max_depth - 1:
                dns[:] = []
        if hits:
            break
    good = [h for h in hits if _looks_like_luna25(h)]
    # prefer the outermost matching folder that really contains data
    good.sort(key=lambda p: (p.count(os.sep), len(p)))
    return good[0] if good else (hits[0] if hits else None)


def remember_luna25(path):
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(os.path.join(WORK_DIR, "luna25_path.txt"), "w") as f:
        f.write(os.path.abspath(path))


# LUNA16_ROOT must contain subset0 ... subset9 folders (each full of *.mhd + *.raw).
LUNA16_ROOT = _find_luna16()
CANDIDATES_CSV = os.environ.get("CANDIDATES_CSV") or _find_csv(LUNA16_ROOT, "candidates_V2.csv", "candidates.csv")
ANNOTATIONS_CSV = os.environ.get("ANNOTATIONS_CSV") or _find_csv(LUNA16_ROOT, "annotations.csv")
# Optional (from the official evaluationScript folder). Findings listed here are
# neither TP nor FP in the CPM/FROC evaluation, exactly like the official script.
EXCLUDED_CSV = os.environ.get("EXCLUDED_CSV") or _find_csv(LUNA16_ROOT, "annotations_excluded.csv")

WORK_DIR = os.environ.get("WORK_DIR", "./work")          # patches, checkpoints, results
PATCH_DIR = os.path.join(WORK_DIR, "patches")
RUNS_DIR = os.path.join(WORK_DIR, "runs")

# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
TARGET_SPACING_MM = 1.0         # isotropic resampling of every patch (mm / voxel)
STORE_SIZE = 48                 # stored cube edge (voxels = mm). Context stream sees all of it.
LOCAL_SIZE = 32                 # local stream: centre crop of the stored cube
HU_CLIP = (-1200, 1200)         # values stored as int16 inside this range
PAD_HU = -1000                  # value used outside the scanned field of view (air)

NEG_PER_SCAN = 30               # random negatives kept per scan for training
NEAR_NEG_PER_SCAN = 10          # extra "near-miss" negatives (5-25 mm from a nodule)
NEAR_NEG_RANGE_MM = (5.0, 25.0)
SEED = 2026

# Multi-window HU encoding: each window -> one input channel scaled to [0,1]
HU_WINDOWS = [
    (-1350, 150),   # lung window      (WL -600, WW 1500)
    (-160, 240),    # mediastinal / soft-tissue window (WL 40, WW 400)
    (-1000, 400),   # wide window keeping both parenchyma and solid tissue
]

# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
TRAIN = dict(
    epochs=30,
    batch_size=32,               # 32 fits an 8 GB GPU in FP32 (64 needs ~6 GB of activations alone)
    samples_per_epoch=24000,    # positives oversampled to POS_FRACTION of every batch
    pos_fraction=0.25,
    lr=1e-3,
    weight_decay=1e-4,
    focal_gamma=2.0,
    focal_alpha=0.5,
    num_workers=8,               # data-loading THREADS (not processes)
    width=24,                   # base channel width of the network
    dropout=0.3,
    amp=True,
)

# GPU-specific batch size chosen automatically by check_setup.py (2_CHECK.bat)
LOCAL_SETTINGS = os.path.join(WORK_DIR, "local_settings.json")
if os.path.exists(LOCAL_SETTINGS):
    import json as _json
    try:
        TRAIN.update({k: v for k, v in _json.load(open(LOCAL_SETTINGS)).items() if k in TRAIN})
    except Exception:
        pass

# Final model, chosen on the VALIDATION subset (step 8): round 1 had the best validation CPM.
FINAL_RUN = os.environ.get("FINAL_RUN", "msgf_f9")

# Fold protocol (patient-level, LUNA16 official subsets). Default = one held-out fold.
DEFAULT_TEST_SUBSET = 9
DEFAULT_VAL_SUBSET = 8

# --------------------------------------------------------------------------------------
# Evaluation / inference
# --------------------------------------------------------------------------------------
FROC_FP_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
MC_SAMPLES = 10                 # MC-dropout passes of the (cheap) head
TTA = True                      # 8 flip combinations at test time


def amp_enabled(device):
    """Mixed precision only on GPUs with fast FP16 (Volta / compute capability >= 7).
    Pascal cards such as the Quadro P4000 (6.1) run FP16 maths very slowly -> use FP32."""
    import torch
    if device.type != "cuda" or not TRAIN["amp"]:
        return False
    return torch.cuda.get_device_capability(device)[0] >= 7
