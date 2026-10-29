# LungScan: LUNA16 pulmonary-nodule pipeline (FYP)

> **On Windows: open `START_HERE.txt` and double-click the numbered .bat files in order.**

Preprocessing, training, evaluation, and full-scan inference for **MSGF-Net**
(Multi-window, Scale-Gated Fusion network with diameter estimation and calibrated uncertainty).
Every script has been run end to end on a synthetic dataset in the exact LUNA16 layout.

## 0. Read this first: what LUNA16 can and cannot give you

| Your files | What they contain |
|---|---|
| `subset0..9/*.mhd + *.raw` | 888 CT scans. `.mhd` is the header (origin, spacing, direction), `.raw` holds the voxels in Hounsfield units |
| `candidates_V2.csv` | ~754k candidate locations (world mm), `class` 1 = nodule, 0 = not a nodule |
| `annotations.csv` | 1,186 reference nodules ≥3 mm with `diameter_mm` |

So your filled-in template reads:
* **Task:** binary 3D classification (nodule vs. non-nodule), i.e. the LUNA16 *false-positive
  reduction* track, plus nodule-diameter regression as an auxiliary task
* **State:** raw and uncleaned, with extreme imbalance (~490 negatives per positive)
* **Labels:** the `class` column of candidates_V2.csv, with `diameter_mm` from annotations.csv
* **Framework:** PyTorch

**LUNA16 has no benign/malignant labels.** "Lung cancer prediction" here means *detecting
nodules*. Nodules are the precursor lesions a radiologist then risk-stratifies. The pipeline
therefore outputs nodule probability, size, a size-based Lung-RADS category and an
uncertainty flag. Do not write "cancer / malignancy classification on LUNA16" in the paper:
reviewers will reject it. Two of the papers you downloaded (MIFNet 2024 and the IJRTI
EfficientNet paper) call the LUNA16 classes "benign/malignant", which is a labelling error.
True malignancy needs LIDC-IDRI radiologist ratings (the `pylidc` package). That is a
reasonable future-work item.

## 1. Where to run it

LUNA16 is about 120 GB. **Kaggle** is the easiest option: add a public LUNA16 dataset as notebook
input (search "LUNA16"), enable a GPU (T4/P100), and upload this folder as a dataset.

```bash
pip install -r requirements.txt           # Kaggle already has everything except SimpleITK
export LUNA16_ROOT=/kaggle/input/<luna16-dataset-folder>   # must contain subset0..subset9 + CSVs
export WORK_DIR=/kaggle/working/work     # needs ~10 GB (patches are ~8 GB)
```
If the Kaggle copy lacks `candidates_V2.csv` or `annotations_excluded.csv`, download them
from the LUNA16 site (`evaluationScript/annotations/annotations_excluded.csv`). Set
`CANDIDATES_CSV=` to use the older `candidates.csv` instead.
In a Kaggle/Colab notebook, prefix every command with `!` and set the variables with
`%env LUNA16_ROOT=...`.

## 2. Commands, in order

| Step | Command | Time (T4 GPU, 4 CPU) | Output |
|---|---|---|---|
| Audit data | `python preprocess.py --audit` | 2–5 min | `patches/audit.json`: dataset table for the paper |
| Preprocess | `python preprocess.py` | 30–90 min, resumable | `patches/patches.npy`, `meta.csv` |
| Train (round 1) | `python train.py --name msgf_f9` | ~1–2 h | `runs/msgf_f9/best.pt`, `history.csv` |
| Mine hard negatives | `python mine_hard_negatives.py --run msgf_f9` | ~40–70 min | `patches/hard_msgf_f9.csv` |
| Train (round 2) | see `run_all.sh` | ~30 min | `runs/msgf_f9_r2/best.pt` |
| Evaluate + calibrate | `python evaluate.py --run msgf_f9_r2 --calibrate` | ~30–45 min | CPM, FROC, ECE, triage, plots |
| Ablations | 5 × train + evaluate (`run_all.sh`) | ~8 h total | ablation table |
| Results table | `python collect_results.py` | seconds | `results_table.csv` |
| One scan (app) | `python infer.py --ckpt runs/msgf_f9_r2/best.pt --scan X.mhd --out rep/` | ~10 s | `report.json` + PNG snapshots |

`run_all.sh` has the whole sequence. Kaggle sessions stop after 12 h, so run each stage in
its own session. `train.py --resume` continues from `last.pt`, and preprocessing skips scans
that are already done.
Quick check before the real run (5 min): `python train.py --name dry --epochs 1 --samples-per-epoch 2000`.

## 3. What preprocessing does (and why)

1. **Indexes** every `.mhd` and maps each seriesuid to its subset (patient-level folds).
2. **Audits** scan counts, spacing range, class imbalance, nodule sizes and missing scans.
3. Converts **world mm to voxel** coordinates using origin, spacing *and direction* (some scans are
   flipped, which is a classic silent bug). This was verified on synthetic flipped scans.
4. **Resamples each patch to 1 mm isotropic** by trilinear interpolation straight from the
   original volume, so slice thickness from 0.6 to 2.5 mm stops mattering.
5. **Clips HU** to [-1200, 1200], stores int16 48³ cubes, and pads outside the field of view with air.
6. **Samples:** all positives, 10 "near-miss" negatives 5–25 mm from a nodule and 30 random
   negatives per scan. The test subset is still scored on **all** of its candidates.
7. Training-time only: **multi-window encoding** (lung / mediastinal / wide → 3 channels),
   flips, 90° rotations, ±3 mm shifts, HU jitter, and balanced batches (25% positives).

Why no SMOTE, CLAHE, 2D slices, or "augment then split": these leak information between
train and test or throw away 3D information. The ~97%-accuracy papers you downloaded report
accuracy on balanced patch sets, which says nothing about false positives per scan. The
LUNA16 metric is **CPM**, the mean sensitivity at 1/8…8 FPs/scan.

## 4. Files

```
config.py              all paths and hyper-parameters
luna_io.py             .mhd/.raw/DICOM reading, world→voxel, 1 mm patch extraction, HU windows
preprocess.py          audit + patch extraction (multiprocess, resumable)
dataset.py             dataset, augmentation, balanced sampler, patient-level split with leak check
model.py               MSGF-Net
train.py               focal + uncertainty-weighted multi-task training, AMP, early stopping
mine_hard_negatives.py round-2 hard-negative curriculum
predictor.py           TTA + MC-dropout scoring, Platt calibration, Lung-RADS size category
metrics.py             LUNA16 FROC/CPM with bootstrap CI, ECE, triage curve
evaluate.py            full held-out-subset evaluation + figures
candidate_gen.py       training-free candidate detector for new scans (apps)
infer.py               LungCAD.analyze(scan) → JSON report (what the doctor app calls)
collect_results.py     one table of all runs
tests/make_synthetic.py fake LUNA16 for smoke tests
METHODOLOGY.md         the method written for the paper + experiment plan
```

## 5. Using it from the apps

```python
from infer import LungCAD
cad = LungCAD("work/runs/msgf_f9_r2/best.pt")          # load once at server start
report = cad.analyze("upload/scan.mhd", out_dir="static/reports/123")   # .mhd or DICOM folder
```
`report["summary"]` and `report["findings"]` (probability, uncertainty, needs_review,
diameter, Lung-RADS size category, snapshot PNG) are what the doctor dashboard displays.
