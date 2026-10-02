# End-to-End Pipeline

This is the planned workflow from raw data to the two applications. It is a **plan**: each stage's real status is tracked in [PROGRESS.md](../PROGRESS.md), and no stage should be described as complete until its code and outputs are in this repository.

## Overview

```
 ┌────────────────────────── TRAINING DATA ──────────────────────────┐      ┌──── EXTERNAL TEST DATA ────┐
 │  LUNA16 (888 CTs, subset of LIDC-IDRI)   +   LIDC-IDRI (1,018 CTs)  │      │  LUNA25 (NLST screening CT) │
 └───────────────────────────────┬────────────────────────────────────┘      └─────────────┬──────────────┘
                                 ▼                                                        │  never used for
 [1] Data harmonisation: de-duplicate LUNA16 ⊂ LIDC-IDRI · patient-level split · labels     │  training or tuning
                                 ▼                                                        │
 [2] Preprocessing: resample ~1 mm³ · HU clip [-1000, 400] · normalise · slices · augment  │
                                 ▼                                                        │
 [3] Nodule segmentation (modified U-Net)                                                 │
                                 ▼                                                        │
 [4] Nodule malignancy classifier (core) · slice-sequence model (extension)               │
                                 ▼                                                        │
 [5] Growth-region forecasting (STRETCH)                                                  │
                                 ▼                                                        │
 [6] Explainability: confidence colours + Grad-CAM++                                      │
                                 ▼                                                        │
 [7] Internal validation on held-out LUNA16 / LIDC-IDRI patients                          │
                                 ▼                                                        ▼
 [8] External testing on LUNA25 ◀─────────────────────────────────────────────────────────┘
                                 ▼
 [9] Backend REST API (serves the trained models)
                ┌────────────────┴────────────────┐
                ▼                                 ▼
 [10] Doctor web app (core)               [11] Patient app (STRETCH)
      radiologist decision support             with agentic AI assistant
```

## Scope: what is realistic

Not every stage carries the same risk. The table below separates what the project commits to from what is attempted only if time and data allow.

| Stage | Tier | Reason |
|---|---|---|
| 1 Data harmonisation | ✅ Core | Public data with known overlap; well-documented tools (`pylidc`) |
| 2 Preprocessing | ✅ Core | Standard steps |
| 3 U-Net segmentation | ✅ Core | LIDC-IDRI provides radiologist outlines to train on |
| 4a Nodule malignancy classifier (nodule-centred patches) | ✅ Core | LIDC-IDRI malignancy ratings for training, LUNA25 labels for testing |
| 4b Slice-sequence model (CNN → Bi-LSTM / Transformer) | 🟡 Extension | Feasible, but there are **no per-slice malignancy labels**, so it can only be judged at nodule level against 4a (ablation) |
| 5 Growth-region forecasting | 🔵 Stretch | Needs repeat scans of the same nodule. LUNA16 / LIDC-IDRI have one scan per patient. LUNA25 has repeat scans but is reserved as the unseen test set. A longitudinal NLST dataset would require a separate data-access application |
| 6 Confidence colours + Grad-CAM++ | ✅ Core | Applied on top of 4a |
| 7 Internal validation | ✅ Core | |
| 8 LUNA25 external test | ✅ Core | Public data with malignancy labels. Runs on the frozen model |
| 9 Backend API | ✅ Core | |
| 10 Doctor web app | ✅ Core | For a new scan, the nodule location comes from the doctor clicking on it (core) or from an automatic detector (🟡 extension) |
| 11 Patient app with AI assistant | 🔵 Stretch | A separate product. Patient-facing medical AI needs doctor review of anything shown |
| Official LUNA25 leaderboard (Docker submission) | 🔵 Stretch | Optional. The official test set is hidden |

**Rule:** report results only for stages that were actually run. Stretch items go under "Future work" in the report unless they are completed. Agree any scope change from the proposal with the supervisor.

## Practical notes

- **LUNA25 size:** the full image set is about 221 GB. Zenodo also provides a separate nodule-blocks archive (crops around each nodule), which is far smaller and enough for nodule-level malignancy testing.
- **Label difference:** LIDC-IDRI malignancy is a 1–5 radiologist rating, while LUNA25 uses binary malignant/benign labels. Expect lower external scores, and report the drop honestly. It is a finding, not a failure.
- **GPU memory:** with an 8 GB GPU, train on nodule-centred patches (e.g. 64³ voxels for 3D) rather than whole volumes.

## Stage details

### 1. Data harmonisation
- **LUNA16 is built from LIDC-IDRI.** The same scans appear in both. Before splitting, de-duplicate by `SeriesInstanceUID` (LUNA16 file names are the series UIDs), so the same scan never ends up in both training and validation.
- **Split by patient, not by slice or nodule**, to prevent leakage.
- **Harmonise labels.** The datasets label different things:

  | Dataset | Labels available | Used for |
  |---|---|---|
  | LUNA16 | Nodule locations and diameters (`annotations.csv`) | Detection / segmentation |
  | LIDC-IDRI | Up to 4 radiologists' nodule outlines + malignancy ratings 1–5 | Segmentation masks, malignancy labels |
  | LUNA25 | Nodule locations + binary malignant / benign labels | External malignancy test |

  ❓ Decide and document the rule that maps LIDC-IDRI's 1–5 ratings to benign/malignant (e.g. how rating 3 is handled), because it directly affects comparison with LUNA25.

### 2. Preprocessing
Resample to ~1×1×1 mm, clip HU to [-1000, 400], normalise to [0, 1], extract axial slices, and augment (rotation, flip, Random Pixel Swap). **Apply the same preprocessing to LUNA25** (it is screening low-dose CT from a different source).

### 3–6. Models
See [ARCHITECTURE.md](ARCHITECTURE.md). Core model: a malignancy classifier on nodule-centred patches. Extension: a slice-sequence model compared against it. Stretch: growth forecasting.

### 7. Internal validation
Held-out LUNA16 / LIDC-IDRI patients. Use it for model selection and threshold tuning.

### 8. External testing on LUNA25
- Freeze the models and thresholds first, then run once on LUNA25.
- Report the same metrics as internal validation (AUC-ROC, sensitivity, specificity, accuracy; Dice/IoU only where masks exist).
- Comparing internal and external results shows how well the model generalises to a new population and scanners.

### 9. Backend API
Planned contract: [API.md](API.md).

### 10. Doctor web app
Radiologist-facing: upload a scan, browse slices, mark or select a nodule, and see its risk score, confidence colour and Grad-CAM++ heatmap (plus the growth forecast, if that stretch goal is reached). Code: `apps/doctor-web-app/`.

### 11. Patient app (stretch)
Patient-facing app with an agentic AI assistant. Built only after the core pipeline works. Code: `apps/patient-app/`. Features to be defined. ❓ Also decide which results patients can see and whether a doctor reviews them first.

## Dataset roles at a glance

| Dataset | Train | Validate | External test |
|---|:-:|:-:|:-:|
| LUNA16 | ✔ | ✔ (held-out patients) | |
| LIDC-IDRI | ✔ | ✔ (held-out patients) | |
| LUNA25 | | | ✔ |
