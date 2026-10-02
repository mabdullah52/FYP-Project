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
 [4] Slice-level malignancy risk (CNN backbone → Bi-LSTM / Transformer)                   │
                                 ▼                                                        │
 [5] Growth-region forecasting                                                            │
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
 [10] Doctor web app                      [11] Patient app
      radiologist decision support             with agentic AI assistant
```

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
See [ARCHITECTURE.md](ARCHITECTURE.md).

### 7. Internal validation
Held-out LUNA16 / LIDC-IDRI patients. Use it for model selection and threshold tuning.

### 8. External testing on LUNA25
- Freeze the models and thresholds first, then run once on LUNA25.
- Report the same metrics as internal validation (AUC-ROC, sensitivity, specificity, accuracy; Dice/IoU only where masks exist).
- Comparing internal and external results shows how well the model generalises to a new population and scanners.

### 9. Backend API
Planned contract: [API.md](API.md).

### 10. Doctor web app
Radiologist-facing: upload a scan, browse slices, see confidence-coloured nodules, open Grad-CAM++ heatmaps and the growth forecast. Code: `apps/doctor-web-app/`.

### 11. Patient app
Patient-facing app with an agentic AI assistant. Code: `apps/patient-app/`. Features to be defined. ❓ Also decide which results patients can see and whether a doctor reviews them first.

## Dataset roles at a glance

| Dataset | Train | Validate | External test |
|---|:-:|:-:|:-:|
| LUNA16 | ✔ | ✔ (held-out patients) | |
| LIDC-IDRI | ✔ | ✔ (held-out patients) | |
| LUNA25 | | | ✔ |
