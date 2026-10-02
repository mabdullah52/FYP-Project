# Robust Lung Cancer Prediction: Handling Incomplete and Noisy Clinical Data Using Deep Learning

Final Year Project, BS Computer Science, Department of Computer Sciences, Bahria University Lahore Campus.

| | |
|---|---|
| **Team** | Abdullah Asim, Hussnain Khalid |
| **Supervisor** | Rabia Masood |
| **Status** | Research and design phase complete; implementation in progress (see [PROGRESS.md](PROGRESS.md)) |

> **Honesty note.** This repository currently contains the project proposal, methodology, literature review material and a planned code structure. Model code, trained weights and experimental results are **not yet in this repository**. No performance numbers are reported here until they come from real experiments.

---

## Overview

Radiologists review hundreds of CT slices per scan, and most existing AI tools give a single "cancer / no cancer" answer with no explanation. This project proposes an **interpretable Computer-Aided Diagnosis (CAD) system** that:

- segments lung nodules in CT scans,
- estimates **nodule malignancy risk**, with a slice-by-slice risk view as an extension,
- explains its predictions with **confidence-coloured overlays** and **Grad-CAM++ heatmaps**, and
- is tested on an **external dataset (LUNA25)** that it never saw during training,

served by a Python AI engine to a **doctor web app** for radiologists.

**Stretch goals** (only if time and data allow; see [Scope](#scope-core-vs-stretch)): forecasting the next likely region of tumour growth, and a patient app with an AI assistant.

## Planned System Architecture

```
┌──────────────────────────────┐   HTTP / REST (JSON)   ┌───────────────────────────────────┐
│  Doctor web app               │ ─────────────────────▶ │  Python AI Engine (FastAPI/Flask)  │
│  (radiologist dashboard)      │ ◀───────────────────── │                                   │
│  • Slice viewer               │   risk scores, boxes,  │  1. Preprocessing & slice extract  │
│  • Confidence overlays        │   heatmap images       │  2. Nodule segmentation (U-Net)    │
│  • Grad-CAM toggle            │                        │  3. Slice-level risk (CNN + RNN)   │
└──────────────────────────────┘                        │  4. Growth forecast (stretch)      │
                                                        │  5. Grad-CAM++ explanations        │
                                                        └───────────────────────────────────┘
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Modules (from the methodology)

The main model is a **nodule malignancy classifier** on nodule-centred CT patches (backbone such as DenseNet121 / ResNet50, 2D/2.5D/3D to be chosen).

| Module | Approach (planned) | Output |
|---|---|---|
| Preprocessing | Resample to ~1×1×1 mm, clip HU to [-1000, 400], normalise to [0, 1], axial slicing, augmentation incl. Random Pixel Swap | Normalised 2D/3D inputs |
| Nodule segmentation | Modified U-Net with residual or attention gates | Binary nodule mask per slice |
| Slice-level risk trend (extension) | CNN backbone (DenseNet121 / ResNet50) → Bi-LSTM or Transformer encoder | Risk score 0–1 per slice + contributing features |
| Growth forecasting (stretch) | Centroid + boundary-gradient analysis of segmented nodule | Probabilistic growth heatmap |
| Interpretability | Confidence colours + Grad-CAM++ | Green < 30%, Yellow 30–70%, Red > 70%; pixel heatmaps |

### Confidence colour scheme

| Colour | Malignancy probability | Meaning |
|---|---|---|
| 🟢 Green | < 30% | Low risk (likely benign) |
| 🟡 Yellow | 30–70% | Uncertain, needs manual review |
| 🔴 Red | > 70% | High risk (likely malignant) |

## Pipeline

**Train on LUNA16 + LIDC-IDRI → freeze the models → test on LUNA25 (external, unseen) → serve to the doctor web app.**

```
LUNA16 + LIDC-IDRI ─▶ de-duplicate + patient-level split ─▶ preprocess
        ─▶ U-Net segmentation + nodule malignancy classifier ─▶ Grad-CAM++ / confidence colours
        ─▶ internal validation ─▶ freeze weights + thresholds
                                         │
LUNA25 ─▶ same preprocessing ────────────┴─▶ external test (AUC-ROC, sensitivity, specificity)
                                         │
                                  Backend REST API
                                         │
                                   Doctor web app
                    (stretch: patient app · growth-region forecast)
```

Full stage-by-stage plan: [docs/PIPELINE.md](docs/PIPELINE.md)

## Scope: core vs stretch

| Tier | Components | Why |
|---|---|---|
| ✅ **Core** (committed) | Data de-duplication and patient split · preprocessing · U-Net segmentation · nodule malignancy classifier · confidence colours + Grad-CAM++ · internal validation · LUNA25 external test · REST API · doctor web app | The public data has labels for all of these, and they fit one GPU workstation |
| 🟡 **Extension** (if time allows) | Slice-sequence risk model (CNN → Bi-LSTM / Transformer) compared against the single-patch model · automatic nodule detection on whole scans | Feasible, but adds training time. There are no per-slice malignancy labels, so it is evaluated at nodule level |
| 🔵 **Stretch / future work** | Growth-region forecasting · patient app with an AI assistant · official LUNA25 leaderboard submission | Growth forecasting needs follow-up scans of the same nodule. LUNA16/LIDC-IDRI have only one scan per patient, and LUNA25 (which has repeat scans) is kept aside as the unseen test set. The patient app is a separate product with extra safety requirements |

Details: [docs/PIPELINE.md](docs/PIPELINE.md#scope-what-is-realistic)

## Datasets

| Dataset | Role |
|---|---|
| LUNA16 | Training and validation |
| LIDC-IDRI | Training and validation (overlaps LUNA16, de-duplicated) |
| LUNA25 | External test only (never used for training or tuning) |

The CT data is **not stored in this repository** (too large, and redistribution is governed by each dataset's terms). See [docs/DATASET.md](docs/DATASET.md) for download and placement instructions.

## Evaluation Plan

| Task | Metrics |
|---|---|
| Segmentation | Dice Similarity Coefficient, IoU |
| Classification | Accuracy, Sensitivity, Specificity, AUC-ROC |
| Slice-sequence model (extension) | Nodule-level AUC-ROC compared with the single-patch model (ablation) |
| Interpretability | Qualitative comparison of Grad-CAM maps with radiologist annotations; ablation of the sequential module |
| External generalisation | Nodule-level malignancy AUC-ROC, sensitivity, specificity on LUNA25 with the frozen model |

## Results

**Not yet available.** Results will be added to [`results/`](results/) only after experiments are run and verified.

## Repository Structure

```
.
├── README.md
├── PROGRESS.md               # What is done / what remains
├── requirements.txt          # Python dependencies (versions to be pinned)
├── configs/                  # Example configuration files
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PIPELINE.md           # Train on LUNA16 + LIDC-IDRI, external test on LUNA25, apps
│   ├── DATASET.md
│   ├── API.md                # Planned REST API contract
│   ├── proposal/             # Proposal document and presentation
│   ├── thesis/               # Report chapters, methodology, research gap analysis
│   ├── drafts/               # Working drafts
│   └── literature-matrices/  # Paper comparison spreadsheets
├── literature/
│   ├── journals/             # Reference papers grouped by research theme
│   └── summaries/            # Paper summaries
├── src/                      # AI engine source (preprocessing, models, explainability)
├── backend/                  # REST API server
├── apps/
│   ├── doctor-web-app/       # Radiologist decision-support web app
│   └── patient-app/          # Stretch goal: patient app with AI assistant
├── notebooks/                # Experiments
├── data/                     # Local datasets (git-ignored)
├── models/                   # Trained weights (git-ignored)
├── results/                  # Verified metrics and figures
├── scripts/                  # Helper scripts
└── tests/
```

## Setup

> The AI engine code is not yet committed. The steps below describe the intended environment.

```bash
git clone https://github.com/mabdullah52/FYP-Project.git
cd FYP-Project

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml   # then edit paths
cp .env.example .env
```

Then download the dataset as described in [docs/DATASET.md](docs/DATASET.md).

## Tech Stack (planned)

- **Language:** Python
- **Deep learning:** PyTorch or TensorFlow/Keras (final choice to be confirmed)
- **Medical imaging:** SimpleITK, OpenCV, NumPy
- **API:** FastAPI or Flask
- **Doctor web app:** Flutter (Dart) per the methodology, Provider or Riverpod for state (to be confirmed)
- **Patient app (stretch):** framework and AI-agent stack to be decided
- **Hardware:** NVIDIA GPU workstation

## Disclaimer

This is an academic research project. It is **not a medical device** and must not be used for clinical diagnosis.

## Acknowledgements

Supervisor: Rabia Masood, Bahria University Lahore Campus. The datasets are credited in [docs/DATASET.md](docs/DATASET.md).
