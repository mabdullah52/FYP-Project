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

- detects and segments lung nodules in CT scans,
- predicts a **malignancy risk score for every slice**, not just one global label,
- forecasts the **next likely region of tumour growth**, and
- explains its predictions with **confidence-coloured overlays** and **Grad-CAM++ heatmaps**,

all presented to radiologists through a Flutter desktop dashboard backed by a Python AI engine.

## Planned System Architecture

```
┌──────────────────────────────┐   HTTP / REST (JSON)   ┌───────────────────────────────────┐
│  Radiologist Dashboard        │ ─────────────────────▶ │  Python AI Engine (FastAPI/Flask)  │
│  Flutter desktop app          │ ◀───────────────────── │                                   │
│  • Slice viewer               │   risk scores, boxes,  │  1. Preprocessing & slice extract  │
│  • Confidence overlays        │   heatmap images       │  2. Nodule segmentation (U-Net)    │
│  • Grad-CAM toggle            │                        │  3. Slice-level risk (CNN + RNN)   │
└──────────────────────────────┘                        │  4. Growth-region forecasting      │
                                                        │  5. Grad-CAM++ explanations        │
                                                        └───────────────────────────────────┘
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Modules (from the methodology)

| Module | Approach (planned) | Output |
|---|---|---|
| Preprocessing | Resample to ~1×1×1 mm, clip HU to [-1000, 400], normalise to [0, 1], axial slicing, augmentation incl. Random Pixel Swap | Normalised 2D/3D inputs |
| Nodule segmentation | Modified U-Net with residual or attention gates | Binary nodule mask per slice |
| Slice-level risk trend | CNN backbone (DenseNet121 / ResNet50) → Bi-LSTM or Transformer encoder | Risk score 0–1 per slice + contributing features |
| Growth forecasting | Centroid + boundary-gradient analysis of segmented nodule | Probabilistic growth heatmap |
| Interpretability | Confidence colours + Grad-CAM++ | Green < 30%, Yellow 30–70%, Red > 70%; pixel heatmaps |

### Confidence colour scheme

| Colour | Malignancy probability | Meaning |
|---|---|---|
| 🟢 Green | < 30% | Low risk (likely benign) |
| 🟡 Yellow | 30–70% | Uncertain, needs manual review |
| 🔴 Red | > 70% | High risk (likely malignant) |

## Datasets

The CT data is **not stored in this repository** (too large, and redistribution is governed by each dataset's terms). See [docs/DATASET.md](docs/DATASET.md) for download and placement instructions.

## Evaluation Plan

| Task | Metrics |
|---|---|
| Segmentation | Dice Similarity Coefficient, IoU |
| Classification | Accuracy, Sensitivity, Specificity, AUC-ROC |
| Risk trend | MSE against slice-level expert annotations (where available) |
| Interpretability | Qualitative comparison of Grad-CAM maps with radiologist annotations; ablation of the sequential module |

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
│   ├── doctor-dashboard/     # Flutter radiologist dashboard
│   └── patient-app/          # Patient app (scope to be confirmed)
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
- **Frontend:** Flutter (Dart), Provider or Riverpod for state
- **Hardware:** NVIDIA GPU workstation

## Disclaimer

This is an academic research project. It is **not a medical device** and must not be used for clinical diagnosis.

## Acknowledgements

Supervisor: Rabia Masood, Bahria University Lahore Campus. The datasets are credited in [docs/DATASET.md](docs/DATASET.md).
