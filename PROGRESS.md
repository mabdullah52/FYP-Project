# Project Progress

Scope tiers (see [docs/PIPELINE.md](docs/PIPELINE.md#scope-what-is-realistic)): **core** = committed · **ext** = extension if time allows · **stretch** = future work unless completed

Legend: ✅ done · 🟡 partly done · ⬜ not started · 🏫 **NEEDS UNIVERSITY PC** · ❓ decision needed

_Last updated: 2026-10-02_

## Research & documentation

| Item | Status | Location |
|---|---|---|
| Proposal | ✅ | `docs/proposal/` |
| Proposal presentation | ✅ | `docs/proposal/` |
| Chapter 1, Chapter 2 | ✅ (drafts) | `docs/thesis/` |
| Methodology | ✅ | `docs/thesis/Methodology.docx` |
| Research gap analysis | ✅ | `docs/thesis/` |
| Literature papers & summaries | ✅ | `literature/` |
| README, pipeline, architecture, dataset, API docs | ✅ | root, `docs/` |
| Remaining report chapters (implementation, results, conclusion) | ⬜ | `docs/thesis/` |

## Repository setup

| Item | Status |
|---|---|
| Folder structure | ✅ |
| `.gitignore` (data, weights, secrets) | ✅ |
| `requirements.txt` (unpinned) | 🟡 exact versions 🏫 |
| `configs/config.example.yaml`, `.env.example` | ✅ (training hyperparameters 🏫) |
| Literature PDFs: check redistribution rights | ⬜ |

## Implementation

| Module | Status | Notes |
|---|---|---|
| LUNA16 downloaded | 🏫 | On university PC |
| LIDC-IDRI downloaded | ⬜ | Training data |
| LUNA25 downloaded | ⬜ | External test only (~221 GB) |
| De-duplication LUNA16 ⊂ LIDC-IDRI + patient-level split | ⬜ | `src/preprocessing/` |
| Preprocessing code | 🏫 | `src/preprocessing/` |
| Nodule segmentation (U-Net) | 🏫 | `src/segmentation/` |
| Nodule malignancy classifier (core) | 🏫 | `src/risk_model/` |
| Slice-sequence model (ext) | ⬜ | `src/risk_model/`, ablation vs. classifier |
| Growth forecasting (stretch) | ⬜ | `src/growth_forecast/`; needs longitudinal data |
| Grad-CAM++ explainability | 🏫 / ⬜ | `src/explainability/` |
| Training notebooks | 🏫 | `notebooks/` |
| Trained weights | 🏫 | external link in `models/README.md` |
| Internal validation results | 🏫 | `results/`, only real numbers |
| External test on LUNA25 | ⬜ | After models are frozen |
| REST API | ⬜ | `backend/`, contract in `docs/API.md` |
| Doctor web app (core) | ⬜ / 🏫 | `apps/doctor-web-app/` |
| Patient app with AI assistant (stretch) | ⬜ | `apps/patient-app/`, features TBC |
| Tests | ⬜ | `tests/` |

## Open decisions ❓

- Framework: PyTorch or TensorFlow
- Keep IQ-OTH/NCCD from the methodology, or drop it?
- Rule for turning LIDC-IDRI malignancy ratings (1–5) into benign/malignant labels
- Growth-forecast dataset
- Patient app: framework, agent features, which results patients see
- Project title: the proposal title ("…Handling Incomplete and Noisy Clinical Data…") vs. the methodology's focus (interpretable slice-level risk + growth forecasting)
