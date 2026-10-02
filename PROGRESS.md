# Project Progress

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
| README, architecture, dataset, API docs | ✅ | root, `docs/` |
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
| Dataset downloaded (LUNA16) | 🏫 | On university PC |
| Preprocessing code | 🏫 | `src/preprocessing/` |
| Nodule segmentation (U-Net) | 🏫 | `src/segmentation/` |
| Slice-level risk model | 🏫 | `src/risk_model/` |
| Growth forecasting | 🏫 / ⬜ | `src/growth_forecast/` |
| Grad-CAM++ explainability | 🏫 / ⬜ | `src/explainability/` |
| Training notebooks | 🏫 | `notebooks/` |
| Trained weights | 🏫 | external link in `models/README.md` |
| Results (Dice, IoU, AUC, …) | 🏫 | `results/`, only real numbers |
| REST API | ⬜ | `backend/`, contract in `docs/API.md` |
| Flutter radiologist dashboard | ⬜ / 🏫 | `apps/doctor-dashboard/` |
| Patient app | ❓ | Scope to confirm |
| Tests | ⬜ | `tests/` |

## Open decisions ❓

- Framework: PyTorch or TensorFlow
- Datasets: LUNA16 only, or also IQ-OTH/NCCD as in the methodology?
- Growth-forecast dataset
- Patient app in scope?
- Project title: the proposal title ("…Handling Incomplete and Noisy Clinical Data…") vs. the methodology's focus (interpretable slice-level risk + growth forecasting)
