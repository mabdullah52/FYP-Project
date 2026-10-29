# Project Progress

_Last updated: 2026-10-30_

Legend: ✅ done · 🟡 in progress · ⬜ not started

## Experiments (all run on the university GPU, logs in `logs/`)

| Item | Status | Result | Where |
|---|---|---|---|
| LUNA16 preprocessing (888 scans) | ✅ | | `preprocess.py`, `logs/3_preprocess.txt` |
| MSGF-Net training + model selection on subset 8 | ✅ | | `train.py`, `work/runs/msgf_f9` |
| Test on subset 9 with TTA + MC dropout + calibration | ✅ | CPM 0.903 (0.843–0.956) | `work/results_table.csv` |
| Ablation study (5 variants) | ✅ | no significant differences | `work/significance_table.csv` |
| Comparison with 3D ResNet-18, 2.5D ResNet-18, plain 3D CNN | ✅ | MSGF-Net > 3D ResNet-18 (p = 0.034); ties plain 3D CNN | `work/baseline_table.csv` |
| Own candidate finder + end-to-end evaluation | ✅ | recall 92.4%, CPM 0.867 | `eval_end_to_end.py`, `logs/17_end_to_end.txt` |
| Speed-up (GPU finder, two-stage scoring) | ✅ | 267 s → 72 s per scan, CPM 0.864 | `logs/19_speed_check.txt` |
| Uncertainty triage analysis | ✅ | 5% referred catches all errors | `logs/20_triage.txt` |
| External test on LUNA25 (no retraining) | ✅ | mean sensitivity 0.754 | `work/luna25/external` |
| LUNA25 cancer-risk model | ✅ | AUC 0.900 vs size 0.856 | `work/luna25_final` |
| Growth / prior-scan model | ✅ | | `work/luna25_temporal` |
| 5-fold patient-level cross-validation | ✅ | stacked 0.932 vs single 0.904 (p < 0.001) | `work/luna25_cv` |
| Federated averaging (4 simulated hospitals) | ✅ | 0.808 vs 0.665 alone, 0.880 pooled | `work/federated_table.csv` |

## Software

| Item | Status | Where |
|---|---|---|
| AI server (FastAPI), doctor accounts, studies, PDF-style reports, Grad-CAM, Brock risk | ✅ | `doctor_app/` |
| Flutter doctor app (web + Android + Windows) | ✅ | `doctor_flutter/` |
| Patient web app with assistant, reminders, symptom diary, doctor requests, code recovery | ✅ | `patient_app/` |
| Flutter patient app (Android APK) | ✅ | `lungcare_mobile/` |
| Online deployment (Hugging Face Space, built automatically by GitHub Actions) | 🟡 | `deploy/huggingface/`, `.github/workflows/` |

## Writing

| Item | Status | Where |
|---|---|---|
| Proposal, Chapters 1–2, methodology, research gap analysis, literature review | ✅ | `docs/` |
| Paper 1: MSGF-Net (LUNA16 detection, comparison, external test, federated) | ✅ v4 | `paper/` |
| Paper 2: LUNA25 cancer risk and growth | ✅ draft | |
| Progress report | ✅ | |
| Final FYP report (thesis) | 🟡 | |
