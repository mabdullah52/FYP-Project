# LungScan AI: Robust Lung Cancer Prediction

Official FYP title: *Robust Lung Cancer Prediction: Handling Incomplete and Noisy Clinical Data Using Deep Learning.*

Final Year Project, BS Computer Science, Bahria University Lahore Campus
**Abdullah Asim** (03-134231-005) · **Hussnain Khalid** (03-134222-035) · Supervisor: **Rabia Masood**

LungScan AI finds pulmonary nodules on chest CT, estimates their size, calibrated probability and
uncertainty, estimates cancer risk, and uses last year's scan when one exists. It ships with a
Flutter doctor app, a Flutter (Android) and web patient app, and an online demo.

> Research prototype. Decision support for radiologists only. It is not a medical device and does not diagnose.

## Results at a glance

| Task | Data | Result |
|---|---|---|
| Nodule detection (FP reduction), MSGF-Net, 3.35 M parameters | LUNA16, **5-fold patient-level cross-validation, all 888 scans** | **CPM 0.877** (95% CI 0.856–0.895); folds 0.861–0.889 |
| Same model, development fold | LUNA16, test subset 9 (88 scans, 105 nodules) | CPM 0.903 (95% CI 0.843–0.956) |
| Same protocol, standard networks (development fold) | subset 9 | plain 3D CNN 0.908 (tie, p = 0.68) · 2.5D ResNet-18 0.861 · 3D ResNet-18 0.856 (MSGF-Net better, p = 0.034) |
| Nodule size | 5-fold, 888 scans | MAE 0.81 mm (r = 0.981) |
| Calibration / uncertainty | 5-fold, 754,975 candidates | ECE 0.0307 → 0.0002 · uncertainty separates errors with AUROC 0.915 |
| Complete system from raw CT (own candidate finder) | same | CPM 0.864, 72 s per scan on GPU |
| External test without retraining | LUNA25 (6,163 nodules, NLST low-dose) | mean sensitivity 0.754 (0.907 at 8 FP/scan) |
| Cancer risk from one scan | LUNA25, 424 unseen patients | AUC 0.900 vs 0.856 for size alone (p = 0.001); 5-fold CV 0.875 vs 0.830 |
| Cancer risk with last year's scan | LUNA25, 5-fold CV, 2,401 nodules | AUC 0.932 vs 0.904 single scan (p < 0.001) |
| Federated training (4 simulated hospitals) | LUNA16 | CPM 0.808 vs 0.665 one hospital alone, 0.880 pooled |

All confidence intervals come from patient- or scan-level bootstrap; all comparisons are paired.
Full numbers are in `work/*.csv`, `work/*/…metrics.json` and the run logs in `logs/`.

## Papers

* `paper/MSGF-Net_paper.pdf`: *Calibrated, uncertainty-aware pulmonary nodule false-positive reduction with multi-window scale-gated 3D fusion and size estimation* (LaTeX source in `paper/`)
* `paper2/Growth-Aware_Malignancy_LUNA25.pdf`: *Does last year's scan help? Growth-aware malignancy prediction for screen-detected lung nodules on LUNA25* (LaTeX source in `paper2/`)

Single-column versions for journal submission: `paper/Paper1_MSGF-Net_single_column.pdf`, `paper2/Paper2_Growth_Aware_Malignancy_LUNA25_single_column.pdf`.

Corresponding author of both papers: Rabia Masood (rabiamasood.bulc@bahria.edu.pk), Department of Computer Sciences, Bahria University Lahore Campus.

## FYP final report and defence

* `docs/final/LungScanAI_FYP_Final_Report.pdf` (and `.docx`): final report (thesis) in the Bahria University template
* `docs/final/LungScanAI_Chapter6_SDGs.docx`: Sustainable Development Goals chapter
* `docs/final/LungScanAI_Defense_Presentation.pptx`: defence presentation

## Documentation

| Document | What it covers |
|---|---|
| [Progress](PROGRESS.md) | What is done |
| [Methodology](METHODOLOGY.md) · [Pipeline guide](docs/PIPELINE_GUIDE.md) | How the system and experiments work |
| [Trained models](docs/MODELS.md) | Which model files are included and how to publish the rest |
| [Research design](docs/RESEARCH.md) · [Datasets](docs/DATASET.md) · [References](docs/REFERENCES.md) | Original research plan, data, literature (32 papers) |
| `docs/thesis/`, `docs/proposal/`, `docs/literature-matrices/`, `literature/summaries/` | Thesis chapters, proposal, literature matrices and summaries |

## Repository layout

| Path | What it is |
|---|---|
| `preprocess.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `metrics.py` | LUNA16 pipeline: patches, MSGF-Net, training, FROC/CPM evaluation |
| `baselines.py`, `compare_baselines.py` | 3D ResNet-18, 2.5D ResNet-18 and plain 3D CNN baselines |
| `candidate_gen.py`, `infer.py`, `predictor.py`, `explain.py` | Full-scan inference from raw CT, Grad-CAM |
| `luna25_*.py`, `risk_model.py`, `brock.py` | LUNA25 external test, cancer-risk models, growth model, Brock score |
| `fed_train.py` | Federated averaging experiment |
| `stats_compare.py` | Paired bootstrap tests |
| `doctor_app/` | Python (FastAPI) server: AI, doctor accounts, studies, reports |
| `doctor_flutter/` | Flutter doctor app (web, Android, Windows) |
| `patient_app/`, `lungcare_mobile/` | Patient server + web app, and the Flutter Android app |
| `deploy/huggingface/` | Online deployment (one Docker container, port 7860) |
| `work/` | Result tables, metrics, training histories, the final MSGF-Net model (`work/runs/msgf_f9/best.pt`) |
| `logs/` | Logs of every experiment |
| `*.bat` | Numbered Windows steps used on the university GPU PC (see `START_HERE.txt`) |
| `docs/PIPELINE_GUIDE.md`, `METHODOLOGY.md` | Detailed pipeline and methodology notes |

## Data (not included)

* **LUNA16**: https://luna16.grand-challenge.org/ (888 CT scans, 66.7 GB, CC BY 4.0; Zenodo records 3723295 and 4121926)
* **LUNA25**: https://luna25.grand-challenge.org/ (4,069 NLST low-dose CT scans, 221.1 GB, images CC BY 4.0 on Zenodo 14223624; annotations CC BY-NC 4.0, DOI 10.5281/zenodo.14673658)

See [docs/DATASET.md](docs/DATASET.md) for size, format, features and licences.

Splits: LUNA16 5-fold cross-validation over the official subsets (`cv5.py`; test 0+1, 2+3, 4+5, 6+7, 8+9) and one development fold (train 0–7, validation 8, test 9). LUNA25: patient-level 5-fold split.

## Run it

```bash
python -m venv lungenv && source lungenv/bin/activate      # Windows: lungenv\Scripts\activate
pip install -r requirements.txt
python doctor_app/server.py          # doctor workstation  -> http://127.0.0.1:8000
python patient_app/server.py         # patient web app     -> http://127.0.0.1:8100
```

The trained detector `work/runs/msgf_f9/best.pt` and the LUNA25 cancer-risk ensemble (`work/runs/l25_scratch*/best.pt`) are included; see [docs/MODELS.md](docs/MODELS.md) for the other trained models.
To reproduce training, follow `START_HERE.txt` (Windows) or `docs/PIPELINE_GUIDE.md`.

Flutter apps: `doctor_flutter/BUILD_DOCTOR_FLUTTER.bat` (web + Android) and `lungcare_mobile/SETUP_AND_BUILD_ANDROID.bat`.

## Online demo

**Live:** doctor app https://lungscan-ai.thankfulocean-bfa8eccd.koreacentral.azurecontainerapps.io/doctor/ · patient app https://lungscan-ai.thankfulocean-bfa8eccd.koreacentral.azurecontainerapps.io/patient/ · Android app: see Releases (LungCare.apk).

Every push to `main` runs `.github/workflows/deploy.yml`, which
1. builds the Flutter doctor web app and packs the whole system into one Docker image, `ghcr.io/mabdullah52/lungscan-ai:latest` (port 7860: `/` and `/doctor/` for doctors, `/patient/` for patients), and
2. builds the LungCare Android APK and publishes it under **Releases**.

The image runs on **Azure Container Apps** (Azure for Students). Set the container's environment variables `ADMIN_USER` and `ADMIN_PASSWORD` (the doctor administrator account),
and the repository variable `LUNGCARE_SERVER` (`https://<app>.azurecontainerapps.io/patient`) so the APK has the address built in.
The image runs on CPU, so analysing a scan takes several minutes, and its data resets when the container restarts.
`deploy/huggingface/` can also deploy the same folder to a Hugging Face Space (requires a PRO account for Docker Spaces).

## Ethics and privacy

No patient data is stored in this repository. The apps keep data locally (`app_data/`, git-ignored).
Doctor accounts use salted password hashes; the patient portal uses per-patient access codes.
