# Trained models

## In this repository

| File | Model | Used by |
|---|---|---|
| `work/runs/msgf_f9/best.pt` | MSGF-Net nodule detector (development fold: train subsets 0–7, validation 8, test 9) | Doctor app, `infer.py` |
| `work/runs/l25_scratch/best.pt` | LUNA25 cancer-risk 3D CNN, seed 0 | Doctor app (risk ensemble) |
| `work/runs/l25_scratch_s1/best.pt` | LUNA25 cancer-risk 3D CNN, seed 1 | Doctor app (risk ensemble) |
| `work/runs/l25_scratch_s2/best.pt` | LUNA25 cancer-risk 3D CNN, seed 2 | Doctor app (risk ensemble) |

Each file is about 13 MB. These are the models the doctor app loads, so the system runs from a fresh clone.

## Kept on the training PC

The models below were trained on the university GPU workstation. Their results (metrics, predictions summaries, training histories) are in `work/`, but the weight files are not in git because there are many of them:

| Run folder in `work/runs/` | What it is |
|---|---|
| `cv5_msgf_f1` … `cv5_msgf_f5` | MSGF-Net, 5-fold cross-validation (Paper 1, Chapter 5) |
| `base_cnn3d`, `base_r3d18`, `base_resnet25d` | Baseline architectures |
| `abl_*` | Ablation models |
| `fed_*` | Simulated federated training |
| `l25_cv*`, `l25_temporal*` | LUNA25 cross-validation and growth (TG-MSGF) models |

To publish them, run `39_PACK_MODELS_FOR_GITHUB.bat` on the training PC. It collects every `best.pt` with its `config.json` into `models_all.zip`. Upload that zip as an asset of a GitHub Release (Releases → Draft a new release); release assets can be up to 2 GB, while normal git files are limited to 100 MB.

Every model can also be retrained from the code: `36_FIVE_FOLD_CV.bat` (5 folds), `34_BASELINE_MODELS_OVERNIGHT.bat`, `6_EXPERIMENTS_OVERNIGHT.bat`, `18_FEDERATED_OVERNIGHT.bat`, `14_LUNA25_FINAL.bat` and `15_GROWTH_MODEL_OVERNIGHT.bat`.
