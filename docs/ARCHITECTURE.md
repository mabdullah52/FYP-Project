# System Architecture (planned)

Source: `docs/thesis/Methodology.docx`, Sections 3.1–3.7. For the full data → training → testing → apps workflow see [PIPELINE.md](PIPELINE.md). This describes the **design**; implementation status is tracked in [PROGRESS.md](../PROGRESS.md).

## Client–server layout

| Layer | Technology | Responsibility |
|---|---|---|
| Doctor web app | Flutter per methodology (TBC) | Load scans, slice viewer, colour overlays, Grad-CAM toggle, growth forecast |
| Patient app | TBC | Patient-facing app with an agentic AI assistant |
| Server | Python, FastAPI or Flask | Run models, return JSON (risk scores, boxes, heatmap images) |
| Models | PyTorch or TensorFlow/Keras | Segmentation, risk, growth forecasting, explanations |

## Pipeline

```
CT volume (.mhd/.raw or DICOM)
   │
   ▼
[1] Preprocessing
    resample ~1 mm³ · clip HU [-1000, 400] · normalise [0,1] · axial slices · augmentation (rotation, flip, Random Pixel Swap)
   │
   ▼
[2] Nodule segmentation: modified U-Net (attention / residual)
    → binary mask per slice; only segmented nodules go forward
   │
   ▼
[3] Slice-level risk trend: CNN backbone (DenseNet121 / ResNet50) → Bi-LSTM or Transformer encoder
    → malignancy risk 0–1 per slice + contributing features (e.g. spiculated edges, necrotic centre)
   │
   ├──▶ [4] Growth forecasting: 3D centroid + boundary gradient → probabilistic growth heatmap
   │
   ▼
[5] Interpretability
    confidence colours (green < 0.30 ≤ yellow ≤ 0.70 < red) · Grad-CAM++ pixel heatmaps
   │
   ▼
Backend API ─▶ Doctor web app · Patient app
```

## Source-code mapping (planned)

| Pipeline step | Folder |
|---|---|
| 1 | `src/preprocessing/` |
| 2 | `src/segmentation/` |
| 3 | `src/risk_model/` |
| 4 | `src/growth_forecast/` |
| 5 | `src/explainability/` |
| API | `backend/` |
| Doctor web app | `apps/doctor-web-app/` |
| Patient app | `apps/patient-app/` |

## Open design decisions

- [ ] DL framework: PyTorch vs TensorFlow/Keras
- [ ] U-Net variant: attention vs residual
- [ ] Backbone: DenseNet121 vs ResNet50
- [ ] Sequence model: Bi-LSTM vs Transformer encoder
- [ ] API framework: FastAPI vs Flask
- [ ] Dataset for the growth forecasting module
- [ ] Patient app: framework, agent features, and which results patients may see
- [ ] Label rule mapping LIDC-IDRI malignancy ratings (1–5) to benign/malignant
