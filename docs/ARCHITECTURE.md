# System Architecture (planned)

Source: `docs/thesis/Methodology.docx`, Sections 3.1–3.7. This describes the **design**; implementation status is tracked in [PROGRESS.md](../PROGRESS.md).

## Client–server layout

| Layer | Technology | Responsibility |
|---|---|---|
| Client | Flutter desktop app (Provider or Riverpod) | Load scans, slice viewer, colour overlays, Grad-CAM toggle |
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
Flutter dashboard
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
| UI | `apps/doctor-dashboard/` |

## Open design decisions

- [ ] DL framework: PyTorch vs TensorFlow/Keras
- [ ] U-Net variant: attention vs residual
- [ ] Backbone: DenseNet121 vs ResNet50
- [ ] Sequence model: Bi-LSTM vs Transformer encoder
- [ ] API framework: FastAPI vs Flask
- [ ] Dataset for the growth forecasting module
- [ ] Patient app: in scope or not (the proposal describes a desktop-only deliverable)
