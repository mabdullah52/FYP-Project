# REST API: Planned Contract

> **Status: DESIGN ONLY.** None of these endpoints are implemented yet. This contract comes from Methodology Section 3.7 (the server returns risk scores, bounding boxes and heatmap images as JSON). Update this file to match the real code once `backend/` exists.

Base URL (local): `http://127.0.0.1:8000`

## `GET /health`

Returns server and model status.

```json
{ "status": "ok", "models_loaded": true }
```

## `POST /predict`

Upload a CT scan and receive per-slice risk.

**Request:** `multipart/form-data`
- `file`: scan (`.mhd` + `.raw` as zip, or DICOM series as zip). Accepted formats TBC.

**Response (shape only, example values are placeholders):**

```json
{
  "scan_id": "string",
  "num_slices": 0,
  "slices": [
    {
      "index": 0,
      "risk_score": 0.0,
      "confidence_level": "green | yellow | red",
      "nodules": [
        { "bbox": [x_min, y_min, x_max, y_max], "risk_score": 0.0 }
      ]
    }
  ],
  "growth_heatmap_available": false
}
```

`confidence_level` mapping: `green` < 0.30 ≤ `yellow` ≤ 0.70 < `red`.

## `GET /explain/{scan_id}/{slice_index}`

Returns the Grad-CAM++ heatmap for one slice. The dashboard calls it when the "interpretability" button is pressed.

**Response:** `image/png`

## `GET /growth/{scan_id}`

Returns the growth-forecast heatmap.

**Response:** `image/png` (format TBC)

## Errors

| Code | Meaning |
|---|---|
| 400 | Unsupported or corrupt file |
| 404 | Unknown `scan_id` / slice |
| 500 | Inference failure |
