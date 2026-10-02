# Datasets

No medical image data is committed to this repository. Download each dataset yourself, accept its terms of use, and place it under `data/` as shown below. `data/` is git-ignored.

| Dataset | Role | Status |
|---|---|---|
| LUNA16 | Training + validation | Downloaded on university PC |
| LIDC-IDRI | Training + validation | To download |
| LUNA25 | External test only | To download |

See [PIPELINE.md](PIPELINE.md) for how they are combined.

## LUNA16 (training)

- **What:** 888 CT scans selected from LIDC-IDRI, with nodule annotations.
- **Format:** `.mhd` header + `.raw` volume per scan, split into `subset0` … `subset9`, plus CSV files (`annotations.csv`, `candidates*.csv`).
- **Source:** LUNA16 Grand Challenge: https://luna16.grand-challenge.org/
- **Cite:** the LUNA16 and LIDC-IDRI papers.

## LIDC-IDRI (training)

- **What:** 1,018 thoracic CT cases. Each has nodule outlines from up to four radiologists, plus malignancy ratings on a 1–5 scale.
- **Format:** DICOM series with XML annotation files.
- **Source:** The Cancer Imaging Archive (TCIA): https://www.cancerimagingarchive.net/collection/lidc-idri/ (download with the NBIA Data Retriever).
- **Tip:** the `pylidc` Python library reads the XML annotations and builds consensus masks.
- **Important:** LUNA16 is a subset of LIDC-IDRI. De-duplicate by `SeriesInstanceUID` before splitting (see [PIPELINE.md](PIPELINE.md#1-data-harmonisation)).

## LUNA25 (external test only)

- **Role:** unseen external test set. **Never used for training or tuning.**
- **What:** 4,069 low-dose chest CT scans from 2,120 patients in the US National Lung Screening Trial (NLST). Annotations cover 555 malignant and 5,608 benign nodules. The task is nodule malignancy risk estimation.
- **Source:** LUNA25 Grand Challenge: https://luna25.grand-challenge.org/
  - Images: https://zenodo.org/records/14223624 (≈221 GB in split zip archives, CC BY 4.0)
  - Annotations: https://zenodo.org/records/14673658 (`LUNA25_Public_Training_Development_Data.csv`, CC BY-NC 4.0, non-commercial)
- **Note:** after downloading, record the image file format and the CSV columns here.

## Expected layout

```
data/
├── luna16/
│   ├── subset0/ … subset9/      # *.mhd + *.raw
│   ├── annotations.csv
│   └── candidates*.csv
├── lidc-idri/
│   └── LIDC-IDRI-XXXX/…         # DICOM series + XML
└── luna25/
    ├── images/
    └── LUNA25_Public_Training_Development_Data.csv
```

## Other datasets named in the methodology

| Dataset | Status |
|---|---|
| IQ-OTH/NCCD | Not part of the current pipeline. Confirm whether to keep it |
| Longitudinal dataset for growth forecasting | Not yet selected |

## Preprocessing

See `configs/config.example.yaml`: resample to ~1 mm isotropic, clip HU to [-1000, 400], normalise to [0, 1], extract axial slices. The same steps apply to every dataset.

## Privacy

These are public, de-identified research datasets. Never commit patient data, including derived arrays (`.npy`, `.h5`) or screenshots that show identifiers.
