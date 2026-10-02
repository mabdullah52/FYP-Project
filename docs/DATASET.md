# Datasets

No medical image data is committed to this repository. Download each dataset yourself, accept its terms of use, and place it under `data/` as shown below. `data/` is git-ignored.

## LUNA16 (in use)

- **What:** CT scans for lung nodule analysis, derived from LIDC-IDRI, with nodule annotations.
- **Format:** `.mhd` header + `.raw` volume per scan, split into `subset0` … `subset9`, plus CSV files (`annotations.csv`, `candidates*.csv`).
- **Source:** LUNA16 Grand Challenge — https://luna16.grand-challenge.org/
- **License / terms:** see the download page; cite the LUNA16 and LIDC-IDRI papers in the report.

Expected layout:

```
data/
└── luna16/
    ├── subset0/ … subset9/      # *.mhd + *.raw
    ├── annotations.csv
    ├── candidates*.csv
    └── seg-lungs-LUNA16/        # optional lung masks, if downloaded
```

## Datasets named in the methodology (status to be confirmed)

| Dataset | Role in methodology | Status |
|---|---|---|
| LIDC-IDRI (TCIA, 1,018 cases) | Nodule detection | TBC — LUNA16 is a curated subset of it |
| IQ-OTH/NCCD | Slice-level classification | TBC |
| Spatio-temporal / longitudinal dataset | Growth forecasting module | TBC — not yet selected |

## Preprocessing summary

See `configs/config.example.yaml`: resample to ~1 mm isotropic, clip HU to [-1000, 400], normalise to [0, 1], extract axial slices.

## Privacy

These are public, de-identified research datasets. Never commit patient data, including derived arrays (`.npy`, `.h5`) or screenshots that show identifiers.
