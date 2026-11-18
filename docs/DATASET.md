# Datasets

No medical image data is committed to this repository. Download each dataset yourself from the official source, keep to its licence, and point `config.py` (LUNA16) and `work/luna25_path.txt` (LUNA25) at the folders.

| | LUNA16 | LUNA25 |
|---|---|---|
| **Origin** | LIDC-IDRI public thoracic CT collection | US National Lung Screening Trial (NLST); participants enrolled 2002–2004 at 33 US centres |
| **Provider** | LUNA16 challenge, https://luna16.grand-challenge.org | LUNA25 challenge, https://luna25.grand-challenge.org |
| **Download** | Zenodo records [3723295](https://zenodo.org/records/3723295) (subsets 0–6, CSVs) and [4121926](https://zenodo.org/records/4121926) (subsets 7–9) | Images: Zenodo [14223624](https://zenodo.org/records/14223624); annotations: Zenodo [14673658](https://zenodo.org/records/14673658) (DOI 10.5281/zenodo.14673658) |
| **Size** | 888 CT scans, 1,186 nodules, 754,975 candidates; 66.7 GB | 4,069 low-dose CT scans of 2,120 participants; 6,163 nodules (555 malignant, 5,608 benign); 221.1 GB |
| **Format** | MetaImage `.mhd` + `.raw` in `subset0`–`subset9`; CSV files | MetaImage `.mha` (split zip archives); one CSV file |
| **Features** | `annotations.csv`: seriesuid, world x/y/z, diameter (mm). `candidates_V2.csv`: seriesuid, x/y/z, class (nodule/non-nodule). Lung masks. | `LUNA25_Public_Training_Development_Data.csv`: series UID, world x/y/z, malignancy label, participant ID, lesion ID linking screening rounds, age, sex, study date |
| **Reference standard** | Nodules ≥ 3 mm accepted by ≥ 3 of 4 radiologists; slice thickness ≤ 2.5 mm | Malignant or benign label per nodule; up to three yearly screening rounds |
| **Licence** | CC BY 4.0 (LIDC-IDRI images: CC BY 3.0) | Images CC BY 4.0; annotations CC BY-NC 4.0 (non-commercial) |
| **Permission** | None needed beyond the licence; cite Setio et al. 2017 and Armato et al. 2011 | None needed beyond the licence; non-commercial use only; cite the LUNA25 challenge and NLST (TCIA DOI 10.7937/TCIA.HMQ8-J677) |
| **Speciality** | Standard benchmark with fixed subsets and the official CPM evaluation | Largest public set of screen-detected nodules with malignancy labels and repeat scans of the same lesion |
| **Use in this project** | Detection: 5-fold cross-validation (`cv5.py`) and development fold (train 0–7, validation 8, test 9) | Cancer risk and growth (5 folds by participant); external test of the detector |

## Expected layout

```
LUNA16/
├── subset0/ … subset9/        # *.mhd + *.raw
├── annotations.csv
├── annotations_excluded.csv
└── candidates_V2.csv
LUNA25/
├── images/                    # *.mha
└── LUNA25_Public_Training_Development_Data.csv
```

## Privacy

Both are public, de-identified research datasets. Never commit patient data, including derived arrays (`.npy`, `.h5`) or screenshots that show identifiers.
