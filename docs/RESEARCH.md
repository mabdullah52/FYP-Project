# Research Design

This file sets out the research questions, hypotheses, experiments and paper outline. **All results tables are empty on purpose.** They will be filled only with numbers from experiments run in this repository.

## 1. Problem and research gap

Our literature review ([REFERENCES.md](REFERENCES.md); full analysis in `docs/thesis/Research gap analysis.docx`) identified these gaps:

1. **Black-box predictions.** Most lung-nodule malignancy models output a single label without showing which image evidence drove it, and this limits clinicians' trust.
2. **Weak generalisation evidence.** Many studies train and test on splits of the same dataset. Performance often falls on data from different scanners, protocols or populations.
3. **Single global decision.** Few systems show how risk varies across the axial slices of a nodule, or flag uncertain cases for review.
4. **Single time point.** Nodules are usually assessed at one time point, without modelling growth over time.

This project addresses gaps 1–3 directly. Gap 4 is a stretch goal (see [PIPELINE.md](PIPELINE.md#scope-what-is-realistic)).

## 2. Research questions

| ID | Question | Tier |
|---|---|---|
| **RQ1** | How well does a nodule malignancy model trained on LUNA16 / LIDC-IDRI generalise to an independent screening cohort (LUNA25)? | Core |
| **RQ2** | Do Grad-CAM++ explanations focus on the nodule itself, as defined by radiologist outlines? | Core |
| **RQ3** | Does a three-band confidence scheme (green / yellow / red) concentrate model errors in the "review" (yellow) band? | Core |
| **RQ4** | Does modelling the sequence of slices through a nodule (CNN → Bi-LSTM / Transformer) improve malignancy prediction over a single-patch CNN? | Extension |

## 3. Hypotheses

- **H1:** External AUC-ROC on LUNA25 will be lower than internal validation AUC because of domain shift and different labels. The size of this drop is itself a reported result.
- **H2:** Most of the Grad-CAM++ heatmap mass will fall inside, or close to, the radiologist nodule mask for correctly classified nodules.
- **H3:** Error rate in the yellow band will be significantly higher than in the green and red bands.
- **H4 (extension):** The slice-sequence model will outperform the single-patch model on nodule-level AUC-ROC.

## 4. Planned contributions

1. A leakage-free training setup for LUNA16 + LIDC-IDRI (series-level de-duplication, patient-level split), with **external validation on LUNA25**.
2. A **quantitative check of explanation quality**: Grad-CAM++ heatmaps compared against radiologist segmentations, not only shown as pictures.
3. A **confidence-band analysis** showing whether the green / yellow / red scheme separates reliable from unreliable predictions.
4. An **ablation** of slice-sequence versus single-patch modelling (extension).
5. An open-source pipeline plus a radiologist web dashboard that presents the outputs.

## 5. Experimental design

**Data.** Train and validate on de-duplicated LUNA16 + LIDC-IDRI, split by patient. Test once on the LUNA25 public set with frozen weights and thresholds. See [DATASET.md](DATASET.md).

**Labels.** LIDC-IDRI malignancy ratings (1–5) are mapped to benign / malignant using a rule fixed **before** training and reported in the paper (e.g. mean ≤ 2 benign, ≥ 4 malignant, 3 excluded).

**Statistics.** Report 95% confidence intervals (bootstrap) for all metrics. LUNA25 is imbalanced (555 malignant vs. 5,608 benign nodules), so report AUC-ROC, sensitivity at fixed specificity, and specificity at fixed sensitivity alongside accuracy.

### Experiment tables (to be filled with real results)

**E1: Segmentation (internal validation)**

| Model | Dice | IoU |
|---|---|---|
| U-Net (baseline) | — | — |
| Attention / residual U-Net | — | — |

**E2: Malignancy classification (RQ1)**

| Model | Internal AUC | Internal Sens / Spec | LUNA25 AUC | LUNA25 Sens / Spec |
|---|---|---|---|---|
| Single-patch CNN | — | — | — | — |
| Slice-sequence model (extension, RQ4) | — | — | — | — |

**E3: Explanation alignment (RQ2)**

| Model | Heatmap mass inside nodule mask | Pointing-game hit rate |
|---|---|---|
| Single-patch CNN + Grad-CAM++ | — | — |

**E4: Confidence bands (RQ3)**

| Band | Probability range | % of nodules | Error rate |
|---|---|---|---|
| Green | < 0.30 | — | — |
| Yellow | 0.30–0.70 | — | — |
| Red | > 0.70 | — | — |

## 6. Threats to validity

- **Label noise.** LIDC-IDRI malignancy is a subjective radiologist rating, not a pathology result.
- **Domain shift.** LUNA25 comes from a different screening trial, scanners and time period.
- **Dataset overlap.** LUNA16 ⊂ LIDC-IDRI. Handled by series-level de-duplication.
- **Thresholds.** The 0.30 / 0.70 band cut-offs come from the methodology, not from data. Sensitivity to other cut-offs will be reported.
- **Explanation methods.** Grad-CAM++ shows where the model looked, not why. It is not a causal explanation.

## 7. Paper outline (draft)

1. Introduction: clinical motivation, gaps, contributions
2. Related work: nodule malignancy prediction, explainability, external validation
3. Data: LUNA16, LIDC-IDRI, LUNA25; de-duplication; label mapping
4. Methods: preprocessing, segmentation, classifier, slice-sequence model, Grad-CAM++, confidence bands
5. Experiments: E1–E4
6. Results
7. Discussion and limitations
8. Conclusion and future work (growth forecasting, patient-facing tools)
