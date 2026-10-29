# Methodology (paper draft material)

**Working title:** *MSGF-Net: Uncertainty-Calibrated Multi-Window Scale-Gated Fusion for Pulmonary
Nodule False-Positive Reduction in CT*

## 1. Problem statement
Candidate detectors used for lung-cancer screening CT find almost every nodule, but they also
produce hundreds of false positives (vessels, pleura, scars) per scan. The FP-reduction stage
must (i) reject these at very low FP/scan, (ii) tell the clinician *how sure* it is, and (iii)
give the nodule size that drives screening management (Lung-RADS). Most published LUNA16
classifiers address only (i). Many of them also report accuracy on artificially balanced,
augmented-before-split data, which is not comparable to the challenge metric.

## 2. Contributions (write them like this, honestly and specifically)
1. **Multi-window HU encoding (MW):** each 3D patch becomes three channels (lung WL-600/WW1500,
   mediastinal WL40/WW400, wide [-1000, 400]). The network sees parenchymal texture and
   solid-tissue contrast at once, rather than a single min-max normalised volume.
2. **Scale-Gated Fusion (SGF)** between a 32 mm local stream and a 48 mm context stream.
   (a) The context descriptor *recalibrates* the channels of the local feature map before
   pooling. (b) A learned per-channel gate α mixes local and context descriptors. Unlike
   multi-level contextual 3D CNNs that average the output probabilities of separate networks
   (Dou et al., CUMedVis, the best FP-reduction system in the LUNA16 paper, CPM 0.908), SGF
   fuses at feature level inside one network. α is interpretable: report its correlation
   with nodule diameter (`alpha_vs_diameter_spearman`).
3. **Diameter-aware multi-task learning:** an auxiliary head regresses log-diameter on
   positives. Kendall et al. homoscedastic uncertainty weighting balances it against a focal
   classification loss. The model outputs size, which maps to a Lung-RADS size category.
4. **Calibrated, uncertainty-aware inference:** 8-flip TTA × 10 MC-dropout samples of the
   head only (encoders run once per view, so it is cheap). Platt scaling is fitted on the
   validation subset's full candidate list. The paper reports ECE before and after, and an
   **uncertainty-triage curve**: automatic performance when the k% most uncertain candidates
   are referred to a radiologist.
5. **Two-round hard-negative curriculum:** the round-1 model mines the 20 most confusing
   negatives per training scan from all ~600k training candidates, then round 2 fine-tunes on
   them. Combined with proximity-aware sampling (near-miss negatives 5–25 mm from nodules).
6. **End-to-end deployable CAD:** a training-free LoG candidate detector, MSGF-Net and a JSON
   report consumed by a clinician web app. Report the detector's recall.

Contributions 2, 4 and the clinical-deployment angle carry the novelty. MW, focal loss
and hard-negative mining are known ingredients, so present them as *components* validated by
ablation, not as inventions.

## 3. Data and protocol
* LUNA16: 888 CT scans (LIDC-IDRI, slice thickness ≤ 2.5 mm), 1,186 nodules ≥ 3 mm, candidates_V2.
  Take the exact numbers from `audit.json`.
* **Patient-level split by the official subsets:** train 0–7, validation 8, test 9. The code
  asserts that no scan is shared between splits. If time allows, repeat for folds 0, 3 and 6,
  and report mean ± std. Full 10-fold is the challenge standard, so say that you used a subset.
* Test = **all** candidates of the held-out subset (no sub-sampling).
* Metrics: CPM (mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan) with bootstrap 95% CI,
  FROC curve, candidate AUC/AP, ECE, diameter MAE and Pearson r, AUC of uncertainty for
  detecting errors, and the triage curve.

## 4. Implementation details (copy into the paper)
Patches 48³ at 1 mm (context) and 32³ centre crop (local). Each stream: stem conv + 6 residual
SE blocks (widths 24/48/96/96), 3D, BatchNorm, ReLU. Head: dropout 0.3 → FC64 → dropout → FC1.
Diameter head: FC → softplus + 1 mm. AdamW (lr 1e-3, wd 1e-4), 1-epoch warm-up + cosine, batch
64, 24k samples/epoch with 25% positives, 30 epochs, early stopping on validation AUC
(patience 8), mixed precision. Focal γ = 2, α = 0.5. Round 2: 10 epochs, lr 3e-4.
Augmentation: random flips (3 axes), 90° axial rotations, ±3 voxel shifts, HU scale 0.97–1.03 and
offset ±20. Parameters: 3.35 M. Report the GPU used and the training time from `history.csv`.

## 5. Experiments = tables and figures of the paper
**Table 1 (dataset):** from `audit.json`.

**Table 2 (main result, test subset 9):** MSGF-Net (round 2, TTA+MC, calibrated): sensitivity at
7 FP rates, CPM [95% CI], AUC. Put the LUNA16 FP-reduction track numbers from Setio et al. 2017
next to it (CUMedVis 0.908, JackFPR 0.872, DIAG_CONVNET 0.854) with this caveat: *those are
10-fold results on the V1 candidate set, so the comparison is indicative.* To be comparable,
state the fold protocol precisely.

**Table 3 (ablation, single-pass `_plain` evaluations):**

| Variant | Change | CPM |
|---|---|---|
| Full MSGF-Net (round 1) | none | |
| no context stream | `--no-context` | |
| concat fusion instead of SGF | `--fusion concat` | |
| single HU window | `--windows single` | |
| no diameter task | `--aux-weight 0` | |
| BCE instead of focal | `--loss bce` | |
| + round-2 hard negatives | `msgf_f9_r2` | |
| + TTA + MC dropout | `msgf_f9_r2 _full` | |

**Table 4 (clinical outputs):** ECE raw vs. calibrated, diameter MAE and r, uncertainty-AUC
for errors, % of candidates referred to reach sensitivity ≥ 0.95.

**Figures:** pipeline diagram, SGF block diagram, FROC (`froc.png`), reliability diagram,
triage curve, α vs. diameter scatter (from `predictions.csv`), 3–4 qualitative app snapshots
(TP / FP / flagged-uncertain).

## 6. Honest limitations section (reviewers reward this)
* The model detects nodules. It does not predict malignancy (LUNA16 has no pathology labels).
* The evaluation covers one or a few folds rather than the full 10-fold protocol, and uses a
  single dataset with no external validation.
* Lung-RADS categories are derived from size only (no solid/part-solid type, no growth).
* The candidate detector for new scans is classical and generates more candidates than the
  challenge's merged detectors.

## 7. Pitfalls found in the downloaded papers (don't repeat them)
* **Sensors 2022 (AlexNet, 97.42% accuracy):** compares 2D ImageNet-style CNNs and optimizers,
  reports accuracy/sensitivity/specificity only (no CPM or FROC), and frames LUNA16 as "lung
  cancer classification". At ~1:490 imbalance, accuracy on a balanced patch set does not tell
  you false positives per scan.
* **Sci. Rep. 2024 MIFNet:** calls LUNA16 nodule/non-nodule "benign/malignant". It balances
  classes by rotation, and its 6,484/1,622 train/test counts are an exact 80/20 split of the
  augmented pool, which suggests augmented copies of the same nodule may sit on both sides of the
  split. It also uses 2D 50×50 crops.
* **IJRTI 2022 EfficientNet:** same benign/malignant confusion, 2D 65×65 crops.
* **Sci. Rep. 2026 federated Transformer-U-Net:** segmentation (Dice), a different task. Cite
  it for Dice-Focal/imbalance handling and for privacy as future work.
* **Setio et al. 2017 (LUNA16):** *the* reference for protocol and metric. Follow it.
