"""Step 4 - full-scan inference: the function the doctor web app will call.

    python infer.py --ckpt work/runs/msgf_f9/best.pt --scan path/to/scan.mhd --out report_dir
    python infer.py ... --scan path/to/dicom_folder/          # DICOM series also works

Returns / writes report.json:
    findings: [{id, world_xyz_mm, voxel_zyx, slice_index, probability, uncertainty,
                needs_review, diameter_mm, lung_rads_size_category, snapshot}]
    summary : counts + a plain-language, NON-diagnostic statement for the clinician
The output is decision support for a radiologist, never an autonomous diagnosis.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config as C
from candidate_gen import generate_candidates
from luna_io import Scan
from predictor import Predictor, lung_rads_size_category

REVIEW_UNCERTAINTY = 0.15      # std of MC/TTA probabilities above which a finding is flagged


def nms(df, radius_mm=6.0, size_aware=False):
    """Keep the most probable candidate and drop others within radius_mm of it.
    size_aware=True (used by the apps) also drops candidates inside the kept nodule itself
    (within half its estimated diameter), so one large nodule is not reported twice.
    The evaluation scripts keep the original fixed radius, so published CPM values are unchanged."""
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)
    xyz = df[["coordX", "coordY", "coordZ"]].to_numpy()
    diam = df["diameter_mm"].to_numpy(float) if size_aware and "diameter_mm" in df else None
    keep, taken = [], np.zeros(len(df), bool)
    for i in range(len(df)):
        if taken[i]:
            continue
        keep.append(i)
        r = max(radius_mm, 0.5 * diam[i]) if diam is not None else radius_mm
        taken |= np.linalg.norm(xyz - xyz[i], axis=1) < r
    return df.iloc[keep].reset_index(drop=True)


def snapshot(scan, world, diameter, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    idx = scan.world_to_index_xyz(world)
    z = int(np.clip(round(idx[2]), 0, scan.array.shape[0] - 1))
    sl = np.clip((scan.array[z].astype(np.float32) + 1350) / 1500, 0, 1)
    half = int(48 / scan.spacing[0])
    cy, cx = int(round(idx[1])), int(round(idx[0]))
    y0, x0 = max(0, cy - half), max(0, cx - half)
    crop = sl[y0:cy + half, x0:cx + half]
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=130)
    axs[0].imshow(sl, cmap="gray"); axs[0].add_patch(Circle((cx, cy), 12 / scan.spacing[0], fill=False, color="#eb6834", lw=1.5))
    axs[1].imshow(crop, cmap="gray")
    axs[1].add_patch(Circle((cx - x0, cy - y0), max(3, diameter) / scan.spacing[0] / 2 + 3, fill=False, color="#eb6834", lw=1.5))
    for a in axs:
        a.axis("off")
    axs[0].set_title(f"axial slice {z}", fontsize=8); axs[1].set_title("zoom (96 mm)", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return z


def cam_snapshot(scan, world, cam, path):
    """Zoomed axial view (32 mm) through the finding with the Grad-CAM overlay, for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    patch = scan.extract_patch(world, size=C.LOCAL_SIZE)
    c = C.LOCAL_SIZE // 2
    fig, axs = plt.subplots(1, 2, figsize=(4.2, 2.2), dpi=130)
    for a in axs:
        a.imshow(np.clip((patch[c].astype(np.float32) + 1350) / 1500, 0, 1), cmap="gray")
        a.axis("off")
    axs[1].imshow(cam[c], cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axs[0].set_title("CT (32 mm)", fontsize=7); axs[1].set_title("AI attention (Grad-CAM)", fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


RESCORE_TOP = int(os.environ.get("LUNGSCAN_RESCORE_TOP", "150"))   # fast mode: candidates re-scored with full TTA + MC dropout
RESCORE_MIN_PROB = 0.05    # ... plus any candidate whose quick probability reaches this


def score_two_stage(pred, patches, rescore_top=RESCORE_TOP, rescore_min_prob=RESCORE_MIN_PROB):
    """Two-stage scoring: one quick pass (no flips, no MC dropout) over every candidate, then the
    full 8-flip TTA + MC-dropout scoring only for the candidates that could matter (the top
    `rescore_top` and any with quick probability >= `rescore_min_prob`). Reported findings are
    scored with the full method; the AI step is about 4x faster. Check: eval_end_to_end.py --fast."""
    n = len(patches)
    if n <= rescore_top or (not pred.tta and pred.mc <= 1):
        return pred.score(patches)
    tta, mc = pred.tta, pred.mc
    pred.tta, pred.mc = False, 1
    try:
        quick = pred.score(patches)
    finally:
        pred.tta, pred.mc = tta, mc
    order = np.argsort(-quick["prob"])
    sel = np.union1d(order[:rescore_top], np.where(quick["prob"] >= rescore_min_prob)[0])
    full = pred.score(patches[sel])
    out = {k: np.array(v, copy=True) for k, v in quick.items()}
    for k, v in full.items():
        out[k][sel] = v
    out["n_full"] = len(sel)
    return out


class LungCAD:
    def __init__(self, ckpt, device=None, tta=True, mc=C.MC_SAMPLES, use_risk=True):
        self.pred = Predictor(ckpt, device=device, tta=tta, mc=mc)
        self.risk = None
        if use_risk:
            from risk_model import RiskModel
            if RiskModel.available():
                try:
                    self.risk = RiskModel(device=device)
                except Exception as e:          # the detector still works without the risk model
                    print("cancer-risk model not loaded:", e)

    def score_fast(self, patches):
        return score_two_stage(self.pred, patches)

    def analyze(self, scan_path, out_dir=None, threshold=0.5, max_findings=10, candidates=None, age=None,
                review_uncertainty=REVIEW_UNCERTAINTY, explain=True, fast=True):
        t0 = time.time()
        timing = {}
        scan = Scan(scan_path)
        timing["load_s"] = round(time.time() - t0, 1)
        if candidates is None:
            t = time.time()
            candidates, _ = generate_candidates(scan.image)
            timing["candidate_finder_s"] = round(time.time() - t, 1)
        t = time.time()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as ex:
            patches = scan.extract_many(candidates[["coordX", "coordY", "coordZ"]].to_numpy(), executor=ex)
        timing["patches_s"] = round(time.time() - t, 1)
        t = time.time()
        s = self.score_fast(patches) if fast else self.pred.score(patches)
        timing["ai_scoring_s"] = round(time.time() - t, 1)
        t = time.time()
        df = candidates.copy().reset_index(drop=True)
        df["patch_idx"] = np.arange(len(df))
        df["probability"], df["uncertainty"] = s["prob"], s["uncertainty"]
        df["diameter_mm"] = s["diameter_mm"]
        df = nms(df, size_aware=True)
        found = df[df.probability >= threshold].head(max_findings)
        # borderline candidates below threshold but highly uncertain are surfaced for review
        border = df[(df.probability < threshold) & (df.probability >= 0.25)
                    & (df.uncertainty > review_uncertainty)].head(3)
        sel = pd.concat([found, border])
        risks = None
        if self.risk is not None and len(sel):
            from risk_model import risk_band
            risks = self.risk.score(patches[sel.patch_idx.to_numpy()], sel.diameter_mm.to_numpy(), age)
        findings = []
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for k, (_, r) in enumerate(sel.iterrows(), 1):
            w = r[["coordX", "coordY", "coordZ"]].to_numpy(float)
            vi = scan.world_to_index_xyz(w)
            f = {
                "id": k,
                "world_xyz_mm": [round(float(x), 2) for x in w],
                "voxel_zyx": [int(round(vi[2])), int(round(vi[1])), int(round(vi[0]))],
                "probability": round(float(r.probability), 4),
                "uncertainty": round(float(r.uncertainty), 4),
                "above_threshold": bool(r.probability >= threshold),
                "needs_review": bool(r.uncertainty > review_uncertainty or 0.3 <= r.probability <= 0.7),
                "diameter_mm": round(float(r.diameter_mm), 1),
                "lung_rads_size_category": lung_rads_size_category(float(r.diameter_mm)),
            }
            if risks is not None:
                f["cancer_risk"] = round(float(risks[k - 1]), 4)
                f["cancer_risk_band"] = risk_band(float(risks[k - 1]))
            if out_dir:
                p = os.path.join(out_dir, f"finding_{k}.png")
                f["slice_index"] = snapshot(scan, w, f["diameter_mm"], p)
                f["snapshot"] = os.path.basename(p)
                if explain:
                    try:                     # Grad-CAM heatmap for "why did the AI flag this?"
                        from explain import gradcam
                        cam = gradcam(self.pred.model, patches[int(r.patch_idx)], self.pred.windows, self.pred.device)
                        np.save(os.path.join(out_dir, f"cam_{k}.npy"), (cam * 255).astype(np.uint8))
                        f["cam"] = f"cam_{k}.npy"
                        cam_snapshot(scan, w, cam, os.path.join(out_dir, f"finding_{k}_cam.png"))
                        f["cam_snapshot"] = f"finding_{k}_cam.png"
                    except Exception as e:
                        print("Grad-CAM failed:", e)
            findings.append(f)
        timing["risk_heatmaps_s"] = round(time.time() - t, 1)
        print("analysis timing:", timing)
        n_pos = sum(f["above_threshold"] for f in findings)
        largest = max([f["diameter_mm"] for f in findings if f["above_threshold"]], default=0)
        summary = {
            "scan": os.path.basename(os.path.normpath(scan_path)),
            "n_candidates_screened": int(len(candidates)),
            "n_nodules_detected": int(n_pos),
            "n_flagged_for_review": int(sum(f["needs_review"] for f in findings)),
            "largest_nodule_mm": largest,
            "max_probability": round(float(df.probability.max()), 4) if len(df) else 0.0,
            "statement": ("No pulmonary nodule above the detection threshold." if n_pos == 0 else
                          f"{n_pos} suspected pulmonary nodule(s); largest estimated {largest:.1f} mm. "
                          "Size-based Lung-RADS categories are indicative only (solid nodule assumed) "
                          "and must be confirmed by a radiologist."),
            "threshold": threshold,
            "runtime_s": round(time.time() - t0, 1),
            "timing": timing,
            "scoring": (f"fast: quick pass on all, full TTA + MC on {s.get('n_full', len(df))}" if fast else "full TTA + MC on all"),
            "disclaimer": "Research prototype - decision support only, not a medical diagnosis.",
            "cancer_risk_model": ("LUNA25/NLST-trained ensemble + size" + (" + age" if age is not None else "")
                                  if self.risk is not None else None),
        }
        report = {"summary": summary, "findings": findings}
        if out_dir:
            json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scan", required=True, help=".mhd file or DICOM folder")
    ap.add_argument("--out", default="report")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--luna-candidates", action="store_true",
                    help="use candidates_V2.csv rows for this scan instead of the built-in detector")
    args = ap.parse_args()
    cad = LungCAD(args.ckpt)
    cands = None
    if args.luna_candidates:
        uid = os.path.splitext(os.path.basename(args.scan))[0]
        c = pd.read_csv(C.CANDIDATES_CSV)
        cands = c[c.seriesuid == uid][["coordX", "coordY", "coordZ"]].reset_index(drop=True)
    rep = cad.analyze(args.scan, args.out, args.threshold, candidates=cands)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
