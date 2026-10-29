"""Shared scoring engine used by evaluate.py, mine_hard_negatives.py, infer.py and the apps.

Uncertainty-aware inference:
    * test-time augmentation: 8 flip combinations of the 3D patch
    * Monte-Carlo dropout: for each TTA view the (expensive) 3D encoders run ONCE and
      only the small dropout head is sampled `mc` times, so uncertainty is almost free.
Outputs per candidate: probability (mean), uncertainty (std of the sampled
probabilities), predictive entropy, nodule diameter in mm, mean fusion gate alpha.
"""
import itertools
import json
import os

import numpy as np
import torch

import config as C
from dataset import centre_crop
from luna_io import hu_to_windows
from model import build_model

FLIPS = [f for r in range(4) for f in itertools.combinations((2, 3, 4), r)]   # 8 combos


class Predictor:
    def __init__(self, ckpt_path, device=None, tta=C.TTA, mc=C.MC_SAMPLES, apply_calibration=True):
        torch.manual_seed(0)                      # reproducible MC-dropout samples
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = ck["cfg"]
        self.model = build_model(self.cfg).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.windows = [tuple(w) for w in self.cfg["windows"]]
        self.tta, self.mc = tta, max(1, mc)
        # Platt scaling fitted on the validation subset by `evaluate.py --calibrate`
        cal_path = os.path.join(os.path.dirname(ckpt_path), "calibration.json")
        self.calibration = json.load(open(cal_path)) if apply_calibration and os.path.exists(cal_path) else None

    @torch.no_grad()
    def score(self, patches_hu, batch_size=64):
        """patches_hu: (N, 48, 48, 48) HU -> dict of numpy arrays (N,)."""
        out = {k: [] for k in ("prob", "uncertainty", "entropy", "diameter_mm", "alpha")}
        flips = FLIPS if self.tta else [()]
        use_ctx = self.cfg["use_context"]
        for s in range(0, len(patches_hu), batch_size):
            x = torch.from_numpy(hu_to_windows(patches_hu[s:s + batch_size], self.windows)).to(self.device)
            probs, diams, alphas = [], [], []
            for f in flips:
                xc = torch.flip(x, f) if f else x
                loc = centre_crop(xc, C.LOCAL_SIZE)
                with torch.autocast(self.device.type, enabled=C.amp_enabled(self.device)):
                    z, alpha = self.model.features(loc, xc if use_ctx else None)
                z = z.float()
                for m in range(self.mc):
                    logit, diam = self.model.head(z, mc=self.mc > 1)
                    probs.append(torch.sigmoid(logit))
                    if m == 0:
                        diams.append(diam)
                if alpha is not None:
                    alphas.append(alpha.float().mean(1))
            P = torch.stack(probs)                                   # (views*mc, B)
            p = P.mean(0)
            ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
            out["prob"].append(p.cpu()); out["uncertainty"].append(P.std(0, unbiased=False).cpu())  # 0 when only 1 sample
            out["entropy"].append(ent.cpu()); out["diameter_mm"].append(torch.stack(diams).mean(0).cpu())
            out["alpha"].append(torch.stack(alphas).mean(0).cpu() if alphas else torch.full_like(p.cpu(), float("nan")))
        res = {k: (torch.cat(v).numpy() if v else np.zeros(0, np.float32)) for k, v in out.items()}
        res["prob_raw"] = res["prob"]
        if self.calibration is not None:
            res["prob"] = platt(res["prob"], self.calibration)
        return res


def platt(p, cal):
    """Monotonic recalibration -> the ranking (and therefore FROC/CPM/AUC) is unchanged."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = cal["a"] * np.log(p / (1 - p)) + cal["b"]
    return (0.5 * (1 + np.tanh(0.5 * z))).astype(np.float32)          # overflow-safe sigmoid


def lung_rads_size_category(d_mm):
    """Size-only Lung-RADS v2022 baseline category for a SOLID nodule (decision support only)."""
    if d_mm < 6:
        return "2 (benign appearance, <6 mm)"
    if d_mm < 8:
        return "3 (probably benign, 6-<8 mm)"
    if d_mm < 15:
        return "4A (suspicious, 8-<15 mm)"
    return "4B (very suspicious, >=15 mm)"
