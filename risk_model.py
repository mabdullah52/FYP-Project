"""Per-nodule cancer-risk model (LUNA25-trained) used by infer.py / the doctor app.

Loads work/runs/l25_risk/risk_model.json written by luna25_final.py:
ensemble of MSGF-Net malignancy models -> logistic fusion with the predicted diameter (+ age).
"""
import json
import os

import numpy as np

import config as C
from predictor import Predictor

RISK_JSON = os.path.join(C.RUNS_DIR, "l25_risk", "risk_model.json")


def risk_band(p):
    if p < 0.05:
        return "low"
    if p < 0.30:
        return "intermediate"
    return "high"


class RiskModel:
    def __init__(self, path=RISK_JSON, device=None):
        self.cfg = json.load(open(path))
        self.members = [Predictor(os.path.join(C.RUNS_DIR, *m.replace("\\", "/").split("/")), device=device, apply_calibration=False)
                        for m in self.cfg["members"]]

    @staticmethod
    def available(path=RISK_JSON):
        return os.path.exists(path)

    def score(self, patches_hu, diameters_mm, age=None):
        if len(patches_hu) == 0:
            return np.zeros(0)
        raw = np.mean([m.score(patches_hu)["prob_raw"] for m in self.members], 0)
        p = np.clip(raw, 1e-6, 1 - 1e-6)
        x = [np.log(p / (1 - p)), np.asarray(diameters_mm, float)]
        use_age = age is not None and self.cfg.get("with_age")
        if use_age:
            x.append(np.full(len(p), float(age)))
        X = np.column_stack(x)
        k = X.shape[1]
        X = (X - np.array(self.cfg["feature_mean"][:k])) / np.array(self.cfg["feature_sd"][:k])
        w = self.cfg["with_age"] if use_age else self.cfg["without_age"]
        z = X @ np.array(w["coef"]) + w["intercept"]
        return 1 / (1 + np.exp(-z))
