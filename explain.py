"""Explainability: 3D Grad-CAM for MSGF-Net findings ("why did the AI flag this?").

Grad-CAM (Selvaraju et al., ICCV 2017) on the LOCAL stream at 16^3 resolution (2 mm cells over the
32 mm local cube), w.r.t. the nodule logit, upsampled to 32^3 at 1 mm. The cube is centred on the
finding and aligned with the patient axes, so the doctor app can overlay it on the CT slices.
"""
import numpy as np
import torch
import torch.nn.functional as F

import config as C
from dataset import make_inputs


def gradcam(model, patch_hu, windows, device):
    """patch_hu: (48,48,48) int16 HU around the finding. Returns float32 (32,32,32) in [0,1] (z,y,x)."""
    model.eval()
    feats = {}

    def hook(_, __, out):
        out.retain_grad()
        feats["a"] = out

    layer = model.local.layers[2]                      # 16^3 feature map
    h = layer.register_forward_hook(hook)
    try:
        loc, ctx = make_inputs(np.asarray(patch_hu), windows, model.use_context)
        loc, ctx = loc[None].to(device), (ctx[None].to(device) if model.use_context else None)
        with torch.enable_grad():
            z, _ = model.features(loc, ctx)
            logit, _ = model.head(z, mc=False)
            model.zero_grad(set_to_none=True)
            logit.sum().backward()
        a, g = feats["a"], feats["a"].grad
        w = g.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(C.LOCAL_SIZE,) * 3, mode="trilinear", align_corners=False)[0, 0]
        cam = cam.detach().float().cpu().numpy()
    finally:
        h.remove()
    m = cam.max()
    return (cam / m).astype(np.float32) if m > 0 else cam.astype(np.float32)


def cam_to_rgba(slice2d):
    """(H,W) in [0,1] -> (H,W,4) uint8 'jet-like' colours with alpha rising with the value."""
    v = np.clip(slice2d, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    a = np.clip((v - 0.15) / 0.6, 0, 1) * 0.65
    return (np.stack([r, g, b, a], -1) * 255).astype(np.uint8)
