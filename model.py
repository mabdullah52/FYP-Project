"""MSGF-Net: Multi-window, Scale-Gated Fusion network for pulmonary nodule
false-positive reduction with diameter estimation and uncertainty.

Input
    local   : (B, W, 32, 32, 32)  1 mm voxels, 32 mm field of view around the candidate
    context : (B, W, 48, 48, 48)  1 mm voxels, 48 mm field of view (same centre)
    W = number of HU windows (default 3: lung, mediastinal, wide)

Architecture
    two 3D residual streams with squeeze-excitation (SE) -> global descriptors z_l, z_c
    Scale-Gated Fusion (SGF):
        1. context-conditioned recalibration: F_l' = F_l * sigmoid(W_r z_c)
           (the surrounding anatomy decides which local features matter - e.g. a
           tubular context down-weights "round blob" evidence for vessels)
        2. per-channel gate  a = sigmoid(MLP([z_l', z_c]))
           z = a * z_l' + (1 - a) * z_c      (a is returned for interpretability)
    heads: nodule logit (MC-dropout head) and nodule diameter in mm (auxiliary task)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE3d(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(4, c // r)), nn.ReLU(inplace=True),
                                nn.Linear(max(4, c // r), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3, 4)))
        return x * w[:, :, None, None, None]


class ResBlock3d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.se = SE3d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.se(self.b2(self.c2(y)))
        return F.relu(y + (x if self.short is None else self.short(x)), inplace=True)


class Stream(nn.Module):
    """3D ResNet-SE encoder. stem_stride=2 for the larger context cube."""

    def __init__(self, cin, w, stem_stride=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(cin, w, 3, stem_stride, 1, bias=False),
                                  nn.BatchNorm3d(w), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResBlock3d(w, w), ResBlock3d(w, 2 * w, 2), ResBlock3d(2 * w, 2 * w),
            ResBlock3d(2 * w, 4 * w, 2), ResBlock3d(4 * w, 4 * w),
            ResBlock3d(4 * w, 4 * w, 2),
        )
        self.out_ch = 4 * w

    def forward(self, x):
        return self.layers(self.stem(x))


class MSGFNet(nn.Module):
    def __init__(self, in_ch=3, width=24, dropout=0.3, use_context=True, fusion="gated"):
        super().__init__()
        self.use_context, self.fusion = use_context, fusion
        self.local = Stream(in_ch, width, 1)
        c = self.local.out_ch
        if use_context:
            self.context = Stream(in_ch, width, 2)
            if fusion == "gated":
                self.recal = nn.Linear(c, c)
                self.gate = nn.Sequential(nn.Linear(2 * c, c), nn.ReLU(inplace=True), nn.Linear(c, c))
                feat = c
            elif fusion == "concat":
                feat = 2 * c
            else:
                raise ValueError(fusion)
        else:
            feat = c
        self.feat_dim = feat
        self.drop_p = dropout
        self.cls1 = nn.Linear(feat, 64)
        self.cls2 = nn.Linear(64, 1)
        self.diam = nn.Linear(feat, 1)

    # ---------------------------------------------------------------------------------
    def features(self, local, context=None):
        fl = self.local(local)
        alpha = None
        if not self.use_context:
            return fl.mean(dim=(2, 3, 4)), alpha
        fc = self.context(context)
        zc = fc.mean(dim=(2, 3, 4))
        if self.fusion == "concat":
            return torch.cat([fl.mean(dim=(2, 3, 4)), zc], 1), alpha
        fl = fl * torch.sigmoid(self.recal(zc))[:, :, None, None, None]
        zl = fl.mean(dim=(2, 3, 4))
        alpha = torch.sigmoid(self.gate(torch.cat([zl, zc], 1)))
        return alpha * zl + (1 - alpha) * zc, alpha

    def head(self, z, mc=False):
        """mc=True keeps dropout active (Monte-Carlo dropout) even in eval mode."""
        train = self.training or mc
        h = F.dropout(z, self.drop_p, train)
        h = F.relu(self.cls1(h), inplace=True)
        h = F.dropout(h, self.drop_p, train)
        logit = self.cls2(h).squeeze(1)
        diam = F.softplus(self.diam(z).squeeze(1)) + 1.0     # mm, >= 1
        return logit, diam

    def forward(self, local, context=None):
        z, alpha = self.features(local, context)
        logit, diam = self.head(z)
        return {"logit": logit, "diameter": diam, "alpha": alpha}


def build_model(cfg):
    if cfg.get("arch", "msgf") != "msgf":            # comparison architectures (baselines.py)
        from baselines import build_baseline
        return build_baseline(cfg)
    return MSGFNet(in_ch=cfg["in_ch"], width=cfg["width"], dropout=cfg["dropout"],
                   use_context=cfg["use_context"], fusion=cfg["fusion"])


if __name__ == "__main__":
    m = MSGFNet()
    n = sum(p.numel() for p in m.parameters())
    o = m(torch.randn(2, 3, 32, 32, 32), torch.randn(2, 3, 48, 48, 48))
    print(f"params: {n / 1e6:.2f} M", {k: tuple(v.shape) for k, v in o.items()})
