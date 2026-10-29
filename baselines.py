"""Comparison architectures for paper 1, trained and evaluated with EXACTLY the same data, split,
sampling, augmentation, loss, training budget and inference as MSGF-Net (only the network differs).

    resnet3d18   3D ResNet-18 (Hara et al. 2018 style, standard stride-2 stem + max-pool), 32 mm cube, ~33 M parameters
    resnet25d    2.5D ResNet-18: axial + coronal + sagittal centre planes of the 3 HU windows (9 channels)
    cnn3d        plain 3D CNN (4 conv blocks + max-pooling), the classic FP-reduction baseline
All use the local 32 mm view only and the same classification + diameter heads as MSGF-Net
(so TTA, MC dropout, calibration and the evaluation code apply unchanged).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ("resnet3d18", "resnet25d", "cnn3d")


class _Heads(nn.Module):
    """Same heads as MSGF-Net: dropout MLP classifier (MC-dropout capable) + diameter regressor."""

    use_context = False
    fusion = None

    def _init_heads(self, feat, dropout):
        self.feat_dim = feat
        self.drop_p = dropout
        self.cls1 = nn.Linear(feat, 64)
        self.cls2 = nn.Linear(64, 1)
        self.diam = nn.Linear(feat, 1)

    def head(self, z, mc=False):
        train = self.training or mc
        h = F.dropout(z, self.drop_p, train)
        h = F.relu(self.cls1(h), inplace=True)
        h = F.dropout(h, self.drop_p, train)
        logit = self.cls2(h).squeeze(1)
        diam = F.softplus(self.diam(z).squeeze(1)) + 1.0
        return logit, diam

    def forward(self, local, context=None):
        z, alpha = self.features(local, context)
        logit, diam = self.head(z)
        return {"logit": logit, "diameter": diam, "alpha": alpha}


class Block(nn.Module):
    def __init__(self, cin, cout, stride, dim):
        super().__init__()
        Conv, BN = (nn.Conv3d, nn.BatchNorm3d) if dim == 3 else (nn.Conv2d, nn.BatchNorm2d)
        self.c1 = Conv(cin, cout, 3, stride, 1, bias=False)
        self.b1 = BN(cout)
        self.c2 = Conv(cout, cout, 3, 1, 1, bias=False)
        self.b2 = BN(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(Conv(cin, cout, 1, stride, bias=False), BN(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)), inplace=True)


def _resnet18_body(cin, dim, widths=(64, 128, 256, 512), standard_stem=False):
    Conv, BN = (nn.Conv3d, nn.BatchNorm3d) if dim == 3 else (nn.Conv2d, nn.BatchNorm2d)
    if standard_stem:   # standard ResNet stem: 7x7(x7) conv stride 2 + 3x3(x3) max-pool stride 2
        Pool = nn.MaxPool3d if dim == 3 else nn.MaxPool2d
        layers = [Conv(cin, widths[0], 7, 2, 3, bias=False), BN(widths[0]), nn.ReLU(inplace=True), Pool(3, 2, 1)]
    else:
        layers = [Conv(cin, widths[0], 3, 1, 1, bias=False), BN(widths[0]), nn.ReLU(inplace=True)]
    c = widths[0]
    for i, w in enumerate(widths):
        s = 1 if i == 0 else 2
        layers += [Block(c, w, s, dim), Block(w, w, 1, dim)]
        c = w
    return nn.Sequential(*layers), c


class ResNet3D18(_Heads):
    def __init__(self, in_ch=3, dropout=0.3):
        super().__init__()
        self.body, c = _resnet18_body(in_ch, 3, standard_stem=True)
        self._init_heads(c, dropout)

    def features(self, local, context=None):
        return self.body(local).mean(dim=(2, 3, 4)), None


class ResNet25D(_Heads):
    """Three orthogonal centre planes (each with the 3 HU windows) stacked as 9 input channels."""

    def __init__(self, in_ch=3, dropout=0.3):
        super().__init__()
        self.body, c = _resnet18_body(3 * in_ch, 2)
        self._init_heads(c, dropout)

    def features(self, local, context=None):
        n, ch, d, h, w = local.shape
        ax = local[:, :, d // 2]                  # (n, ch, h, w)
        co = local[:, :, :, h // 2]               # (n, ch, d, w)
        sa = local[:, :, :, :, w // 2]            # (n, ch, d, h)
        return self.body(torch.cat([ax, co, sa], 1)).mean(dim=(2, 3)), None


class CNN3D(_Heads):
    def __init__(self, in_ch=3, dropout=0.3, widths=(32, 64, 128, 256)):
        super().__init__()
        layers, c = [], in_ch
        for i, w in enumerate(widths):
            layers += [nn.Conv3d(c, w, 3, 1, 1, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True),
                       nn.Conv3d(w, w, 3, 1, 1, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool3d(2))
            c = w
        self.body = nn.Sequential(*layers)
        self._init_heads(c, dropout)

    def features(self, local, context=None):
        return self.body(local).mean(dim=(2, 3, 4)), None


def build_baseline(cfg):
    arch, in_ch, dp = cfg["arch"], cfg["in_ch"], cfg["dropout"]
    if arch == "resnet3d18":
        return ResNet3D18(in_ch, dp)
    if arch == "resnet25d":
        return ResNet25D(in_ch, dp)
    if arch == "cnn3d":
        return CNN3D(in_ch, dp)
    raise ValueError(f"unknown architecture {arch}")


if __name__ == "__main__":
    for a in ARCHS:
        m = build_baseline({"arch": a, "in_ch": 3, "dropout": 0.3})
        o = m(torch.randn(2, 3, 32, 32, 32))
        print(a, f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M", {k: tuple(v.shape) for k, v in o.items() if v is not None})
