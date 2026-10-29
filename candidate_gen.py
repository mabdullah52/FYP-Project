"""Candidate generation for NEW scans (the apps have no candidates.csv).

Classical, training-free, high-sensitivity detector:
  1. resample to 1 mm isotropic
  2. lung mask: large air components with vessels (mean HU > -970) not touching the image edge
     (robust to a circular field of view cutting the lungs), 10 mm slice-wise closing + hole
     filling (keeps juxtapleural nodules and masses), dilate 3 mm
  3. multi-scale scale-normalised Laplacian-of-Gaussian blob response (sigma 1.5-8 mm)
  4. local maxima inside the lung mask with HU > -700, ranked by LoG response x Hessian blobness
     (round blobs beat vessel segments); 150 places reserved for large blobs, `max_cands` in total
MSGF-Net then removes the false positives. Measure its recall on a LUNA16 subset:

    python candidate_gen.py --eval-subset 9
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

FINDER_VERSION = "v5"       # v5: + Hessian blobness ranking, min HU -700 (v2 0.53, v3 0.70, v4 0.75 recall on subset 9)
SIGMAS_MM = (1.5, 2.5, 4.0, 6.0, 8.0)
LARGE_QUOTA = 150            # candidates reserved for large blobs (sigma >= 4 mm, i.e. >= ~14 mm)


def resample_iso(img, spacing=1.0):
    sp, sz = np.array(img.GetSpacing()), np.array(img.GetSize())
    new_size = [int(round(s)) for s in sz * sp / spacing]
    return sitk.Resample(img, new_size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                         (spacing,) * 3, img.GetDirection(), -1000, sitk.sitkInt16)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _disk_closing(m, r=10):
    """Exact binary closing of a 2D mask with the disk x^2+y^2 <= r^2 (border = False), via Euclidean
    distance transforms: ~50x faster than ndi.binary_closing with a 21x21 structuring element."""
    dil = ndi.distance_transform_edt(~m) <= r
    q = r + 1
    return ndi.distance_transform_edt(np.pad(dil, q))[q:-q, q:-q] > r


def _lungs_inside_body(vol, pad):
    """Fallback when the outside air cannot be separated from the lungs in 3D (e.g. the FOV padding
    has the same value as air): per axial slice, the body (largest > -500 HU region, holes filled)
    encloses the lungs on every slice where the field of view does not cut them."""
    lungs = np.zeros(vol.shape, bool)
    for z in range(vol.shape[0]):
        b = vol[z] > -500
        lab, n = ndi.label(b)
        if n == 0:
            continue
        sizes = ndi.sum(b, lab, range(1, n + 1))
        body = ndi.binary_fill_holes(lab == (np.argmax(sizes) + 1))
        lungs[z] = body & (vol[z] < -400) & ~pad[z]
    lab, n = ndi.label(lungs)
    if n == 0:
        return lungs
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(lungs, lab, idx)
    means = ndi.mean(vol, lab, idx)
    ok = (sizes > 50000) & (means > -970)
    if not ok.any():
        return np.zeros(vol.shape, bool)
    ok &= sizes >= 0.1 * sizes[ok].max()
    return np.isin(lab, idx[ok])


def lung_mask(vol):
    """Lung mask that survives a circular field of view cutting through the body or lungs.
    Air-like voxels (-1500..-400 HU; padding outside the scanner FOV is usually < -1500) are split
    into 3D components; components touching the image edge are outside air; among the rest, LUNG
    components are large and contain vessels (mean HU above -970, pure air is about -1000), which
    also rejects the air ring between body and FOV circle. Then a 10 mm slice-wise closing + hole
    filling keeps nodules/masses that touch the chest wall or mediastinum inside the mask."""
    # padding outside the scanner's circular FOV (usually -2000/-3024 HU). Resampling blends it
    # with the image into a rim of "air-like" values that would glue the lungs to the outside air,
    # so the padding is grown by 3 mm and removed before looking for air components.
    pad = ndi.binary_opening(vol < -1150, iterations=1)
    pad = ndi.binary_dilation(pad, iterations=3)
    air = ~pad & (vol < -400)
    lab, n = ndi.label(air)
    lungs = np.zeros(vol.shape, bool)
    if n:
        edge = np.unique(np.concatenate([lab[:, 0, :].ravel(), lab[:, -1, :].ravel(),
                                         lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        idx = np.arange(1, n + 1)
        sizes = ndi.sum(air, lab, idx)
        means = ndi.mean(vol, lab, idx)
        ok = (sizes > 50000) & (means > -970) & ~np.isin(idx, edge)
        if ok.any():
            ok &= sizes >= 0.1 * sizes[ok].max()
            lungs = np.isin(lab, idx[ok])
    if lungs.sum() < 300000:                      # 2nd try: air enclosed by the body, slice by slice
        lungs = _lungs_inside_body(vol, pad)
    if lungs.sum() < 300000:                      # lungs not found: search the whole field of view
        return ndi.binary_erosion(~pad, iterations=5)
    for z in range(lungs.shape[0]):
        if lungs[z].any():
            m = _disk_closing(np.pad(lungs[z], 12), 10)[12:-12, 12:-12]
            lungs[z] = ndi.binary_fill_holes(m)
    lungs = ndi.binary_closing(lungs, iterations=2)
    return ndi.binary_dilation(lungs, iterations=3)


# ---------------------------------------------------------------------------------------------
# Filters. On a GPU (torch) the 3D Gaussian / LoG filters take a few seconds instead of minutes;
# they use the same kernels and border handling as scipy.ndimage, so the candidates are the same.
# Without a GPU, the scipy filters run in parallel threads (scipy releases the GIL).
# ---------------------------------------------------------------------------------------------
_TORCH_DEV = None


def _torch_device():
    """CUDA device for the finder, or None (CPU / disabled with LUNGSCAN_FINDER_GPU=0)."""
    global _TORCH_DEV
    if _TORCH_DEV is None:
        _TORCH_DEV = False
        if os.environ.get("LUNGSCAN_FINDER_GPU", "1") != "0":
            try:
                import torch
                if torch.cuda.is_available():
                    _TORCH_DEV = torch.device("cuda")
            except Exception:
                pass
    return _TORCH_DEV or None


class _TorchFilters:
    """scipy.ndimage.gaussian_filter / gaussian_laplace (truncate 4, mode 'reflect') in torch."""

    def __init__(self, v, device):
        import torch
        self.torch, self.dev = torch, device
        self.v = torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(device)[None, None]

    def _conv(self, x, sigma, order, axis):
        import torch.nn.functional as F
        from scipy.ndimage._filters import _gaussian_kernel1d
        r = int(4.0 * float(sigma) + 0.5)
        w = _gaussian_kernel1d(float(sigma), order, r)[::-1].copy()    # same weights as scipy
        n = x.shape[2 + axis]
        if r >= n:
            raise ValueError("volume too small for the GPU filter")
        d = 2 + axis
        lo = x.narrow(d, 0, r).flip(d)                                 # 'reflect' = d c b a | a b c d
        hi = x.narrow(d, n - r, r).flip(d)
        xp = self.torch.cat([lo, x, hi], d)
        shape = [1, 1, 1, 1, 1]
        shape[d] = len(w)
        k = self.torch.tensor(w, dtype=x.dtype, device=x.device).view(shape)
        return F.conv3d(xp, k)

    def gauss(self, sigma, orders=(0, 0, 0)):
        x = self.v
        for ax, o in enumerate(orders):
            x = self._conv(x, sigma, o, ax)
        return x

    def log_norm(self, sigma):
        """-sigma^2 * LoG (bright blobs positive)."""
        c = self._conv
        g0 = c(self.v, sigma, 0, 0)                                    # smooth along z
        t = c(c(g0, sigma, 0, 1), sigma, 2, 2)                          # d2/dx2
        t += c(c(g0, sigma, 2, 1), sigma, 0, 2)                         # d2/dy2
        del g0
        t += c(c(c(self.v, sigma, 2, 0), sigma, 0, 1), sigma, 0, 2)     # d2/dz2
        return t.mul_(-(sigma ** 2))[0, 0]


def _log_stack(v):
    """Max over scales of the scale-normalised LoG response and the scale it came from (numpy)."""
    dev = _torch_device()
    if dev is not None:
        try:
            import torch
            prev = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False                       # full float32, like scipy
            try:
                with torch.no_grad():
                    tf = _TorchFilters(v, dev)
                    best = torch.full(v.shape, -float("inf"), device=dev)
                    best_s = torch.zeros(v.shape, device=dev)
                    for s in SIGMAS_MM:
                        r = tf.log_norm(s)
                        upd = r > best
                        best = torch.where(upd, r, best)
                        best_s[upd] = s
                        del r, upd
                    mx = torch.nn.functional.max_pool3d(best[None, None], 5, 1, 2)[0, 0]
                    is_max = (best == mx).cpu().numpy()
                    out = best.cpu().numpy(), best_s.cpu().numpy(), is_max
                    del tf, best, best_s, mx
                    return out + (True,)
            finally:
                torch.backends.cudnn.allow_tf32 = prev
                torch.cuda.empty_cache()
        except Exception as e:                       # e.g. GPU memory taken by another program
            print("finder: GPU filters unavailable (", str(e)[:80], ") - using CPU")
    with ThreadPoolExecutor(min(len(SIGMAS_MM), os.cpu_count() or 1)) as ex:
        resp = list(ex.map(lambda s: -(s ** 2) * ndi.gaussian_laplace(v, s), SIGMAS_MM))
    best = np.full(v.shape, -np.inf, np.float32)
    best_s = np.zeros(v.shape, np.float32)
    for s, r in zip(SIGMAS_MM, resp):
        upd = r > best
        best[upd], best_s[upd] = r[upd], s
    del resp
    return best, best_s, best == ndi.maximum_filter(best, size=5), False


def _hessian_nb(nb):
    """Finite-difference Hessian (n,3,3) from the 3x3x3 neighbourhoods (n,3,3,3) of the smoothed volume."""
    g = lambda dz, dy, dx: nb[:, 1 + dz, 1 + dy, 1 + dx]
    c = g(0, 0, 0)
    hzz = g(1, 0, 0) - 2 * c + g(-1, 0, 0)
    hyy = g(0, 1, 0) - 2 * c + g(0, -1, 0)
    hxx = g(0, 0, 1) - 2 * c + g(0, 0, -1)
    hzy = (g(1, 1, 0) - g(1, -1, 0) - g(-1, 1, 0) + g(-1, -1, 0)) / 4
    hzx = (g(1, 0, 1) - g(1, 0, -1) - g(-1, 0, 1) + g(-1, 0, -1)) / 4
    hyx = (g(0, 1, 1) - g(0, 1, -1) - g(0, -1, 1) + g(0, -1, -1)) / 4
    return np.stack([np.stack([hzz, hzy, hzx], -1), np.stack([hzy, hyy, hyx], -1),
                     np.stack([hzx, hyx, hxx], -1)], -2)


def _blobness(v, zz, yy, xx, sig, use_gpu=False):
    """Hessian eigenvalue ratio at each peak, at its own LoG scale: ~1 for a round blob (nodule),
    ~0 for a tube (vessel) or a plate (fissure). Computed only at the peaks, so it is cheap."""
    out = np.zeros(len(zz), np.float32)
    Z, Y, X = v.shape
    scales = np.unique(sig)
    smoothed = {}
    if use_gpu and _torch_device() is not None:
        try:
            import torch
            prev = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
            try:
                with torch.no_grad():
                    tf = _TorchFilters(v, _torch_device())
                    for s in scales:
                        i = np.where(sig == s)[0]
                        z, y, x = np.clip(zz[i], 1, Z - 2), np.clip(yy[i], 1, Y - 2), np.clip(xx[i], 1, X - 2)
                        g = tf.gauss(float(s))[0, 0]
                        # only the 3x3x3 neighbourhoods of the peaks go back to the CPU
                        off = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1)
                        zi = torch.as_tensor(z[:, None] + off[0], device=g.device)
                        yi = torch.as_tensor(y[:, None] + off[1], device=g.device)
                        xi = torch.as_tensor(x[:, None] + off[2], device=g.device)
                        smoothed[s] = g[zi, yi, xi].reshape(-1, 3, 3, 3).cpu().numpy()
                        del g
                    del tf
            finally:
                torch.backends.cudnn.allow_tf32 = prev
                torch.cuda.empty_cache()
        except Exception as e:
            print("finder: GPU blobness unavailable (", str(e)[:80], ") - using CPU")
            smoothed = {}
    if not smoothed:
        def local(s):
            i = np.where(sig == s)[0]
            g = ndi.gaussian_filter(v, float(s))
            z, y, x = np.clip(zz[i], 1, Z - 2), np.clip(yy[i], 1, Y - 2), np.clip(xx[i], 1, X - 2)
            nb = np.stack([g[z + a, y + b, x + c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], 1)
            return s, nb.reshape(-1, 3, 3, 3)
        with ThreadPoolExecutor(min(len(scales), os.cpu_count() or 1) or 1) as ex:
            smoothed = dict(ex.map(local, scales))
    for s in scales:
        i = np.where(sig == s)[0]
        H = _hessian_nb(smoothed[s])
        ev = np.linalg.eigvalsh(H)                       # ascending: l1 <= l2 <= l3
        l1, l3 = ev[:, 0], ev[:, 2]
        out[i] = np.where((l1 < 0) & (l3 < 0), l3 / np.minimum(l1, -1e-6), 0.0)
    return np.clip(out, 0, 1)


def generate_candidates(img, max_cands=1000, min_hu=-700):
    """img: SimpleITK image (HU). Returns DataFrame coordX/Y/Z (world mm), response, est_diameter_mm."""
    iso = resample_iso(img)
    vol = sitk.GetArrayFromImage(iso).astype(np.float32)          # (z, y, x) at 1 mm
    mask = lung_mask(vol)
    # work only inside the lung bounding box (+ a margin wider than the largest filter), which is
    # about half the scan: same candidates, roughly twice as fast
    M = 34
    nz = np.nonzero(mask.any(axis=(1, 2)))[0], np.nonzero(mask.any(axis=(0, 2)))[0], np.nonzero(mask.any(axis=(0, 1)))[0]
    off = [max(0, a[0] - M) if len(a) else 0 for a in nz]
    end = [min(n, a[-1] + M + 1) if len(a) else n for a, n in zip(nz, vol.shape)]
    crop = tuple(slice(o, e) for o, e in zip(off, end))
    vol_c, mask_c = vol[crop], mask[crop]
    v = np.clip(vol_c, -1000, 400)
    best, best_s, is_max, on_gpu = _log_stack(v)
    peaks = is_max & mask_c & (vol_c > min_hu) & (best > 50)
    zz, yy, xx = np.nonzero(peaks)
    resp, sig = best[zz, yy, xx], best_s[zz, yy, xx]
    # rank by LoG response x blobness: round nodules beat vessel segments of similar brightness
    # (tubes are down-weighted, not removed, so nodules touching vessels still get a place)
    score = resp * (0.25 + _blobness(v, zz, yy, xx, sig, use_gpu=on_gpu))
    order = np.argsort(-score)
    large = order[sig[order] >= 4.0][:LARGE_QUOTA]                 # big nodules/masses always kept
    rest = np.setdiff1d(order, large, assume_unique=True)
    rest = rest[np.argsort(-score[rest])][: max(0, max_cands - len(large))]
    top = np.concatenate([large, rest])
    idx_xyz = np.stack([xx[top] + off[2], yy[top] + off[1], zz[top] + off[0]], 1).astype(np.float64)
    D = np.array(iso.GetDirection()).reshape(3, 3)
    world = idx_xyz * 1.0 @ D.T + np.array(iso.GetOrigin())
    return pd.DataFrame({"coordX": world[:, 0], "coordY": world[:, 1], "coordZ": world[:, 2],
                         "response": resp[top],
                         "est_diameter_mm": 2 * np.sqrt(3) * sig[top]}), mask


def main():
    import config as C
    from luna_io import index_scans
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-subset", type=int, required=True)
    ap.add_argument("--max-scans", type=int, default=None)
    args = ap.parse_args()
    idx = index_scans()
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    uids = [u for u, (_, s) in idx.items() if s == args.eval_subset][: args.max_scans]
    hit = tot = ncand = 0
    for k, u in enumerate(uids, 1):
        cands, _ = generate_candidates(sitk.ReadImage(idx[u][0]))
        ncand += len(cands)
        cx = cands[["coordX", "coordY", "coordZ"]].to_numpy()
        for _, a in ann[ann.seriesuid == u].iterrows():
            d = np.linalg.norm(cx - a[["coordX", "coordY", "coordZ"]].to_numpy(float), axis=1)
            hit += int((d <= a.diameter_mm / 2).any()) if len(d) else 0
            tot += 1
        print(f"{k}/{len(uids)}  recall so far {hit}/{tot}  mean candidates/scan {ncand / k:.0f}")
    print(f"Candidate recall: {hit / max(1, tot):.3f} ({hit}/{tot}), {ncand / max(1, len(uids)):.0f} candidates/scan")


if __name__ == "__main__":
    main()
