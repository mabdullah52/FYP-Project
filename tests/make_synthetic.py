"""Create a tiny fake dataset in exact LUNA16 layout to smoke-test every script.

    python tests/make_synthetic.py --out /tmp/FAKE_LUNA16 --scans-per-subset 3
Not for training a real model - only to prove the code runs end to end.
"""
import argparse
import os
import uuid

import numpy as np
import pandas as pd
import SimpleITK as sitk


def draw_sphere(vol, sp, c_mm, r_mm, hu):
    zz, yy, xx = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    d = np.sqrt(((zz * sp[2] - c_mm[2]) ** 2) + ((yy * sp[1] - c_mm[1]) ** 2) + ((xx * sp[0] - c_mm[0]) ** 2))
    w = np.clip(r_mm + 0.5 - d, 0, 1)
    vol[:] = vol * (1 - w) + hu * w


def draw_tube(vol, sp, a, b, r_mm, hu, rng):
    n = int(np.linalg.norm(b - a) / 1.0) + 1
    for t in np.linspace(0, 1, n):
        p = a + t * (b - a)
        lo = np.clip(((p - r_mm - 1) / sp).astype(int)[::-1], 0, vol.shape)
        hi = np.clip(((p + r_mm + 2) / sp).astype(int)[::-1], 0, vol.shape)
        if (hi <= lo).any():
            continue
        sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = np.sqrt((zz * sp[2] - p[2]) ** 2 + (yy * sp[1] - p[1]) ** 2 + (xx * sp[0] - p[0]) ** 2)
        w = np.clip(r_mm + 0.5 - d, 0, 1)
        sub[:] = np.maximum(sub, sub * (1 - w) + hu * w)
    return [a + t * (b - a) for t in rng.uniform(0.1, 0.9, 4)]


def make_scan(rng, flip):
    sp = np.array([rng.uniform(0.7, 0.8), 0, rng.uniform(1.8, 2.5)]); sp[1] = sp[0]    # x, y, z mm
    shape = (int(170 / sp[2]), int(150 / sp[1]), int(150 / sp[0]))                      # z, y, x
    ext = np.array([shape[2] * sp[0], shape[1] * sp[1], shape[0] * sp[2]])              # x, y, z mm
    vol = np.full(shape, -1000.0, np.float32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    X, Y, Z = xx * sp[0], yy * sp[1], zz * sp[2]
    body = ((X - ext[0] / 2) / 70) ** 2 + ((Y - ext[1] / 2) / 60) ** 2 < 1
    vol[np.broadcast_to(body, shape)] = 40
    lungs = []
    for side in (-1, 1):
        c = np.array([ext[0] / 2 + side * 30, ext[1] / 2, ext[2] / 2])
        m = ((X - c[0]) / 24) ** 2 + ((Y - c[1]) / 38) ** 2 + ((Z - c[2]) / 75) ** 2 < 1
        vol[m] = -850
        lungs.append(c)
    vol += rng.normal(0, 25, shape).astype(np.float32)

    def lung_pt(margin=8):
        c = lungs[rng.randint(2)]
        while True:
            p = c + rng.uniform(-1, 1, 3) * np.array([24, 38, 75])
            if (((p - c) / (np.array([24, 38, 75]) - margin)) ** 2).sum() < 1:
                return p

    vessel_pts = []
    for _ in range(12):
        a = lung_pt(); b = a + rng.normal(0, 25, 3)
        vessel_pts += draw_tube(vol, sp, a, b, rng.uniform(1.0, 2.2), 40, rng)
    nods = []
    for _ in range(rng.choice([0, 1, 2, 2])):
        c, r = lung_pt(15), rng.uniform(2.5, 9)
        draw_sphere(vol, sp, c, r, 30)
        nods.append((c, 2 * r))
    arr = np.clip(vol, -1024, 2000).astype(np.int16)
    if flip:                       # store with a flipped x direction (tests direction handling)
        arr = arr[:, :, ::-1].copy()
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(sp))
    origin = np.array([-ext[0] / 2, -ext[1] / 2 - 20, -300.0])
    if flip:
        img.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))
        img.SetOrigin(tuple(origin + [(shape[2] - 1) * sp[0], 0, 0]))
    else:
        img.SetOrigin(tuple(origin))
    # world coordinates: with the flip the physical position of every structure is unchanged
    to_w = lambda p_mm: origin + p_mm
    rand_neg = [lung_pt() for _ in range(15)]
    return img, [(to_w(c), d) for c, d in nods], [to_w(p) for p in vessel_pts], [to_w(p) for p in rand_neg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/FAKE_LUNA16")
    ap.add_argument("--scans-per-subset", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    A, Cn, E = [], [], []
    for s in range(10):
        os.makedirs(os.path.join(args.out, f"subset{s}"), exist_ok=True)
        for k in range(args.scans_per_subset):
            uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001." + str(uuid.UUID(int=rng.randint(0, 2 ** 31) * 7919 + s * 100 + k).int)[:30]
            img, nods, vess, rnd = make_scan(rng, flip=(k == 1))
            sitk.WriteImage(img, os.path.join(args.out, f"subset{s}", uid + ".mhd"))
            for c, d in nods:
                A.append((uid, *c, d))
                for _ in range(rng.randint(1, 3)):
                    Cn.append((uid, *(c + rng.normal(0, 0.7, 3)), 1))
            for p in vess + rnd:
                Cn.append((uid, *p, 0))
            E.append((uid, *rnd[0], -1))
            print("wrote", s, uid[-12:], "nodules:", len(nods))
    pd.DataFrame(A, columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]).to_csv(
        os.path.join(args.out, "annotations.csv"), index=False)
    pd.DataFrame(Cn, columns=["seriesuid", "coordX", "coordY", "coordZ", "class"]).to_csv(
        os.path.join(args.out, "candidates_V2.csv"), index=False)
    pd.DataFrame(E, columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]).to_csv(
        os.path.join(args.out, "annotations_excluded.csv"), index=False)


if __name__ == "__main__":
    main()
