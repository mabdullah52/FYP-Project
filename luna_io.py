"""Reading LUNA16 scans and extracting isotropic 3D patches.

Key points (these are where most LUNA16 projects silently go wrong):
  * .mhd/.raw is read with SimpleITK; the numpy array is indexed (z, y, x).
  * CSV coordinates are WORLD coordinates in mm (x, y, z). They are converted to
    voxel indices with origin, spacing AND direction (some scans are flipped).
  * Every patch is resampled to 1 mm isotropic voxels directly from the original
    volume with trilinear interpolation, so no full-volume resampling is needed and
    scans with 0.6 mm or 2.5 mm slices produce patches of identical physical size.
"""
import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

import config as C


def index_scans(root=C.LUNA16_ROOT):
    """Return {seriesuid: (mhd_path, subset_int)} for every scan under root/subset*/."""
    out = {}
    for mhd in glob.glob(os.path.join(root, "subset*", "**", "*.mhd"), recursive=True):
        parts = [p for p in mhd.replace("\\", "/").split("/") if p.startswith("subset")]
        subset = int("".join(ch for ch in parts[-1] if ch.isdigit()))
        out[os.path.splitext(os.path.basename(mhd))[0]] = (mhd, subset)
    if not out:
        raise FileNotFoundError(f"No .mhd files found under {root}/subset*/ - check LUNA16_ROOT")
    return out


class Scan:
    """A CT volume in HU with the geometry needed to map world mm -> array index."""

    def __init__(self, mhd_path):
        """mhd_path: a .mhd file, any SimpleITK-readable volume (.nii.gz, .nrrd) or a DICOM folder."""
        if os.path.isdir(mhd_path):
            reader = sitk.ImageSeriesReader()
            ids = reader.GetGDCMSeriesIDs(mhd_path)
            if not ids:
                raise ValueError(f"no DICOM series in {mhd_path}")
            # largest series in the folder = the CT volume
            files = max((reader.GetGDCMSeriesFileNames(mhd_path, i) for i in ids), key=len)
            reader.SetFileNames(files)
            img = reader.Execute()
        else:
            img = sitk.ReadImage(mhd_path)
        self.path = mhd_path
        if img.GetPixelID() != sitk.sitkInt16:
            img = sitk.Cast(img, sitk.sitkInt16)
        # zero-copy view on the image buffer (halves memory for 700-slice scans)
        self.array = sitk.GetArrayViewFromImage(img)                       # (z, y, x) HU
        self.origin = np.array(img.GetOrigin(), dtype=np.float64)        # (x, y, z) mm
        self.spacing = np.array(img.GetSpacing(), dtype=np.float64)      # (x, y, z) mm
        self.direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
        self._dir_inv = np.linalg.inv(self.direction)
        self.image = img

    # ----- coordinate transforms --------------------------------------------------
    def world_to_index_xyz(self, world_xyz):
        """world (..., 3) mm -> continuous voxel index (..., 3) in (x, y, z) order."""
        rel = np.asarray(world_xyz, dtype=np.float64) - self.origin
        return (rel @ self._dir_inv.T) / self.spacing

    def index_xyz_to_world(self, idx_xyz):
        idx = np.asarray(idx_xyz, dtype=np.float64) * self.spacing
        return idx @ self.direction.T + self.origin

    # ----- patch extraction ------------------------------------------------------
    def extract_patch(self, center_world_xyz, size=C.STORE_SIZE, spacing_mm=C.TARGET_SPACING_MM):
        """Cube of size^3 voxels at `spacing_mm`, centred on a world point. Returns int16 (z, y, x)."""
        half = (size - 1) / 2.0
        offs = (np.arange(size) - half) * spacing_mm
        # grid of world points in patient axes, ordered (z, y, x) for the output array
        oz, oy, ox = np.meshgrid(offs, offs, offs, indexing="ij")
        pts = np.stack([ox, oy, oz], axis=-1) + np.asarray(center_world_xyz)   # (..., 3) xyz
        idx = self.world_to_index_xyz(pts)                                    # xyz index
        coords = np.stack([idx[..., 2], idx[..., 1], idx[..., 0]], axis=0)    # -> z, y, x
        patch = map_coordinates(self.array, coords, order=1, mode="constant",
                                cval=C.PAD_HU, prefilter=False)
        return np.clip(patch, *C.HU_CLIP).astype(np.int16)

    def extract_many(self, centers_world_xyz, size=C.STORE_SIZE, executor=None):
        """executor: optional ThreadPoolExecutor - the interpolation releases the GIL, so
        threads give a real speed-up without extra processes (important on Windows)."""
        if not len(centers_world_xyz):
            return np.zeros((0, size, size, size), np.int16)
        f = lambda c: self.extract_patch(c, size)
        out = list(executor.map(f, centers_world_xyz)) if executor else [f(c) for c in centers_world_xyz]
        return np.stack(out)


def iter_scan_patches(jobs, threads=8):
    """jobs: list of (uid, scan_path, centers). Yields (uid, patches, error) one scan at a time.
    Single process: the next scan is read in the background while patches of the current
    scan are cut with `threads` threads. Memory stays at ~2 scans."""
    from concurrent.futures import ThreadPoolExecutor

    def load(job):
        try:
            return Scan(job[1]), None
        except Exception as e:  # corrupted / partial file: report and continue
            return None, repr(e)

    if not jobs:
        return
    with ThreadPoolExecutor(1) as loader, ThreadPoolExecutor(max(1, threads)) as pool:
        nxt = loader.submit(load, jobs[0])
        for i, (uid, _, centers) in enumerate(jobs):
            scan, err = nxt.result()
            if i + 1 < len(jobs):
                nxt = loader.submit(load, jobs[i + 1])
            if err is None:
                try:
                    yield uid, scan.extract_many(centers, executor=pool), None
                except Exception as e:
                    yield uid, None, repr(e)
            else:
                yield uid, None, err
            del scan


def hu_to_windows(patch_hu, windows=C.HU_WINDOWS):
    """int16/float HU array (..., D, H, W) -> float32 (..., n_windows, D, H, W) in [0, 1]."""
    x = np.asarray(patch_hu, dtype=np.float32)
    chans = [np.clip((x - lo) / (hi - lo), 0.0, 1.0) for lo, hi in windows]
    return np.stack(chans, axis=-4)
