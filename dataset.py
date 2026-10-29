"""PyTorch dataset over the preprocessed patch memmap + balanced sampler."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

import config as C
from luna_io import hu_to_windows


def centre_crop(x, size):
    s = (x.shape[-1] - size) // 2
    return x[..., s:s + size, s:s + size, s:s + size]


def make_inputs(patch_hu, windows, use_context=True):
    """48^3 HU patch -> (local, context) float tensors with one channel per HU window."""
    ctx = hu_to_windows(patch_hu, windows)                 # (W, 48, 48, 48)
    loc = centre_crop(ctx, C.LOCAL_SIZE)                   # (W, 32, 32, 32)
    loc = torch.from_numpy(np.ascontiguousarray(loc))
    ctx = torch.from_numpy(np.ascontiguousarray(ctx)) if use_context else torch.zeros(0)
    return loc, ctx


def augment(p, rng, max_shift=3):
    """Label-preserving 3D augmentation on an int16 HU cube (z, y, x)."""
    # random translation (nodule stays inside the 32 mm local view)
    if max_shift:
        pad = np.pad(p, max_shift, mode="constant", constant_values=C.PAD_HU)
        dz, dy, dx = rng.randint(0, 2 * max_shift + 1, size=3)
        s = p.shape[0]
        p = pad[dz:dz + s, dy:dy + s, dx:dx + s]
    for ax in range(3):
        if rng.rand() < 0.5:
            p = np.flip(p, ax)
    p = np.rot90(p, rng.randint(4), axes=(1, 2))            # in-plane rotation
    if rng.rand() < 0.5:                                    # small HU calibration jitter
        p = p.astype(np.float32) * rng.uniform(0.97, 1.03) + rng.uniform(-20, 20)
    return np.ascontiguousarray(p)


class PatchDataset(Dataset):
    def __init__(self, meta, arr_path, windows=C.HU_WINDOWS, train=False, use_context=True):
        self.meta = meta.reset_index(drop=True)
        if "file" not in self.meta:
            self.meta["file"] = "patches.npy"
        self.meta["file"] = self.meta["file"].fillna("patches.npy")
        self.files = self.meta["file"].to_numpy()
        self.arr_dir = os.path.dirname(arr_path)
        self.arr = None                                     # opened lazily per worker
        self.windows, self.train, self.use_context = windows, train, use_context
        self.labels = self.meta.label.to_numpy().astype(np.float32)
        d = self.meta.diameter_mm.to_numpy().astype(np.float32)
        self.diam = np.nan_to_num(d, nan=0.0)
        self.diam_mask = (~np.isnan(d)) & (self.labels == 1)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = {f: np.load(os.path.join(self.arr_dir, f), mmap_mode="r") for f in set(self.files)}
        p = np.asarray(self.arr[self.files[i]][int(self.meta.row[i])])
        if self.train:
            p = augment(p, np.random)
        loc, ctx = make_inputs(p, self.windows, self.use_context)
        return loc, ctx, self.labels[i], self.diam[i], float(self.diam_mask[i])


class BalancedEpochSampler(Sampler):
    """Each epoch draws `n` indices: pos_fraction positives (with replacement - there are
    only ~1.5k), the rest negatives without replacement (different ones every epoch)."""

    def __init__(self, labels, n, pos_fraction, seed=0):
        self.pos = np.where(labels == 1)[0]
        self.neg = np.where(labels == 0)[0]
        self.n, self.pf = n, pos_fraction
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        n_pos = int(self.n * self.pf)
        n_neg = self.n - n_pos
        pos = self.rng.choice(self.pos, n_pos, replace=True)
        neg = self.rng.choice(self.neg, n_neg, replace=n_neg > len(self.neg))
        idx = np.concatenate([pos, neg])
        self.rng.shuffle(idx)
        return iter(idx.tolist())


def split_meta(meta_path, test_subset, val_subset, extra_meta=None):
    meta = pd.read_csv(meta_path)
    if extra_meta is not None and os.path.exists(extra_meta):      # model-mined hard negatives
        hard = pd.read_csv(extra_meta)
        hard = hard[~hard.subset.isin([test_subset, val_subset])]  # never mine from val/test
        meta = pd.concat([meta, hard], ignore_index=True)
    tr = meta[~meta.subset.isin([test_subset, val_subset])]
    va = meta[meta.subset == val_subset]
    te = meta[meta.subset == test_subset]
    # hard guarantee against patient leakage
    assert not (set(tr.seriesuid) & set(va.seriesuid)) and not (set(tr.seriesuid) & set(te.seriesuid))
    return tr, va, te


class ThreadLoader:
    """Drop-in replacement for torch DataLoader that uses THREADS instead of worker
    processes. On Windows every DataLoader worker is a full new Python process
    (re-imports torch, ~1-2 GB each); threads share one process and the heavy numpy /
    scipy work releases the GIL, so loading is just as fast without the memory cost."""

    def __init__(self, ds, batch_size, sampler=None, shuffle=False, drop_last=False, threads=8, prefetch=4):
        self.ds, self.bs, self.sampler = ds, batch_size, sampler
        self.shuffle, self.drop_last = shuffle, drop_last
        self.threads, self.prefetch = max(1, threads), prefetch

    def _indices(self):
        if self.sampler is not None:
            return list(iter(self.sampler))
        idx = np.arange(len(self.ds))
        if self.shuffle:
            np.random.shuffle(idx)
        return idx.tolist()

    def _batches(self):
        idx = self._indices()
        n = len(idx) // self.bs if self.drop_last else -(-len(idx) // self.bs)
        return [idx[i * self.bs:(i + 1) * self.bs] for i in range(n)]

    def __len__(self):
        n = len(self.sampler) if self.sampler is not None else len(self.ds)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def _collate(self, items):
        loc, ctx, y, d, dm = zip(*items)
        return (torch.stack(loc), torch.stack(ctx), torch.tensor(y, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32), torch.tensor(dm, dtype=torch.float32))

    def _get(self, fut, i):
        """Result of one item; if the PC is short of RAM, wait and retry instead of crashing."""
        import gc
        import time
        try:
            return fut.result()
        except MemoryError:
            for k in range(10):
                gc.collect(); time.sleep(3 * (k + 1))
                try:
                    return self.ds[i]
                except MemoryError:
                    pass
            raise

    def _collate_safe(self, items):
        import gc
        import time
        for k in range(10):
            try:
                return self._collate(items)
            except (MemoryError, RuntimeError) as e:
                if "memory" not in str(e).lower() and not isinstance(e, MemoryError):
                    raise
                gc.collect(); time.sleep(3 * (k + 1))
        return self._collate(items)

    def __iter__(self):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        batches = self._batches()
        with ThreadPoolExecutor(self.threads) as ex:
            q = deque()
            submit = lambda b: [(ex.submit(self.ds.__getitem__, i), i) for i in b]
            it = iter(batches)
            for b in it:
                q.append(submit(b))
                if len(q) >= self.prefetch:
                    break
            for b in it:
                futs = q.popleft()
                q.append(submit(b))
                yield self._collate_safe([self._get(f, i) for f, i in futs])
            while q:
                yield self._collate_safe([self._get(f, i) for f, i in q.popleft()])
