"""Quick visual + numeric check of the preprocessed patches (also a figure for the paper).

    python show_patches.py      -> work/patch_check.png  +  printed checks
"""
import os

import numpy as np
import pandas as pd

import config as C


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = pd.read_csv(os.path.join(C.PATCH_DIR, "meta.csv"))
    arr = np.load(os.path.join(C.PATCH_DIR, "patches.npy"), mmap_mode="r")
    done = set(open(os.path.join(C.PATCH_DIR, "done_scans.txt")).read().split())
    missing = sorted(set(meta.seriesuid) - done)
    print(f"scans finished: {len(done)} / {meta.seriesuid.nunique()}  (missing: {len(missing)})")

    # every patch row must have been written (an unwritten row is all zeros)
    empty = 0
    for s in range(0, len(arr), 2000):
        blk = np.asarray(arr[s:s + 2000])
        empty += int((blk.reshape(len(blk), -1) == 0).all(1).sum())
    print(f"empty (never written) patches: {empty}   <- must be 0")

    rng = np.random.RandomState(0)
    pos = meta[meta.label == 1].sample(8, random_state=rng)
    big = meta[(meta.label == 1) & (meta.diameter_mm >= 10)]
    c = C.STORE_SIZE // 2
    cen_pos = np.mean([np.asarray(arr[r])[c - 1:c + 1, c - 1:c + 1, c - 1:c + 1].mean() for r in big.row[:200]])
    neg = meta[meta.label == 0].sample(8, random_state=rng)
    print(f"mean HU at the exact centre of nodules >= 10 mm: {cen_pos:.0f}   <- should be about -100 or higher")

    fig, axs = plt.subplots(2, 8, figsize=(16, 4.4), dpi=110)
    for j, (df, name) in enumerate(((pos, "nodule"), (neg, "not nodule"))):
        for i, (_, r) in enumerate(df.iterrows()):
            sl = np.asarray(arr[int(r.row)])[c]
            axs[j, i].imshow(np.clip((sl + 1350) / 1500, 0, 1), cmap="gray")
            axs[j, i].axis("off")
            t = f"{name}" + (f"\n{r.diameter_mm:.1f} mm" if r.label == 1 else "")
            axs[j, i].set_title(t, fontsize=8)
    fig.suptitle("Centre slice of 48 mm patches (nodules should sit in the middle of the top row)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(C.WORK_DIR, "patch_check.png")
    fig.savefig(out)
    print("->", out)


if __name__ == "__main__":
    main()
