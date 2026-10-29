"""Step 30 - figures for paper 2 from results that already exist (no training, about 1-2 minutes).

    python paper2_figures.py   -> work/paper2_figures/{roc_single.png, auc_by_size.png, roc_temporal.png, numbers.json}

Reads work/luna25_final/test_predictions.csv (step 14) and work/luna25_temporal/test_pair_predictions.csv (step 15/15B).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

import config as C  # noqa: E402

OUT = os.path.join(C.WORK_DIR, "paper2_figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"     # validated categorical slots 1-4
DASH = {S1: "-", S2: "--", S3: "-.", S4: ":"}                    # secondary encoding (not colour alone)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def boot_auc(y, p, g, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    pats = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in pats}
    v = []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(pats, len(pats))])
        if 0 < y[s].sum() < len(s):
            v.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def roc_plot(y, curves, path, title):
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=220)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (2, 3)))
    nums = {}
    for name, p, col in curves:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        nums[name] = round(float(auc), 4)
        ax.plot(fpr, tpr, color=col, lw=2, ls=DASH[col], label=f"{name}  (AUC {auc:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("1 - specificity", color=INK); ax.set_ylabel("Sensitivity", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(ax); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return nums


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    # ---- single-scan malignancy (all LUNA25 test nodules)
    f = os.path.join(C.WORK_DIR, "luna25_final", "test_predictions.csv")
    te = pd.read_csv(f, dtype={"patient": str})
    y, g = te.label.to_numpy(), te.patient.astype(str).to_numpy()
    res["roc_single"] = roc_plot(
        y, [("3D model (ensemble)", te.ensemble.to_numpy(), S1), ("Nodule size only", te.size_only.to_numpy(), S2)],
        os.path.join(OUT, "roc_single.png"), f"Malignancy, LUNA25 test ({len(te)} nodules, {int(y.sum())} malignant)")

    # ---- AUC by nodule size (grouped bars with 95% CI)
    bins = pd.cut(te.diameter_pred_mm, [0, 6, 8, 15, 100], right=False, labels=["<6", "6-8", "8-15", "≥15"])
    labels, rows = list(bins.cat.categories), []
    for b in labels:
        m = (bins == b).to_numpy()
        if m.sum() == 0 or not (0 < y[m].sum() < m.sum()):
            rows.append(None); continue
        r = {"bin": b, "n": int(m.sum()), "malignant": int(y[m].sum())}
        for key, col in (("ensemble", "ensemble"), ("size", "size_only")):
            p = te[col].to_numpy()[m]
            r[key] = float(roc_auc_score(y[m], p)); r[key + "_ci"] = boot_auc(y[m], p, g[m], n=1000)
        rows.append(r)
    res["auc_by_size"] = [r for r in rows if r]
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=220)
    x = np.arange(len(labels)); w = 0.36
    for j, (key, col, name) in enumerate((("ensemble", S1, "3D model"), ("size", S2, "Size only"))):
        labelled = False
        for i, r in enumerate(rows):
            if r is None:
                continue
            lo, hi = r[key + "_ci"]
            xi = x[i] + (j - 0.5) * (w + 0.04)
            ax.bar(xi, r[key], w, color=col, label=None if labelled else name, edgecolor="white", linewidth=1)
            labelled = True
            ax.errorbar(xi, r[key], yerr=[[r[key] - lo], [hi - r[key]]], color=INK2, lw=1, capsize=2)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (2, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} mm\nn={r['n']}, {r['malignant']} mal." if r else f"{b} mm" for b, r in zip(labels, rows)],
                       fontsize=7)
    ax.set_ylim(0.3, 1.12); ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_ylabel("Test AUC (95% CI)", color=INK)
    ax.set_title("Malignancy AUC by predicted nodule size", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    style(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "auc_by_size.png")); plt.close(fig)

    # ---- growth-aware models on nodules that have a prior scan
    f = os.path.join(C.WORK_DIR, "luna25_temporal", "test_pair_predictions.csv")
    if os.path.exists(f):
        tp = pd.read_csv(f, dtype={"patient": str})
        yt = tp.label.to_numpy()
        want = [("Single scan (ensemble)", "p_single_scan_ensemble", S1),
                ("Growth formula", "p_handcrafted_growth", S2),
                ("TG-MSGF (learned)", "p_TG-MSGF ensemble", S3),
                ("Combined (stacked)", "p_stacked: single + TG-MSGF + growth", S4)]
        curves = [(n, tp[c].to_numpy(), col) for n, c, col in want if c in tp]
        res["roc_temporal"] = roc_plot(yt, curves, os.path.join(OUT, "roc_temporal.png"),
                                       f"Nodules with a prior scan ({len(tp)}, {int(yt.sum())} malignant)")
    else:
        print("no temporal predictions found (run step 15B first) - skipping roc_temporal.png")
    json.dump(res, open(os.path.join(OUT, "numbers.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
