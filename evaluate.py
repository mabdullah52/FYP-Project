"""Step 3 - evaluate a trained run on its held-out LUNA16 subset (ALL candidates).

    python evaluate.py --run msgf_f9
    python evaluate.py --run abl_nocontext --no-tta --mc 1     # plain single-pass numbers

Writes to WORK_DIR/runs/<run>/eval_subset<k>/:
    predictions.csv  (LUNA16 submission format + uncertainty, diameter, gate alpha)
    metrics.json     CPM (+95% CI), sensitivity@FP/scan, AUC, AP, ECE, diameter MAE
    froc.png, reliability.png, triage.png, triage.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
from luna_io import index_scans, iter_scan_patches
from metrics import cpm_with_ci, expected_calibration_error, triage_curve
from predictor import Predictor, platt

INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def score_subsets(pred, scan_index, cands, subsets, workers, max_scans=None, log_every=10):
    """Score every candidate of the given subsets. One process; `workers` = extraction threads."""
    c = cands[cands.seriesuid.map(lambda s: scan_index.get(s, (None, -1))[1]).isin(subsets)]
    groups = list(c.groupby("seriesuid"))[: max_scans or None]
    jobs = [(u, scan_index[u][0], g[["coordX", "coordY", "coordZ"]].to_numpy()) for u, g in groups]
    rows_of = {u: g for u, g in groups}
    outs, t0 = [], time.time()
    for k, (uid, patches, err) in enumerate(iter_scan_patches(jobs, workers), 1):
        if err is not None:
            raise RuntimeError(f"could not read scan {uid}: {err}")
        s = pred.score(patches)
        g = rows_of[uid].copy()
        g["probability"] = s["prob"]
        g["uncertainty"] = s["uncertainty"]
        g["entropy"] = s["entropy"]
        g["diameter_pred_mm"] = s["diameter_mm"]
        g["alpha"] = s["alpha"]
        outs.append(g)
        if k % log_every == 0 or k == len(jobs):
            el = time.time() - t0
            print(f"  {k}/{len(jobs)} scans, {el / 60:.1f} min, ~{el / k * (len(jobs) - k) / 60:.1f} min left")
    return pd.concat(outs).reset_index(drop=True), [u for u, _ in groups]


def plot_triage(p, y, unc, path, max_pct=10.0):
    """Share of the model's errors (decision at p = 0.5) that reach the radiologist when the
    most-uncertain candidates are referred, against referring the same number at random."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, y, unc = np.asarray(p), np.asarray(y), np.asarray(unc)
    err = ((p >= 0.5) != (y == 1))
    order = np.argsort(-unc, kind="stable")
    caught = np.concatenate([[0], np.cumsum(err[order])]) / max(1, err.sum())
    pct = np.arange(len(p) + 1) / len(p) * 100
    m = (pct >= 0.01) & (pct <= max_pct)
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=200)
    ax.plot(pct[m], caught[m] * 100, color=BLUE, lw=2, label="Most uncertain first (MSGF-Net)")
    ax.plot(pct[m], pct[m], color=INK2, lw=1.2, ls="--", label="Random referral")
    lines = []
    for q in (1.0, 2.0, 5.0):
        k = int(round(q / 100 * len(p)))
        ax.plot([q], [caught[k] * 100], "o", color=ORANGE, ms=5, markeredgecolor="white", zorder=3)
        lines.append(f"refer {q:g}%  ->  {caught[k] * 100:.0f}% of errors caught")
    ax.text(0.03, 0.70, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.01, max_pct); ax.set_ylim(0, 102)
    ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlabel("Most-uncertain candidates referred to radiologist (%)", color=INK)
    ax.set_ylabel("Model errors caught (%)", color=INK)
    ax.set_title(f"Uncertainty-aware triage  ({int(err.sum())} errors in {len(p):,} candidates)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return {f"errors_caught_at_{q:g}pct": float(caught[int(round(q / 100 * len(p)))]) for q in (0.5, 1.0, 2.0, 5.0, 10.0)}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plots(out_dir, froc_curve, metrics, rel_rows, triage_data):
    p_all, y_all, unc_all = triage_data
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fppi, sens = froc_curve
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=200)
    ax.plot(np.clip(fppi, 0.1, None), sens, color=BLUE, lw=2, drawstyle="steps-post")
    rates = [float(r) for r in metrics["froc"]["sensitivity_at_fp"]]
    ax.plot(rates, list(metrics["froc"]["sensitivity_at_fp"].values()), "o", color=BLUE, ms=5,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(rates, ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_xlim(0.1, 10); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Average false positives per scan", color=INK)
    ax.set_ylabel("Sensitivity", color=INK)
    ax.set_title(f"FROC  (CPM = {metrics['froc']['cpm']:.3f})", color=INK, fontsize=11, loc="left")
    _style(ax); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "froc.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Perfect calibration")
    if rel_rows:
        conf, acc, _ = zip(*rel_rows)
        ax.plot(conf, acc, "o-", color=BLUE, lw=2, ms=5, markeredgecolor="white", label="MSGF-Net")
    ax.set_xlabel("Mean predicted probability", color=INK); ax.set_ylabel("Observed nodule fraction", color=INK)
    ax.set_title(f"Reliability  (ECE = {metrics['ece']:.4f})", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "reliability.png")); plt.close(fig)

    plot_triage(p_all, y_all, unc_all, os.path.join(out_dir, "triage.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--subset", type=int, default=None, help="default: the run's test subset")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--mc", type=int, default=C.MC_SAMPLES)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2), help="extraction threads")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-scans", type=int, default=None, help="debug: only the first N scans")
    ap.add_argument("--tag", default="")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit Platt scaling on ALL candidates of the validation subset (saved for the apps)")
    args = ap.parse_args()

    run_dir = os.path.join(C.RUNS_DIR, args.run)
    pred = Predictor(os.path.join(run_dir, "best.pt"), tta=not args.no_tta, mc=args.mc, apply_calibration=False)
    subset = pred.cfg["test_subset"] if args.subset is None else args.subset
    out_dir = os.path.join(run_dir, f"eval_subset{subset}{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    scan_index = index_scans()
    cands = pd.read_csv(C.CANDIDATES_CSV)
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if os.path.exists(C.EXCLUDED_CSV) else None
    if excl is None:
        print("NOTE: annotations_excluded.csv not found -> irrelevant findings count as FPs (CPM slightly lower)")

    cal_path = os.path.join(run_dir, "calibration.json")
    if args.calibrate:
        from sklearn.linear_model import LogisticRegression
        vs = pred.cfg["val_subset"]
        print(f"Calibration: scoring every candidate of validation subset {vs} ...")
        dv, _ = score_subsets(pred, scan_index, cands, [vs], args.workers, args.max_scans)
        pv = np.clip(dv.probability.to_numpy(), 1e-6, 1 - 1e-6)
        lr = LogisticRegression(C=1e4).fit(np.log(pv / (1 - pv))[:, None], dv["class"].to_numpy())
        cal = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "fitted_on_subset": vs,
               "n_candidates": int(len(dv))}
        json.dump(cal, open(cal_path, "w"), indent=2)
        print("calibration:", cal)
        # keep the validation predictions: they are used for model selection (validation CPM)
        vdir = os.path.join(run_dir, f"eval_subset{vs}{args.tag}")
        os.makedirs(vdir, exist_ok=True)
        dv["probability_raw"] = dv.probability
        dv["probability"] = platt(dv.probability_raw.to_numpy(), cal)
        dv.to_csv(os.path.join(vdir, "predictions.csv"), index=False)
        v_uids = [u for u, (_, s_) in scan_index.items() if s_ == vs]
        vfroc, _ = cpm_with_ci(dv, ann[ann.seriesuid.isin(v_uids)], excl, v_uids, n_boot=200)
        json.dump({"run": args.run, "subset": vs, "tta": not args.no_tta, "mc": args.mc, "role": "validation",
                   "n_candidates": int(len(dv)), "froc": vfroc}, open(os.path.join(vdir, "metrics.json"), "w"), indent=2)
        print(f"validation CPM (subset {vs}): {vfroc['cpm']:.4f}")
    cal = json.load(open(cal_path)) if os.path.exists(cal_path) else None

    print(f"Scoring every candidate of subset {subset} (TTA={not args.no_tta}, MC={args.mc}) ...")
    df, uids = score_subsets(pred, scan_index, cands, [subset], args.workers, args.max_scans)
    df["probability_raw"] = df.probability
    if cal is not None:
        df["probability"] = platt(df.probability_raw.to_numpy(), cal)
    df.drop(columns=[c for c in ("subset",) if c in df]).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # every scan of the subset counts in FP/scan, including scans without candidates/nodules
    all_uids = uids if args.max_scans else [u for u, (_, s) in scan_index.items() if s == subset]
    froc, curve = cpm_with_ci(df, ann[ann.seriesuid.isin(all_uids)], excl, all_uids, n_boot=args.n_boot)

    y, p = df["class"].to_numpy(), df.probability.to_numpy()
    ece, rel = expected_calibration_error(p, y)
    ece_raw, _ = expected_calibration_error(df.probability_raw.to_numpy(), y)
    tri = triage_curve(p, y, df.uncertainty.to_numpy())
    tri.to_csv(os.path.join(out_dir, "triage.csv"), index=False)
    pos = df[(df["class"] == 1)]
    pos_d = pos.merge(ann, on="seriesuid", suffixes=("", "_a"))
    pos_d["dist"] = np.linalg.norm(pos_d[["coordX", "coordY", "coordZ"]].to_numpy()
                                   - pos_d[["coordX_a", "coordY_a", "coordZ_a"]].to_numpy(), axis=1)
    pos_d = pos_d[pos_d.dist <= pos_d.diameter_mm / 2].sort_values("dist").drop_duplicates(
        ["seriesuid", "coordX", "coordY", "coordZ"])
    metrics = {
        "run": args.run, "subset": subset, "tta": not args.no_tta, "mc": args.mc,
        "n_candidates": int(len(df)), "n_positive_candidates": int(y.sum()),
        "froc": froc,
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "ap": float(average_precision_score(y, p)) if y.sum() else None,
        "ece": ece, "ece_uncalibrated": ece_raw, "calibrated": cal is not None,
        "diameter_mae_mm": float(np.abs(pos_d.diameter_pred_mm - pos_d.diameter_mm).mean()) if len(pos_d) else None,
        "diameter_pearson_r": float(np.corrcoef(pos_d.diameter_pred_mm, pos_d.diameter_mm)[0, 1]) if len(pos_d) > 2 else None,
        "uncertainty_auc_for_errors": None,
    }
    err = ((p >= 0.5) != (y == 1)).astype(int)
    if 0 < err.sum() < len(err):
        metrics["uncertainty_auc_for_errors"] = float(roc_auc_score(err, df.uncertainty))
    if df.alpha.notna().any() and len(pos_d) > 2:
        metrics["alpha_vs_diameter_spearman"] = float(pos_d[["alpha", "diameter_mm"]].corr("spearman").iloc[0, 1])
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    plots(out_dir, curve, metrics, rel, (p, y, df.uncertainty.to_numpy()))
    print(json.dumps(metrics, indent=2))
    print("->", out_dir)


if __name__ == "__main__":
    main()
