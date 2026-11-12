"""Step 36 - 5-fold patient-level cross-validation on LUNA16 (Paper 1).

The 10 official subsets are paired into 5 folds, so every one of the 888 scans is tested
exactly once, by a model that never saw it:

    fold  test subsets  validation  training
      1      0, 1           2        3-9
      2      2, 3           4        0, 1, 5-9
      3      4, 5           6        0-3, 7-9
      4      6, 7           8        0-5, 9
      5      8, 9           0        1-7

For each fold: train (train.py, same settings as the paper), fit Platt calibration on ALL
candidates of the validation subset, then score ALL candidates of the two test subsets with
TTA + MC dropout. The report pools the 5 test folds into one FROC over all 888 scans.

    python cv5.py run --arch msgf             # proposed model, ~7 h per fold on a Quadro P4000
    python cv5.py run --arch cnn3d            # optional comparison: plain 3D CNN, ~19 h per fold
    python cv5.py run --arch msgf --smoke     # 10-minute check that everything works
    python cv5.py report                      # tables + FROC figure (any time, also mid-way)

Resumable: a finished fold, calibration or test subset is never redone.
Results -> WORK_DIR/cv5/ : cv5_summary.csv, cv5_folds.csv, cv5_comparison.csv, froc_cv5.png
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C

FOLDS = {1: ((0, 1), 2), 2: ((2, 3), 4), 3: ((4, 5), 6), 4: ((6, 7), 8), 5: ((8, 9), 0)}
ARCH_LABEL = {"msgf": "MSGF-Net", "cnn3d": "Plain 3D CNN", "resnet3d18": "3D ResNet-18", "resnet25d": "2.5D ResNet-18"}
OUT = os.path.join(C.WORK_DIR, "cv5")


def run_name(arch, k, smoke=False):
    return f"cv5{'smoke' if smoke else ''}_{arch}_f{k}"


def eval_dir(arch, k, subset, smoke=False):
    return os.path.join(C.RUNS_DIR, run_name(arch, k, smoke), f"eval_subset{subset}")


# ------------------------------------------------------------------ run one architecture
def train_fold(arch, k, smoke):
    tests, val = FOLDS[k]
    cmd = [sys.executable, "train.py", "--name", run_name(arch, k, smoke), "--arch", arch,
           "--test-subsets", ",".join(map(str, tests)), "--val-subset", str(val), "--resume"]
    if smoke:
        cmd += ["--epochs", "1", "--samples-per-epoch", "1024"]
    print(f"\n=== fold {k}/5 ({ARCH_LABEL.get(arch, arch)}): training, test subsets {tests}, validation subset {val}",
          flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"training of fold {k} stopped (exit code {r.returncode})")


def score_fold(arch, k, smoke, workers, tta, mc):
    from sklearn.linear_model import LogisticRegression
    from evaluate import score_subsets
    from luna_io import index_scans
    from predictor import Predictor, platt

    tests, val = FOLDS[k]
    run_dir = os.path.join(C.RUNS_DIR, run_name(arch, k, smoke))
    todo_tests = [s for s in tests if not os.path.exists(os.path.join(eval_dir(arch, k, s, smoke), "predictions.csv"))]
    cal_path = os.path.join(run_dir, "calibration.json")
    if not todo_tests and os.path.exists(cal_path):
        print(f"fold {k}: already scored")
        return
    max_scans = 2 if smoke else None
    pred = Predictor(os.path.join(run_dir, "best.pt"), tta=tta, mc=mc, apply_calibration=False)
    scan_index = index_scans()
    cands = pd.read_csv(C.CANDIDATES_CSV)

    if not os.path.exists(cal_path):
        print(f"fold {k}: calibration - scoring every candidate of validation subset {val} ...", flush=True)
        dv, _ = score_subsets(pred, scan_index, cands, [val], workers, max_scans)
        pv = np.clip(dv.probability.to_numpy(), 1e-6, 1 - 1e-6)
        if dv["class"].nunique() < 2:                     # only possible in the 2-scan smoke test
            print("only one class in these validation candidates -> identity calibration")
            a, b = 1.0, 0.0
        else:
            lr = LogisticRegression(C=1e4).fit(np.log(pv / (1 - pv))[:, None], dv["class"].to_numpy())
            a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        cal = {"a": a, "b": b, "fitted_on_subset": val, "n_candidates": int(len(dv)), "tta": tta, "mc": mc}
        vdir = eval_dir(arch, k, val, smoke)
        os.makedirs(vdir, exist_ok=True)
        dv["probability_raw"] = dv.probability
        dv["probability"] = platt(dv.probability_raw.to_numpy(), cal)
        dv.to_csv(os.path.join(vdir, "predictions.csv"), index=False)
        json.dump(cal, open(cal_path, "w"), indent=2)          # written last = calibration step done
        print("calibration:", cal)
    cal = json.load(open(cal_path))

    for s in todo_tests:
        print(f"fold {k}: scoring every candidate of test subset {s} (TTA={tta}, MC={mc}) ...", flush=True)
        df, _ = score_subsets(pred, scan_index, cands, [s], workers, max_scans)
        df["probability_raw"] = df.probability
        df["probability"] = platt(df.probability_raw.to_numpy(), cal)
        d = eval_dir(arch, k, s, smoke)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "predictions.tmp.csv")
        df.drop(columns=[c for c in ("subset",) if c in df]).to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(d, "predictions.csv"))   # only complete files count as done


def cmd_run(args):
    os.makedirs(OUT, exist_ok=True)
    folds = [int(f) for f in args.folds.split(",")]
    import torch
    if not torch.cuda.is_available() and not args.smoke and not args.allow_cpu:
        raise SystemExit("No NVIDIA GPU found. 5-fold training would take weeks on the CPU - run this on the "
                         "university PC (or pass --allow-cpu if you really mean it).")
    if args.smoke:
        folds = folds[:1]
        print("SMOKE TEST: 1 short epoch and 2 scans per subset - checks the pipeline, numbers mean nothing")
    for k in folds:
        train_fold(args.arch, k, args.smoke)
        score_fold(args.arch, k, args.smoke, args.workers, not args.no_tta, args.mc)
        print(f"=== fold {k}/5 done ===", flush=True)
    if args.smoke:
        print("\nSMOKE TEST PASSED - the full run can start.")
    else:
        cmd_report(args)


# ------------------------------------------------------------------ report
def _diameter_errors(df, ann):
    pos = df[df["class"] == 1].merge(ann, on="seriesuid", suffixes=("", "_a"))
    if pos.empty:
        return np.zeros(0), np.zeros(0)
    pos["dist"] = np.linalg.norm(pos[["coordX", "coordY", "coordZ"]].to_numpy()
                                 - pos[["coordX_a", "coordY_a", "coordZ_a"]].to_numpy(), axis=1)
    pos = pos[pos.dist <= pos.diameter_mm / 2].sort_values("dist").drop_duplicates(["seriesuid", "coordX", "coordY", "coordZ"])
    return pos.diameter_pred_mm.to_numpy(), pos.diameter_mm.to_numpy()


def collect(arch, scan_index, smoke=False):
    """Pooled test predictions of the finished folds of one architecture."""
    parts, done = [], []
    for k, (tests, _) in FOLDS.items():
        files = [os.path.join(eval_dir(arch, k, s, smoke), "predictions.csv") for s in tests]
        if all(os.path.exists(f) for f in files):
            for s, f in zip(tests, files):
                d = pd.read_csv(f)
                d["fold"], d["subset"] = k, s
                parts.append(d)
            done.append(k)
    if not parts:
        return None, []
    return pd.concat(parts, ignore_index=True), done


def cmd_report(args):
    from luna_io import index_scans
    from metrics import cpm_with_ci, expected_calibration_error, froc_from_hits, per_scan_hits

    os.makedirs(OUT, exist_ok=True)
    scan_index = index_scans()
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if C.EXCLUDED_CSV and os.path.exists(C.EXCLUDED_CSV) else None
    rates = [str(r) for r in C.FROC_FP_RATES]
    fold_rows, sum_rows, pooled, curves = [], [], {}, {}

    for arch in ARCH_LABEL:
        df, done = collect(arch, scan_index, getattr(args, "smoke", False))
        if df is None:
            continue
        subsets = sorted({s for k in done for s in FOLDS[k][0]})
        uids = sorted(u for u, (_, s) in scan_index.items() if s in subsets)
        if getattr(args, "smoke", False):
            uids = sorted(df.seriesuid.unique())
        a = ann[ann.seriesuid.isin(uids)]
        for k in done:
            fu = [u for u in uids if scan_index[u][1] in FOLDS[k][0]]
            f, _ = cpm_with_ci(df[df.fold == k], a[a.seriesuid.isin(fu)], excl, fu, n_boot=0)
            fold_rows.append({"model": ARCH_LABEL[arch], "fold": k, "test_subsets": "+".join(map(str, FOLDS[k][0])),
                              "val_subset": FOLDS[k][1], "n_scans": f["n_scans"], "n_nodules": f["n_nodules"],
                              "CPM": round(f["cpm"], 4),
                              **{f"sens@{r}": round(f["sensitivity_at_fp"][r], 3) for r in rates}})
        froc, curve = cpm_with_ci(df, a, excl, uids, n_boot=args.n_boot)
        curves[arch] = (curve, froc)
        pooled[arch] = (per_scan_hits(df, a, excl, uids), uids)
        y, p = df["class"].to_numpy(), df.probability.to_numpy()
        ece, _ = expected_calibration_error(p, y)
        ece_raw, _ = expected_calibration_error(df.probability_raw.to_numpy(), y)
        err = ((p >= 0.5) != (y == 1)).astype(int)
        order = np.argsort(-df.uncertainty.to_numpy(), kind="stable")
        caught5 = err[order[: int(round(0.05 * len(err)))]].sum() / max(1, err.sum())
        dp, dt = _diameter_errors(df, a)
        fcpm = [r["CPM"] for r in fold_rows if r["model"] == ARCH_LABEL[arch]]
        sum_rows.append({
            "model": ARCH_LABEL[arch], "folds_done": len(done), "n_scans": froc["n_scans"], "n_nodules": froc["n_nodules"],
            "n_candidates": int(len(df)), "CPM_pooled": round(froc["cpm"], 4),
            "CPM_95CI": f"{froc['cpm_ci95'][0]:.3f}-{froc['cpm_ci95'][1]:.3f}",
            "CPM_fold_mean": round(float(np.mean(fcpm)), 4), "CPM_fold_sd": round(float(np.std(fcpm, ddof=1)), 4) if len(fcpm) > 1 else None,
            **{f"sens@{r}": round(froc["sensitivity_at_fp"][r], 3) for r in rates},
            "AUC": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
            "AP": round(float(average_precision_score(y, p)), 4) if y.sum() else None,
            "ECE_calibrated": round(ece, 4), "ECE_uncalibrated": round(ece_raw, 4),
            "diameter_MAE_mm": round(float(np.abs(dp - dt).mean()), 3) if len(dt) else None,
            "diameter_r": round(float(np.corrcoef(dp, dt)[0, 1]), 3) if len(dt) > 2 else None,
            "uncertainty_AUROC_errors": round(float(roc_auc_score(err, df.uncertainty)), 4) if 0 < err.sum() < len(err) else None,
            "errors": int(err.sum()), "errors_in_top5pct_uncertain": round(float(caught5), 3),
        })

    if not sum_rows:
        print("No finished folds yet."); return
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "cv5_folds.csv"), index=False)
    summ = pd.DataFrame(sum_rows)
    summ.to_csv(os.path.join(OUT, "cv5_summary.csv"), index=False)

    comp = []
    if "msgf" in pooled:
        ha, ua = pooled["msgf"]
        cpm = lambda h: froc_from_hits(h)[0].mean()
        for arch in ARCH_LABEL:
            if arch == "msgf" or arch not in pooled or pooled[arch][1] != ua:
                continue
            hb = pooled[arch][0]
            rng, n = np.random.RandomState(0), len(ha)
            d = np.array([cpm([ha[j] for j in idx]) - cpm([hb[j] for j in idx])
                          for idx in (rng.randint(0, n, n) for _ in range(args.n_boot_paired))])
            comp.append({"comparison": f"MSGF-Net minus {ARCH_LABEL[arch]}", "n_scans": n,
                         "delta_CPM": round(cpm(ha) - cpm(hb), 4),
                         "delta_95CI": f"{np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}",
                         "p_value": round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 3)})
    pd.DataFrame(comp).to_csv(os.path.join(OUT, "cv5_comparison.csv"), index=False)
    plot(curves)
    json.dump({"summary": sum_rows, "folds": fold_rows, "comparison": comp},
              open(os.path.join(OUT, "cv5_summary.json"), "w"), indent=2)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print("\nPER FOLD\n", pd.DataFrame(fold_rows).to_string(index=False))
        print("\nPOOLED OVER ALL TESTED SCANS\n", summ.T.to_string())
        if comp:
            print("\nPAIRED COMPARISON\n", pd.DataFrame(comp).to_string(index=False))
    print("->", OUT)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"msgf": "#2a78d6", "cnn3d": "#eb6834", "resnet3d18": "#52514e", "resnet25d": "#8a8780"}
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=200)
    for arch, ((fppi, sens), froc) in curves.items():
        ax.plot(np.clip(fppi, 0.1, None), sens, color=colors.get(arch, "#333"), lw=2, drawstyle="steps-post",
                label=f"{ARCH_LABEL[arch]} (CPM {froc['cpm']:.3f})")
    rates = list(C.FROC_FP_RATES)
    ax.set_xscale("log", base=2); ax.set_xticks(rates, ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_xlim(0.1, 10); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Average false positives per scan"); ax.set_ylabel("Sensitivity")
    n = next(iter(curves.values()))[1]["n_scans"]
    ax.set_title(f"FROC, 5-fold cross-validation ({n} scans)", fontsize=10, loc="left")
    ax.grid(True, color="#e4e3df"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "froc_cv5.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arch", default="msgf", choices=list(ARCH_LABEL))
    r.add_argument("--folds", default="1,2,3,4,5")
    r.add_argument("--smoke", action="store_true", help="1 short epoch, 2 scans: checks the pipeline only")
    r.add_argument("--allow-cpu", action="store_true")
    r.add_argument("--no-tta", action="store_true")
    r.add_argument("--mc", type=int, default=C.MC_SAMPLES)
    r.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    for p in (r, sub.add_parser("report")):
        p.add_argument("--n-boot", type=int, default=1000)
        p.add_argument("--n-boot-paired", type=int, default=2000)
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
