"""LUNA25 step 3 (part B) - nodule MALIGNANCY risk model (benign vs malignant, NLST pathology).

    python luna25_malignancy.py train --name l25_transfer --init msgf_f9    # proposed: start from LUNA16 MSGF-Net
    python luna25_malignancy.py train --name l25_scratch                     # comparison: random init
    python luna25_malignancy.py eval  --name l25_transfer                    # val calibration + test metrics
    python luna25_malignancy.py table                                        # final table + paired test

Same MSGF-Net architecture, multi-window input, TTA + MC-dropout uncertainty and Platt calibration
(fitted on the validation fold) as the detection model. Split is patient-level (luna25_prepare.py).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import config as C
from dataset import BalancedEpochSampler, PatchDataset, ThreadLoader
from luna25_common import L25_DIR
from metrics import expected_calibration_error, triage_curve
from model import build_model
from predictor import Predictor, platt
from train import MultiTaskLoss, fit_batch_size, validate


def cv_roles(fold, k):
    """Cross-validation (step 31): fold k = test, fold k+1 = validation, the other three = training.
    k = 0 is the original split (fold 0 test, fold 1 validation)."""
    fold = np.asarray(fold)
    return np.where(fold == k, "test", np.where(fold == (k + 1) % 5, "val", "train"))


def load_meta():
    meta = pd.read_csv(os.path.join(L25_DIR, "meta.csv"), dtype={"seriesuid": str, "patient": str})
    done = set(open(os.path.join(L25_DIR, "done_scans.txt")).read().split())
    meta = meta[meta.seriesuid.isin(done)].reset_index(drop=True)
    k = int(os.environ.get("L25_CV_FOLD", "0") or 0)
    if k:
        meta["role"] = cv_roles(meta.fold, k)
    return meta


# ----------------------------------------------------------------------------------------- train
def ram_report():
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        print(f"RAM available {vm.available / 1e9:.1f} of {vm.total / 1e9:.1f} GB | paging file {sw.total / 1e9:.1f} GB")
        if vm.available < 4e9:
            print("WARNING: less than 4 GB RAM free - close other programs (browser, other Python windows) or restart the PC")
    except ImportError:
        pass


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    run_dir = os.path.join(C.RUNS_DIR, args.name)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "DONE")):
        print(f"{args.name} already trained (delete {run_dir}\\DONE to retrain)"); return
    meta = load_meta()
    tr, va = meta[meta.role == "train"], meta[meta.role == "val"]
    arr_path = os.path.join(L25_DIR, "patches.npy")
    windows = C.HU_WINDOWS
    cfg = dict(in_ch=len(windows), width=C.TRAIN["width"], dropout=C.TRAIN["dropout"], use_context=True,
               fusion="gated", windows=windows, task="malignancy", init_from=args.init, args=vars(args),
               test_subset="luna25_fold0", val_subset="luna25_fold1")
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print(f"train {len(tr)} nodules ({int(tr.label.sum())} malignant) | val {len(va)} ({int(va.label.sum())}) | device {device}")
    ram_report()

    ck = None
    if args.init:      # transfer: same architecture as the LUNA16 model we start from
        ck = torch.load(os.path.join(C.RUNS_DIR, args.init, "best.pt"), map_location=device, weights_only=False)
        for k in ("in_ch", "width", "dropout", "use_context", "fusion", "windows"):
            cfg[k] = ck["cfg"][k]
        windows = [tuple(w) for w in cfg["windows"]]
        json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    model = build_model(cfg).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"])
        print(f"initialised from LUNA16 model '{args.init}' (transfer learning)")
    bs = fit_batch_size(cfg, args.batch_size, len(windows), True, device) if device.type == "cuda" else args.batch_size
    accum = max(1, args.effective_batch // bs)       # gradient accumulation: same effective batch, less memory
    print(f"batch {bs} x {accum} accumulation steps = effective batch {bs * accum}")
    ds_tr = PatchDataset(tr, arr_path, windows, train=True)
    ds_va = PatchDataset(va, arr_path, windows, train=False)
    sampler = BalancedEpochSampler(ds_tr.labels, args.samples_per_epoch, args.pos_fraction, args.seed)
    # small prefetch + same batch for validation: keeps RAM use low (the PC has a small paging file)
    dl_tr = ThreadLoader(ds_tr, bs, sampler=sampler, drop_last=True, threads=args.workers, prefetch=2)
    dl_va = ThreadLoader(ds_va, bs, threads=args.workers, prefetch=2)
    crit = MultiTaskLoss("focal", C.TRAIN["focal_gamma"], C.TRAIN["focal_alpha"], aux_weight=0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.TRAIN["weight_decay"])
    per_ep = max(1, len(dl_tr) // accum)
    steps, warm = args.epochs * per_ep, per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=C.amp_enabled(device))
    best, bad, hist, start = -1, 0, [], 0
    last = os.path.join(run_dir, "last.pt")
    if os.path.exists(last):                       # resume after a crash / restart
        r = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); sched.load_state_dict(r["sched"])
        best, bad, hist, start = r["best"], r["bad"], r["hist"], r["epoch"] + 1
        sampler.rng.set_state(r["sampler_rng"])
        print(f"resumed {args.name} at epoch {start} (best val AUC so far {best:.4f})")
    for ep in range(start, args.epochs):
        model.train(); t0, tl, n, skipped, k = time.time(), 0.0, 0, 0, 0
        opt.zero_grad(set_to_none=True)
        for loc, ctx, y, d, dm in dl_tr:
            try:
                loc, ctx, y, d, dm = (t.to(device, non_blocking=True) for t in (loc, ctx, y, d, dm))
                with torch.autocast(device.type, enabled=C.amp_enabled(device)):
                    out = model(loc, ctx)
                out = {kk: (v.float() if v is not None else None) for kk, v in out.items()}
                loss, _, _ = crit(out, y, d, dm)
                scaler.scale(loss / accum).backward()
            except (torch.OutOfMemoryError, MemoryError, RuntimeError) as e:
                if "memory" not in str(e).lower() and not isinstance(e, (torch.OutOfMemoryError, MemoryError)):
                    raise
                # another program on this shared PC took memory: skip this batch, free memory, wait, continue
                skipped += 1
                loc = ctx = y = d = dm = out = loss = None
                opt.zero_grad(set_to_none=True); k = 0
                import gc; gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(5)
                if skipped > 60:
                    raise
                continue
            tl += loss.item(); n += 1; k += 1
            if k == accum:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True); k = 0
        if skipped:
            print(f"  (skipped {skipped} batches because memory was busy)")
        for tries in range(6):                           # validation also waits if memory is busy
            try:
                v = validate(model, dl_va, device, True); break
            except (torch.OutOfMemoryError, MemoryError, RuntimeError) as e:
                if tries == 5 or ("memory" not in str(e).lower() and not isinstance(e, (torch.OutOfMemoryError, MemoryError))):
                    raise
                import gc; gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("  validation: memory busy, retrying in 30 s"); time.sleep(30)
        hist.append(dict(epoch=ep, loss=tl / max(1, n), val_auc=v["auc"], val_ap=v["ap"], time_s=time.time() - t0))
        pd.DataFrame(hist).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        star = v["auc"] > best
        if star:
            best, bad = v["auc"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "val": v}, os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        print(f"ep {ep:02d} loss {tl / max(1, n):.4f} | val AUC {v['auc']:.4f} AP {v['ap']:.4f} | {time.time() - t0:.0f}s {'*' if star else ''}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "bad": bad, "hist": hist, "sampler_rng": sampler.rng.get_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
        if bad >= args.patience:
            print("early stopping"); break
    open(os.path.join(run_dir, "DONE"), "w").write("finished\n")
    print(f"best val AUC {best:.4f}")


# ----------------------------------------------------------------------------------------- eval
def auc_ci(y, p, groups, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    pats = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in pats}
    vals = []
    for _ in range(n):
        sel = np.concatenate([idx_of[g] for g in rng.choice(pats, len(pats))])
        if 0 < y[sel].sum() < len(sel):
            vals.append(roc_auc_score(y[sel], p[sel]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def score(pred, meta, arr):
    out = {k: [] for k in ("prob_raw", "uncertainty")}
    for s in range(0, len(meta), 256):
        r = pred.score(np.asarray(arr[meta.row.to_numpy()[s:s + 256]]))
        out["prob_raw"].append(r["prob_raw"]); out["uncertainty"].append(r["uncertainty"])
    m = meta.copy()
    m["prob_raw"], m["uncertainty"] = np.concatenate(out["prob_raw"]), np.concatenate(out["uncertainty"])
    return m


def evaluate(args):
    run_dir = os.path.join(C.RUNS_DIR, args.name)
    if not os.path.exists(os.path.join(run_dir, "best.pt")):
        raise SystemExit(f"{args.name} has no trained model yet - its training did not finish. Run step 13 again.")
    meta = load_meta()
    arr = np.load(os.path.join(L25_DIR, "patches.npy"), mmap_mode="r")
    pred = Predictor(os.path.join(run_dir, "best.pt"), apply_calibration=False)
    va = score(pred, meta[meta.role == "val"], arr)
    te = score(pred, meta[meta.role == "test"], arr)
    pv = np.clip(va.prob_raw.to_numpy(), 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e4).fit(np.log(pv / (1 - pv))[:, None], va.label.to_numpy())
    cal = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "fitted_on": "luna25 validation fold"}
    json.dump(cal, open(os.path.join(run_dir, "calibration.json"), "w"), indent=2)
    for d in (va, te):
        d["prob"] = platt(d.prob_raw.to_numpy(), cal)
    # operating point chosen on VALIDATION (Youden), applied unchanged to test
    fpr, tpr, th = roc_curve(va.label, va.prob)
    thr = float(th[np.argmax(tpr - fpr)])
    y, p, g = te.label.to_numpy(), te.prob.to_numpy(), te.patient.to_numpy()
    tp = ((p >= thr) & (y == 1)).sum(); fn = ((p < thr) & (y == 1)).sum()
    tn = ((p < thr) & (y == 0)).sum(); fp = ((p >= thr) & (y == 0)).sum()
    err = ((p >= 0.5) != (y == 1)).astype(int)
    ece, _ = expected_calibration_error(p, y)
    ece_raw, _ = expected_calibration_error(te.prob_raw.to_numpy(), y)
    m = {"run": args.name, "init_from": pred.cfg.get("init_from"), "n_test": int(len(te)), "n_test_malignant": int(y.sum()),
         "n_test_patients": int(len(np.unique(g))),
         "val_auc": float(roc_auc_score(va.label, va.prob)), "test_auc": float(roc_auc_score(y, p)),
         "test_auc_ci95": auc_ci(y, p, g), "test_ap": float(average_precision_score(y, p)),
         "threshold_from_val": thr, "test_sensitivity": float(tp / max(1, tp + fn)),
         "test_specificity": float(tn / max(1, tn + fp)), "test_ece": ece, "test_ece_uncalibrated": ece_raw,
         "uncertainty_auc_for_errors": float(roc_auc_score(err, te.uncertainty)) if 0 < err.sum() < len(err) else None}
    # simple reference: LUNA16 model's predicted nodule SIZE alone as malignancy score
    ext = os.path.join(L25_DIR, "external", "predictions.csv")
    if os.path.exists(ext):
        e = pd.read_csv(ext, dtype={"seriesuid": str, "patient": str})
        e = e[e.role == "test"]
        m["size_only_baseline_test_auc"] = float(roc_auc_score(e.label, e.diameter_pred_mm))
        m["size_only_baseline_auc_ci95"] = auc_ci(e.label.to_numpy(), e.diameter_pred_mm.to_numpy(), e.patient.to_numpy())
    out = os.path.join(run_dir, "eval_luna25_test")
    os.makedirs(out, exist_ok=True)
    te.to_csv(os.path.join(out, "predictions.csv"), index=False)
    triage_curve(p, y, te.uncertainty.to_numpy()).to_csv(os.path.join(out, "triage.csv"), index=False)
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(json.dumps(m, indent=2))


def table(_args):
    rows, preds = [], {}
    for name in sorted(os.listdir(C.RUNS_DIR)):
        f = os.path.join(C.RUNS_DIR, name, "eval_luna25_test", "metrics.json")
        if os.path.exists(f):
            m = json.load(open(f))
            rows.append({"model": name, "init": m.get("init_from") or "scratch", "val_AUC": round(m["val_auc"], 4),
                         "test_AUC": round(m["test_auc"], 4), "AUC_95CI": "{:.3f}-{:.3f}".format(*m["test_auc_ci95"]),
                         "AP": round(m["test_ap"], 4), "sens": round(m["test_sensitivity"], 3),
                         "spec": round(m["test_specificity"], 3), "ECE": round(m["test_ece"], 4),
                         "ECE_raw": round(m["test_ece_uncalibrated"], 4),
                         "unc_AUC_errors": round(m["uncertainty_auc_for_errors"] or float("nan"), 3),
                         "size_only_AUC": round(m.get("size_only_baseline_test_auc", float("nan")), 4)})
            preds[name] = pd.read_csv(os.path.join(C.RUNS_DIR, name, "eval_luna25_test", "predictions.csv"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.WORK_DIR, "luna25_malignancy_table.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df)
    if "l25_transfer" in preds and "l25_scratch" in preds:          # paired bootstrap over patients
        a, b = preds["l25_scratch"], preds["l25_transfer"]
        y, pa, pb, g = a.label.to_numpy(), a.prob.to_numpy(), b.prob.to_numpy(), a.patient.to_numpy()
        rng = np.random.RandomState(0); pats = np.unique(g); idx = {p: np.where(g == p)[0] for p in pats}
        diffs = []
        for _ in range(2000):
            s = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
            if 0 < y[s].sum() < len(s):
                diffs.append(roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]))
        diffs = np.array(diffs)
        pval = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        print(f"\ntransfer - scratch AUC: {roc_auc_score(y, pb) - roc_auc_score(y, pa):+.4f}  "
              f"95% CI [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]  p={pval:.3f}")
    print("->", os.path.join(C.WORK_DIR, "luna25_malignancy_table.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "table"])
    ap.add_argument("--name", default="l25_transfer")
    ap.add_argument("--init", default=None, help="LUNA16 run whose weights start the training (transfer)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=8000)
    ap.add_argument("--pos-fraction", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=32, help="per step; small = less RAM/GPU memory")
    ap.add_argument("--effective-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    if a.lr is None:
        a.lr = 3e-4 if a.init else 1e-3
    {"train": train, "eval": evaluate, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
