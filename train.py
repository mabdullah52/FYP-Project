"""Step 2 - train MSGF-Net on one patient-level LUNA16 fold.

    python train.py --test-subset 9 --val-subset 8 --name msgf_f9          # proposed model
    # ablations (same fold, change one thing each):
    python train.py --name abl_nocontext  --no-context
    python train.py --name abl_concat     --fusion concat
    python train.py --name abl_1window    --windows single
    python train.py --name abl_noaux      --aux-weight 0
    python train.py --name abl_bce        --loss bce
    # round 2 (hard-negative curriculum), after mine_hard_negatives.py --run msgf_f9:
    python train.py --name msgf_f9_r2 --init-from msgf_f9 --hard-meta work/patches/hard_msgf_f9.csv \
                    --epochs 10 --lr 3e-4
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
from dataset import BalancedEpochSampler, PatchDataset, ThreadLoader, split_meta
from model import build_model


def focal_loss(logit, y, gamma, alpha):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    w = alpha * y + (1 - alpha) * (1 - y)
    return (w * (1 - pt) ** gamma * ce).mean()


class MultiTaskLoss(torch.nn.Module):
    """Homoscedastic-uncertainty weighting (Kendall et al., 2018) of the two tasks."""

    def __init__(self, loss="focal", gamma=2.0, alpha=0.5, aux_weight=1.0):
        super().__init__()
        self.loss, self.gamma, self.alpha, self.aux_weight = loss, gamma, alpha, aux_weight
        self.log_vars = torch.nn.Parameter(torch.zeros(2))

    def forward(self, out, y, d, dmask):
        if self.loss == "focal":
            lc = focal_loss(out["logit"], y, self.gamma, self.alpha)
        else:
            lc = F.binary_cross_entropy_with_logits(out["logit"], y)
        if self.aux_weight > 0 and dmask.sum() > 0:
            ld = F.smooth_l1_loss(torch.log(out["diameter"][dmask > 0]), torch.log(d[dmask > 0].clamp(min=1)))
            total = (torch.exp(-self.log_vars[0]) * lc + self.log_vars[0]
                     + self.aux_weight * (torch.exp(-self.log_vars[1]) * ld + self.log_vars[1]))
        else:
            ld = torch.zeros((), device=y.device)
            total = lc
        return total, lc.detach(), ld.detach()


def fit_batch_size(cfg, bs, in_ch, use_context, device):
    """Try one training step on random data; halve the batch size until it fits the GPU."""
    import gc
    while bs >= 4:
        oom = False
        m = opt = loc = ctx = out = None
        try:
            m = build_model(cfg).to(device)
            opt = torch.optim.AdamW(m.parameters(), 1e-4)
            loc = torch.randn(bs, in_ch, C.LOCAL_SIZE, C.LOCAL_SIZE, C.LOCAL_SIZE, device=device)
            ctx = torch.randn(bs, in_ch, C.STORE_SIZE, C.STORE_SIZE, C.STORE_SIZE, device=device) if use_context else None
            with torch.autocast(device.type, enabled=C.amp_enabled(device)):
                out = m(loc, ctx)
            (out["logit"].float().mean() + out["diameter"].float().mean()).backward()
            opt.step()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
        m = opt = loc = ctx = out = None          # free everything outside the except block
        gc.collect()
        torch.cuda.empty_cache()
        if not oom:
            free, total = torch.cuda.mem_get_info()
            print(f"batch size {bs} fits on the GPU ({free / 1e9:.1f} of {total / 1e9:.1f} GB free)")
            return bs
        print(f"batch size {bs} does not fit in GPU memory -> trying {bs // 2}")
        bs //= 2
    raise RuntimeError("GPU has too little free memory even for batch 4 - is another program using the GPU?")


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)


@torch.no_grad()
def validate(model, loader, device, use_context):
    model.eval()
    ps, ys, de, dt = [], [], [], []
    for loc, ctx, y, d, dm in loader:
        loc = loc.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True) if use_context else None
        with torch.autocast(device.type, enabled=C.amp_enabled(device)):
            out = model(loc, ctx)
        ps.append(torch.sigmoid(out["logit"].float()).cpu())
        ys.append(y)
        m = dm > 0
        de.append(out["diameter"].float().cpu()[m])
        dt.append(d[m])
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    de, dt = torch.cat(de).numpy(), torch.cat(dt).numpy()
    res = {"auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
           "ap": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
           "diam_mae_mm": float(np.abs(de - dt).mean()) if len(dt) else float("nan")}
    # sensitivity at the threshold giving 95% specificity on the (sub-sampled) negatives
    if 0 < y.sum() < len(y):
        thr = np.quantile(p[y == 0], 0.95)
        res["sens@spec95"] = float((p[y == 1] >= thr).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="msgf")
    ap.add_argument("--test-subset", type=int, default=C.DEFAULT_TEST_SUBSET)
    ap.add_argument("--val-subset", type=int, default=C.DEFAULT_VAL_SUBSET)
    ap.add_argument("--epochs", type=int, default=C.TRAIN["epochs"])
    ap.add_argument("--batch-size", type=int, default=C.TRAIN["batch_size"])
    ap.add_argument("--samples-per-epoch", type=int, default=C.TRAIN["samples_per_epoch"])
    ap.add_argument("--lr", type=float, default=C.TRAIN["lr"])
    ap.add_argument("--width", type=int, default=C.TRAIN["width"])
    ap.add_argument("--workers", type=int, default=C.TRAIN["num_workers"], help="data-loading threads")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--arch", default="msgf", choices=["msgf", "resnet3d18", "resnet25d", "cnn3d"],
                    help="network; the others are comparison baselines (baselines.py)")
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat"])
    ap.add_argument("--windows", default="multi", choices=["multi", "single"])
    ap.add_argument("--loss", default="focal", choices=["focal", "bce"])
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hard-meta", default=None, help="csv from mine_hard_negatives.py (round 2)")
    ap.add_argument("--init-from", default=None, help="run name whose best.pt initialises the weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    run_dir = os.path.join(C.RUNS_DIR, args.name)
    os.makedirs(run_dir, exist_ok=True)

    windows = C.HU_WINDOWS if args.windows == "multi" else [C.HU_WINDOWS[2]]
    use_context = not args.no_context and args.arch == "msgf"     # baselines use the local view only
    cfg = dict(in_ch=len(windows), width=args.width, dropout=C.TRAIN["dropout"], arch=args.arch,
               use_context=use_context, fusion=args.fusion, windows=windows,
               test_subset=args.test_subset, val_subset=args.val_subset, args=vars(args))
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    meta_path = os.path.join(C.PATCH_DIR, "meta.csv")
    arr_path = os.path.join(C.PATCH_DIR, "patches.npy")
    tr, va, _ = split_meta(meta_path, args.test_subset, args.val_subset, args.hard_meta)
    print(f"train {len(tr)} (pos {int(tr.label.sum())}) | val {len(va)} (pos {int(va.label.sum())}) | "
          f"test subset {args.test_subset} held out | device {device}")

    ds_tr = PatchDataset(tr, arr_path, windows, train=True, use_context=use_context)
    ds_va = PatchDataset(va, arr_path, windows, train=False, use_context=use_context)
    sampler = BalancedEpochSampler(ds_tr.labels, args.samples_per_epoch, C.TRAIN["pos_fraction"], args.seed)
    # threads, not worker processes (see ThreadLoader) - safe on Windows, same speed
    if device.type == "cuda":
        args.batch_size = fit_batch_size(cfg, args.batch_size, len(windows), use_context, device)
    dl_tr = ThreadLoader(ds_tr, args.batch_size, sampler=sampler, drop_last=True, threads=args.workers)
    dl_va = ThreadLoader(ds_va, args.batch_size * 2, shuffle=False, threads=args.workers)

    model = build_model(cfg).to(device)
    if args.init_from:
        ck0 = torch.load(os.path.join(C.RUNS_DIR, args.init_from, "best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck0["model"])
        print(f"initialised from {args.init_from} (val AUC {ck0['val']['auc']:.4f})")
    crit = MultiTaskLoss(args.loss, C.TRAIN["focal_gamma"], C.TRAIN["focal_alpha"], args.aux_weight).to(device)
    params = list(model.parameters()) + list(crit.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=C.TRAIN["weight_decay"])
    steps = args.epochs * len(dl_tr)
    warm = max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=C.amp_enabled(device))
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | mixed precision: {C.amp_enabled(device)}")

    start, best, bad, hist = 0, -1.0, 0, []
    ck_last = os.path.join(run_dir, "last.pt")
    done_flag = os.path.join(run_dir, "DONE")
    if args.resume and os.path.exists(done_flag):
        print(f"run '{args.name}' already finished -> {os.path.join(run_dir, 'best.pt')} (delete {done_flag} to retrain)")
        return
    if args.resume and os.path.exists(ck_last):
        ck = torch.load(ck_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); crit.load_state_dict(ck["crit"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, hist = ck["epoch"] + 1, ck["best"], ck["hist"]
        bad = ck.get("bad", 0)
        print(f"resumed at epoch {start}")

    for ep in range(start, args.epochs):
        model.train()
        t0, tl, tc, td, n = time.time(), 0.0, 0.0, 0.0, 0
        for loc, ctx, y, d, dm in dl_tr:
            loc, y, d, dm = (t.to(device, non_blocking=True) for t in (loc, y, d, dm))
            ctx = ctx.to(device, non_blocking=True) if use_context else None
            with torch.autocast(device.type, enabled=C.amp_enabled(device)):
                out = model(loc, ctx)
            out = {k: (v.float() if v is not None else None) for k, v in out.items()}
            loss, lc, ld = crit(out, y, d, dm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item(); tc += lc.item(); td += ld.item(); n += 1
        v = validate(model, dl_va, device, use_context)
        rec = dict(epoch=ep, loss=tl / n, cls_loss=tc / n, diam_loss=td / n, lr=opt.param_groups[0]["lr"],
                   time_s=time.time() - t0, **{f"val_{k}": x for k, x in v.items()})
        hist.append(rec)
        pd.DataFrame(hist).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        score = v["auc"] if not math.isnan(v["auc"]) else -tl
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "val": v},
                       os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "crit": crit.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad, "hist": hist, "cfg": cfg}, ck_last)
        print(f"ep {ep:02d} loss {tl / n:.4f} (cls {tc / n:.4f}, diam {td / n:.4f}) | val AUC {v['auc']:.4f} "
              f"AP {v['ap']:.4f} sens@spec95 {v.get('sens@spec95', float('nan')):.3f} "
              f"diamMAE {v['diam_mae_mm']:.2f}mm | {rec['time_s']:.0f}s {'*' if improved else ''}")
        if bad >= args.patience:
            print("early stopping")
            break
    open(done_flag, "w").write("finished\n")
    print(f"best val AUC {best:.4f} -> {os.path.join(run_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
