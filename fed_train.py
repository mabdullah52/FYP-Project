"""Swarm / federated learning experiment (privacy-preserving training across hospitals).

Simulates 4 hospitals that never share CT data: LUNA16 training subsets are split by subset
(patient-disjoint) -> A = subsets 0-1, B = 2-3, C = 4-5, D = 6-7. Validation = subset 8, test = subset 9.

    python fed_train.py fedavg            # federated averaging (FedAvg, McMahan et al. 2017)
    python fed_train.py local --site A    # one hospital alone (what a single hospital could do)
    python fed_train.py central           # all data pooled (the privacy-violating upper bound)
    python fed_train.py report            # CPM of all three on test subset 9 + paired test

Each round every hospital starts from the global model, trains locally on its own patches, and sends
ONLY its model weights back; the server averages them (weighted by data size) and sends the new global
model out again. Compute is matched: one federated round = one central epoch = 24,000 training patches.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

import config as C
from dataset import BalancedEpochSampler, PatchDataset, ThreadLoader
from model import build_model
from train import MultiTaskLoss, fit_batch_size, validate

SITES = {"A": [0, 1], "B": [2, 3], "C": [4, 5], "D": [6, 7]}
VAL, TEST = C.DEFAULT_VAL_SUBSET, C.DEFAULT_TEST_SUBSET


def setup(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = C.HU_WINDOWS
    cfg = dict(in_ch=len(windows), width=C.TRAIN["width"], dropout=C.TRAIN["dropout"], use_context=True,
               fusion="gated", windows=windows, test_subset=TEST, val_subset=VAL, args=vars(args))
    meta = pd.read_csv(os.path.join(C.PATCH_DIR, "meta.csv"))
    arr = os.path.join(C.PATCH_DIR, "patches.npy")
    bs = fit_batch_size(cfg, args.batch_size, len(windows), True, device) if device.type == "cuda" else args.batch_size
    va = meta[meta.subset == VAL]
    dl_va = ThreadLoader(PatchDataset(va, arr, windows, train=False), bs, threads=args.workers, prefetch=2)
    return device, cfg, meta, arr, bs, dl_va


def site_loader(meta, arr, subsets, n_samples, bs, seed, workers):
    d = meta[meta.subset.isin(subsets)]
    ds = PatchDataset(d, arr, C.HU_WINDOWS, train=True)
    sampler = BalancedEpochSampler(ds.labels, n_samples, C.TRAIN["pos_fraction"], seed)
    return ThreadLoader(ds, bs, sampler=sampler, drop_last=True, threads=workers, prefetch=2), len(d)


def local_steps(model, crit, opt, loader, device, lr):
    for g in opt.param_groups:
        g["lr"] = lr
    model.train(); tl, n = 0.0, 0
    params = list(model.parameters()) + list(crit.parameters())
    for loc, ctx, y, d, dm in loader:
        loc, ctx, y, d, dm = (t.to(device, non_blocking=True) for t in (loc, ctx, y, d, dm))
        out = model(loc, ctx)
        loss, _, _ = crit(out, y, d, dm)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        tl += loss.item(); n += 1
    return tl / max(1, n)


def average(states, weights):
    """Weighted average of state dicts (float tensors incl. BatchNorm statistics)."""
    w = np.asarray(weights, float) / np.sum(weights)
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            out[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:
            out[k] = max(s[k] for s in states)            # num_batches_tracked
    return out


def run_federated(args, sites, name):
    device, cfg, meta, arr, bs, dl_va = setup(args)
    rd = os.path.join(C.RUNS_DIR, name); os.makedirs(rd, exist_ok=True)
    if os.path.exists(os.path.join(rd, "DONE")):
        print(name, "already finished"); return
    json.dump(cfg, open(os.path.join(rd, "config.json"), "w"), indent=2)
    per_site = args.samples_per_round // len(SITES)           # same patches per site whether alone or in the federation
    loaders, sizes = {}, {}
    for s in list(sites):
        if meta[meta.subset.isin(SITES[s])].label.sum() == 0:
            print(f"site {s} has no nodule patches - left out"); sites.remove(s); continue
        loaders[s], sizes[s] = site_loader(meta, arr, SITES[s], per_site, bs, args.seed + ord(s), args.workers)
    print(f"{name}: sites {sites} | patches per site {sizes} | {per_site} samples per site per round | batch {bs} | {device}")
    glob_model = build_model(cfg).to(device)
    clients = {s: (build_model(cfg).to(device), MultiTaskLoss("focal", C.TRAIN["focal_gamma"], C.TRAIN["focal_alpha"]).to(device))
               for s in sites}
    opts = {s: torch.optim.AdamW(list(m.parameters()) + list(c.parameters()), lr=args.lr, weight_decay=C.TRAIN["weight_decay"])
            for s, (m, c) in clients.items()}
    start, best, hist = 0, -1.0, []
    last = os.path.join(rd, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        glob_model.load_state_dict(ck["global"])
        for s in sites:
            clients[s][1].load_state_dict(ck["crit"][s]); opts[s].load_state_dict(ck["opt"][s])
        start, best, hist = ck["round"] + 1, ck["best"], ck["hist"]
        print(f"resumed at round {start}")
    n_params = sum(p.numel() for p in glob_model.parameters())
    for r in range(start, args.rounds):
        t0 = time.time()
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * r / args.rounds))
        states, losses = [], {}
        for s in sites:
            m, c = clients[s]
            m.load_state_dict(glob_model.state_dict())        # hospital receives the global model
            losses[s] = local_steps(m, c, opts[s], loaders[s], device, lr)
            states.append(copy.deepcopy(m.state_dict()))       # ... and sends back only its weights
        glob_model.load_state_dict(average(states, [sizes[s] for s in sites]))
        v = validate(glob_model, dl_va, device, True)
        hist.append({"round": r, "lr": lr, "val_auc": v["auc"], "val_ap": v["ap"], "time_s": time.time() - t0,
                     **{f"loss_{s}": l for s, l in losses.items()},
                     "MB_exchanged": round(2 * n_params * 4 * len(sites) / 1e6, 1)})
        pd.DataFrame(hist).to_csv(os.path.join(rd, "history.csv"), index=False)
        star = v["auc"] > best
        if star:
            best = v["auc"]
            torch.save({"model": glob_model.state_dict(), "cfg": cfg, "epoch": r, "val": v}, os.path.join(rd, "best.pt"))
        torch.save({"global": glob_model.state_dict(), "crit": {s: clients[s][1].state_dict() for s in sites},
                    "opt": {s: opts[s].state_dict() for s in sites}, "round": r, "best": best, "hist": hist}, last + ".tmp")
        os.replace(last + ".tmp", last)
        print(f"round {r:02d} | val AUC {v['auc']:.4f} AP {v['ap']:.4f} | "
              + " ".join(f"{s}:{l:.3f}" for s, l in losses.items()) + f" | {time.time() - t0:.0f}s {'*' if star else ''}")
    open(os.path.join(rd, "DONE"), "w").write("ok\n")
    print(f"{name}: best val AUC {best:.4f}")


def run_central(args):
    """All hospitals' data pooled in one place (upper bound) - same total samples and rounds."""
    cmd = [sys.executable, "train.py", "--name", "fed_central", "--epochs", str(args.rounds), "--samples-per-epoch",
           str(args.samples_per_round), "--patience", str(args.rounds), "--workers", str(args.workers), "--resume"]
    print(" ".join(cmd)); sys.exit(subprocess.call(cmd))


def evaluate_runs(args):
    runs = ["fed_avg", "fed_central"] + [f"fed_local_{s}" for s in SITES]
    for r in runs:
        rd = os.path.join(C.RUNS_DIR, r)
        if not os.path.exists(os.path.join(rd, "best.pt")):
            continue
        out = os.path.join(rd, f"eval_subset{TEST}_plain", "metrics.json")
        if not os.path.exists(out):
            print("evaluating", r, "on test subset", TEST)
            subprocess.call([sys.executable, "evaluate.py", "--run", r, "--subset", str(TEST), "--no-tta", "--mc", "1",
                             "--tag", "_plain", "--workers", str(args.workers)])
    rows, preds = [], {}
    for r in runs:
        f = os.path.join(C.RUNS_DIR, r, f"eval_subset{TEST}_plain", "metrics.json")
        if os.path.exists(f):
            m = json.load(open(f))
            m = m.get("froc", m)
            h = pd.read_csv(os.path.join(C.RUNS_DIR, r, "history.csv"))
            vcol = "val_auc"
            rows.append({"model": r, "data": {"fed_avg": "4 hospitals, weights only", "fed_central": "all data pooled"}
                         .get(r, f"hospital {r[-1]} alone"), "best_val_AUC": round(h[vcol].max(), 4),
                         "test_CPM": round(m["cpm"], 4), "CPM_95CI": "{:.3f}-{:.3f}".format(*m["cpm_ci95"])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.WORK_DIR, "federated_table.csv"), index=False)
    print("\n", df.to_string(index=False))
    try:
        cmp_ = [(a, b) for a, b in (("fed_central", "fed_avg"), ("fed_local_A", "fed_avg"))
                if os.path.exists(os.path.join(C.RUNS_DIR, a, f"eval_subset{TEST}_plain"))
                and os.path.exists(os.path.join(C.RUNS_DIR, b, f"eval_subset{TEST}_plain"))]
        from stats_compare import paired_cpm
        for a, b in cmp_:
            res = paired_cpm(os.path.join(C.RUNS_DIR, a, f"eval_subset{TEST}_plain"),
                             os.path.join(C.RUNS_DIR, b, f"eval_subset{TEST}_plain"))
            print(f"{b} vs {a}: delta {res['delta']:+.4f}  95% CI [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]  p={res['p']:.3f}")
    except Exception as e:                                  # significance test is optional
        print("(paired test skipped:", e, ")")
    print("->", os.path.join(C.WORK_DIR, "federated_table.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fedavg", "local", "central", "report"])
    ap.add_argument("--site", default="A", choices=list(SITES))
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--samples-per-round", type=int, default=24000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=C.TRAIN["lr"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    if a.mode == "fedavg":
        run_federated(a, list(SITES), "fed_avg")
    elif a.mode == "local":
        run_federated(a, [a.site], f"fed_local_{a.site}")
    elif a.mode == "central":
        run_central(a)
    else:
        evaluate_runs(a)


if __name__ == "__main__":
    main()
