"""Paper 2 - GROWTH-AWARE malignancy prediction from the current AND the prior screening scan.

LUNA25 (NLST) follows the same nodule over up to 3 yearly screening rounds (LesionID). For every
nodule seen at an earlier round we build a pair (current patch, prior patch, years between them).

Proposed model - TG-MSGF (Temporal-Gated MSGF-Net):
    shared MSGF-Net encoder (initialised from the LUNA25 malignancy model) -> z_cur, z_prior
    change vector  d = z_cur - z_prior
    temporal gate  beta = sigmoid(MLP[z_cur, d, dt])        (how much the change should matter)
    risk           = MLP[z_cur, beta * d]
Comparisons on the SAME test nodules (patient-level split from luna25_prepare.py):
    * single-scan ensemble (luna25_final.py)                - no prior information
    * hand-crafted growth: logistic regression on current risk + diameter growth per year
    * TG-MSGF with the prior replaced by the current scan    - "no-prior" control (same network)
    * TG-MSGF (3 seeds, ensembled)                           - proposed

    python luna25_temporal.py pairs                 -> pair table + statistics
    python luna25_temporal.py train --seed 0 [--no-prior]
    python luna25_temporal.py eval                  -> work/luna25_temporal/{temporal_metrics.json, temporal_table.csv}
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
from dataset import BalancedEpochSampler, ThreadLoader, augment, make_inputs
from luna25_common import L25_DIR, find_csv
from model import build_model

OUT = os.path.join(C.WORK_DIR, "luna25_temporal")
FINAL = os.path.join(C.WORK_DIR, "luna25_final")
CV_FOLD = int(os.environ.get("L25_CV_FOLD", "0") or 0)       # step 31: cross-validation fold (0 = original split)
INIT_RUN = "l25_scratch" if CV_FOLD == 0 else f"l25_cv{CV_FOLD}"
SEEDS = [0, 1, 2]


# ------------------------------------------------------------------------------------ pairs
def _n(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def build_pairs():
    meta = pd.read_csv(os.path.join(L25_DIR, "meta.csv"), dtype={"seriesuid": str, "patient": str})
    raw = pd.read_csv(find_csv(C.find_luna25(), ask=False))
    col = {_n(c): c for c in raw.columns}
    les = col.get("lesionid") or col.get("noduleid")
    ann, date = col.get("annotationid"), col.get("studydate")
    if not (les and ann and date):
        raise SystemExit(f"CSV has no LesionID/AnnotationID/StudyDate columns: {list(raw.columns)}")
    lk = raw[[ann, les, date]].rename(columns={ann: "annotation", les: "lesion", date: "date"})
    lk["annotation"] = lk["annotation"].astype(str)
    meta["annotation"] = meta["annotation"].astype(str)
    m = meta.drop(columns=[c for c in ("lesion", "date") if c in meta]).merge(lk, on="annotation", how="left")
    m["date"] = pd.to_datetime(m["date"].astype(str), format="%Y%m%d", errors="coerce")
    m = m.dropna(subset=["lesion", "date"])
    m["lesion_key"] = m.patient.astype(str) + "_" + m.lesion.astype(str)
    ext = os.path.join(L25_DIR, "external", "predictions.csv")
    if os.path.exists(ext):
        e = pd.read_csv(ext)[["row", "diameter_pred_mm"]]
        m = m.merge(e, on="row", how="left")
    rows = []
    for _, g in m.sort_values("date").groupby("lesion_key"):
        g = g.drop_duplicates("date")
        for i in range(1, len(g)):
            c, p = g.iloc[i], g.iloc[i - 1]
            rows.append({"row_cur": int(c.row), "row_prior": int(p.row), "patient": c.patient, "role": c.role,
                         "label": int(c.label), "label_prior": int(p.label), "lesion_key": c.lesion_key,
                         "dt_years": (c.date - p.date).days / 365.25, "round": i,
                         "diam_cur": c.get("diameter_pred_mm", np.nan), "diam_prior": p.get("diameter_pred_mm", np.nan),
                         "age": c.get("age", np.nan)})
    pairs = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    pairs.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    st = {"lesions_total": int(m.lesion_key.nunique()), "pairs": int(len(pairs)),
          "label_changes_between_rounds": int((pairs.label != pairs.label_prior).sum()),
          "median_interval_years": float(pairs.dt_years.median()) if len(pairs) else None}
    for r in ("train", "val", "test"):
        d = pairs[pairs.role == r]
        st[f"{r}_pairs_malignant_patients"] = [int(len(d)), int(d.label.sum()), int(d.patient.nunique())]
    json.dump(st, open(os.path.join(OUT, "pairs_stats.json"), "w"), indent=2)
    print(json.dumps(st, indent=2))
    return pairs


def load_pairs():
    p = os.path.join(OUT, "pairs.csv")
    pairs = pd.read_csv(p, dtype={"patient": str}) if os.path.exists(p) else build_pairs()
    if CV_FOLD:
        meta = pd.read_csv(os.path.join(L25_DIR, "meta.csv"), dtype={"patient": str}, usecols=["patient", "fold"])
        fold = pairs.patient.astype(str).map(meta.drop_duplicates("patient").set_index("patient").fold)
        pairs["role"] = np.where(fold == CV_FOLD, "test", np.where(fold == (CV_FOLD + 1) % 5, "val", "train"))
    return pairs


# ------------------------------------------------------------------------------------ model
class TGMSGF(nn.Module):
    def __init__(self, base, drop=0.3):
        super().__init__()
        self.enc = base                       # MSGF-Net; only .features() is used
        c = base.feat_dim
        self.gate = nn.Sequential(nn.Linear(2 * c + 1, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * c, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, lc, cc, lp, cp, dt):
        n = lc.shape[0]
        z, _ = self.enc.features(torch.cat([lc, lp]), torch.cat([cc, cp]))   # one pass, shared weights
        zc, zp = z[:n], z[n:]
        d = zc - zp
        beta = torch.sigmoid(self.gate(torch.cat([zc, d, dt[:, None]], 1)))
        return self.head(torch.cat([zc, beta * d], 1)).squeeze(1), beta.mean(1)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, train, no_prior=False):
        self.p, self.train, self.no_prior = pairs.reset_index(drop=True), train, no_prior
        self.labels = self.p.label.to_numpy().astype(np.float32)
        self.arr = None

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(os.path.join(L25_DIR, "patches.npy"), mmap_mode="r")
        r = self.p.iloc[i]
        cur = np.asarray(self.arr[int(r.row_cur)])
        pri = cur if self.no_prior else np.asarray(self.arr[int(r.row_prior)])
        if self.train:
            seed = np.random.randint(1 << 30)          # same flips/rotation for both time points
            cur = augment(cur, np.random.RandomState(seed)); pri = augment(pri, np.random.RandomState(seed))
        lc, cc = make_inputs(cur, C.HU_WINDOWS)
        lp, cp = make_inputs(pri, C.HU_WINDOWS)
        dt = 0.0 if self.no_prior else float(r.dt_years)
        return lc, cc, lp, cp, dt, self.labels[i]


class PairLoader(ThreadLoader):
    def _collate(self, items):
        lc, cc, lp, cp, dt, y = zip(*items)
        return (torch.stack(lc), torch.stack(cc), torch.stack(lp), torch.stack(cp),
                torch.tensor(dt, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))


def run_name(seed, no_prior):
    pre = "l25_temporal" if CV_FOLD == 0 else f"l25_cv{CV_FOLD}_temporal"
    return f"{pre}{'_noprior' if no_prior else ''}_s{seed}"


@torch.no_grad()
def predict(model, pairs, device, no_prior, bs=16, tta=True):
    model.eval()
    dl = PairLoader(PairDataset(pairs, False, no_prior), bs, threads=3, prefetch=2)
    flips = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)] if tta else [()]
    out, betas = [], []
    for lc, cc, lp, cp, dt, _ in dl:
        lc, cc, lp, cp, dt = (t.to(device) for t in (lc, cc, lp, cp, dt))
        ps, bs_ = [], []
        for f in flips:
            fl = (lambda t: torch.flip(t, f)) if f else (lambda t: t)
            lg, b = model(fl(lc), fl(cc), fl(lp), fl(cp), dt)
            ps.append(torch.sigmoid(lg)); bs_.append(b)
        out.append(torch.stack(ps).mean(0).cpu()); betas.append(torch.stack(bs_).mean(0).cpu())
    return torch.cat(out).numpy(), torch.cat(betas).numpy()


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = run_name(args.seed, args.no_prior)
    rd = os.path.join(C.RUNS_DIR, name)
    os.makedirs(rd, exist_ok=True)
    if os.path.exists(os.path.join(rd, "DONE")):
        print(f"{name} already trained"); return
    pairs = load_pairs()
    tr, va = pairs[pairs.role == "train"], pairs[pairs.role == "val"]
    print(f"{name}: train {len(tr)} pairs ({int(tr.label.sum())} malignant) | val {len(va)} ({int(va.label.sum())}) | {device}")
    ck = torch.load(os.path.join(C.RUNS_DIR, INIT_RUN, "best.pt"), map_location=device, weights_only=False)
    base = build_model(ck["cfg"]); base.load_state_dict(ck["model"])
    model = TGMSGF(base).to(device)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": args.lr * 0.1},
                             {"params": list(model.gate.parameters()) + list(model.head.parameters()), "lr": args.lr}],
                            weight_decay=1e-4)
    ds = PairDataset(tr, True, args.no_prior)
    sampler = BalancedEpochSampler(ds.labels, args.samples_per_epoch, 0.25, args.seed)
    dl = PairLoader(ds, args.batch_size, sampler=sampler, drop_last=True, threads=3, prefetch=2)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    best, bad, hist, start = -1.0, 0, [], 0
    last = os.path.join(rd, "last.pt")
    if os.path.exists(last):
        r = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); sched.load_state_dict(r["sched"])
        best, bad, hist, start = r["best"], r["bad"], r["hist"], r["epoch"] + 1
        sampler.rng.set_state(r["rng"])
        print(f"resumed at epoch {start}")
    for ep in range(start, args.epochs):
        model.train(); t0, tl, n, skipped = time.time(), 0.0, 0, 0
        for lc, cc, lp, cp, dt, y in dl:
            try:
                lc, cc, lp, cp, dt, y = (t.to(device) for t in (lc, cc, lp, cp, dt, y))
                lg, _ = model(lc, cc, lp, cp, dt)
                p = torch.sigmoid(lg)
                bce = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
                pt = p * y + (1 - p) * (1 - y)
                loss = (0.5 * (1 - pt) ** 2 * bce).mean() * 4          # focal loss, gamma 2
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
                tl += loss.item(); n += 1
            except (torch.OutOfMemoryError, MemoryError, RuntimeError) as e:
                if "memory" not in str(e).lower() and not isinstance(e, (torch.OutOfMemoryError, MemoryError)):
                    raise
                skipped += 1; opt.zero_grad(set_to_none=True)
                import gc; gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(5)
                if skipped > 60:
                    raise
        pv, _ = predict(model, va, device, args.no_prior, tta=False)
        auc = roc_auc_score(va.label, pv)
        hist.append({"epoch": ep, "loss": tl / max(1, n), "val_auc": auc})
        pd.DataFrame(hist).to_csv(os.path.join(rd, "history.csv"), index=False)
        star = auc > best
        if star:
            best, bad = auc, 0
            torch.save({"model": model.state_dict(), "cfg": ck["cfg"], "no_prior": args.no_prior}, os.path.join(rd, "best.pt"))
        else:
            bad += 1
        print(f"ep {ep:02d} loss {tl / max(1, n):.4f} | val AUC {auc:.4f} | {time.time() - t0:.0f}s {'*' if star else ''}"
              + (f" (skipped {skipped})" if skipped else ""))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "bad": bad, "hist": hist, "rng": sampler.rng.get_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
        if bad >= args.patience:
            print("early stopping"); break
    open(os.path.join(rd, "DONE"), "w").write("ok\n")
    print(f"best val AUC {best:.4f}")


# ------------------------------------------------------------------------------------ evaluation
def boot(groups, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    pats = np.unique(groups); w = {g: np.where(groups == g)[0] for g in pats}
    for _ in range(n):
        yield np.concatenate([w[g] for g in rng.choice(pats, len(pats))])


def auc_ci(y, p, g):
    v = [roc_auc_score(y[s], p[s]) for s in boot(g) if 0 < y[s].sum() < len(s)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired(y, pa, pb, g):
    d = np.array([roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]) for s in boot(g) if 0 < y[s].sum() < len(s)])
    return {"delta": float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def evaluate(_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs()
    va, te = pairs[pairs.role == "val"].reset_index(drop=True), pairs[pairs.role == "test"].reset_index(drop=True)
    yv, yt, gt = va.label.to_numpy(), te.label.to_numpy(), te.patient.astype(str).to_numpy()
    # single-scan ensemble scores from luna25_final.py (val + test)
    sv = pd.read_csv(os.path.join(FINAL, "val_predictions.csv"))[["row", "ensemble"]]
    st = pd.read_csv(os.path.join(FINAL, "test_predictions.csv"))[["row", "ensemble"]]
    look = pd.concat([sv, st]).set_index("row").ensemble
    for d in (va, te):
        d["single_cur"] = look.reindex(d.row_cur).to_numpy()
        d["single_prior"] = look.reindex(d.row_prior).to_numpy()
    res = {"n_test_pairs": int(len(te)), "n_test_malignant": int(yt.sum()), "n_test_patients": int(len(np.unique(gt))),
           "n_val_pairs": int(len(va)), "models": {}, "paired_tests": {}}
    S = {"single_scan_ensemble": (va.single_cur.to_numpy(), te.single_cur.to_numpy())}
    # hand-crafted growth: current risk + diameter growth per year (+ change in risk)
    feats = lambda d: np.column_stack([logit(d.single_cur), d.diam_cur, (d.diam_cur - d.diam_prior) / d.dt_years.clip(lower=0.25),
                                       logit(d.single_cur) - logit(d.single_prior)])
    Xv, Xt = feats(va), feats(te)
    mu, sd = np.nanmean(Xv, 0), np.nanstd(Xv, 0) + 1e-6
    lr = LogisticRegression(C=1.0).fit(np.nan_to_num((Xv - mu) / sd), yv)
    S["handcrafted_growth"] = (lr.predict_proba(np.nan_to_num((Xv - mu) / sd))[:, 1],
                               lr.predict_proba(np.nan_to_num((Xt - mu) / sd))[:, 1])
    res["handcrafted_growth_coefficients"] = dict(zip(["risk_now", "diameter_now", "diameter_growth_per_year", "risk_change"],
                                                      map(float, lr.coef_[0])))
    # learned temporal models
    for no_prior in (False, True):
        pv_all, pt_all, beta_t = [], [], []
        for seed in SEEDS:
            ck_path = os.path.join(C.RUNS_DIR, run_name(seed, no_prior), "best.pt")
            if not os.path.exists(ck_path):
                continue
            tag = f"TG-MSGF{' no-prior' if no_prior else ''} seed {seed}"
            cache = os.path.join(OUT, f"scores_{run_name(seed, no_prior)}.npz")
            if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(ck_path):
                z = np.load(cache); pv, pt, bt = z["pv"], z["pt"], z["bt"]
            else:
                ck = torch.load(ck_path, map_location=device, weights_only=False)
                base = build_model(ck["cfg"])
                m = TGMSGF(base).to(device); m.load_state_dict(ck["model"])
                pv, _ = predict(m, va, device, no_prior)
                pt, bt = predict(m, te, device, no_prior)
                np.savez(cache, pv=pv, pt=pt, bt=bt)
            S[tag] = (pv, pt); pv_all.append(pv); pt_all.append(pt); beta_t.append(bt)
            print("  scored", tag)
        if pt_all:
            S[f"TG-MSGF{' no-prior' if no_prior else ''} ensemble"] = (np.mean(pv_all, 0), np.mean(pt_all, 0))
            if not no_prior:
                te["beta"] = np.mean(beta_t, 0)
                growth = (te.diam_cur - te.diam_prior).to_numpy()
                ok = ~np.isnan(growth)
                from scipy.stats import spearmanr
                res["gate_beta_vs_diameter_growth_spearman"] = float(spearmanr(te.beta[ok], growth[ok]).correlation)
                res["gate_beta_mean_malignant_vs_benign"] = [float(te.beta[yt == 1].mean()), float(te.beta[yt == 0].mean())]
    # stacked fusion (fitted on validation only): single-scan risk + learned temporal risk + growth features
    if "TG-MSGF ensemble" in S:
        fs = lambda d, tg: np.column_stack([logit(d.single_cur), logit(tg), (d.diam_cur - d.diam_prior) / d.dt_years.clip(lower=0.25),
                                            logit(d.single_cur) - logit(d.single_prior)])
        Xv2, Xt2 = fs(va, S["TG-MSGF ensemble"][0]), fs(te, S["TG-MSGF ensemble"][1])
        mu2, sd2 = np.nanmean(Xv2, 0), np.nanstd(Xv2, 0) + 1e-6
        lr2 = LogisticRegression(C=1.0).fit(np.nan_to_num((Xv2 - mu2) / sd2), yv)
        S["stacked: single + TG-MSGF + growth"] = (lr2.predict_proba(np.nan_to_num((Xv2 - mu2) / sd2))[:, 1],
                                                    lr2.predict_proba(np.nan_to_num((Xt2 - mu2) / sd2))[:, 1])
        res["stacked_coefficients"] = dict(zip(["single_scan_risk", "TG-MSGF_risk", "diameter_growth_per_year", "risk_change"],
                                               map(float, lr2.coef_[0])))
    rows = []
    for k, (pv, pt) in S.items():
        cal = LogisticRegression(C=1e4).fit(logit(pv)[:, None], yv)
        ptc = cal.predict_proba(logit(pt)[:, None])[:, 1]
        res["models"][k] = {"val_auc": float(roc_auc_score(yv, pv)), "test_auc": float(roc_auc_score(yt, pt)),
                            "test_auc_ci95": auc_ci(yt, pt, gt), "test_ap": float(average_precision_score(yt, pt))}
        rows.append({"model": k, "val_AUC": round(res["models"][k]["val_auc"], 4), "test_AUC": round(res["models"][k]["test_auc"], 4),
                     "AUC_95CI": "{:.3f}-{:.3f}".format(*res["models"][k]["test_auc_ci95"]),
                     "AP": round(res["models"][k]["test_ap"], 4)})
        te[f"p_{k}"] = ptc
    base_t = S["single_scan_ensemble"][1]
    for k in S:
        if k != "single_scan_ensemble" and ("ensemble" in k or k == "handcrafted_growth" or k.startswith("stacked")):
            res["paired_tests"][f"{k} vs single_scan_ensemble"] = paired(yt, base_t, S[k][1], gt)
    if "TG-MSGF ensemble" in S and "TG-MSGF no-prior ensemble" in S:
        res["paired_tests"]["TG-MSGF vs no-prior control"] = paired(yt, S["TG-MSGF no-prior ensemble"][1], S["TG-MSGF ensemble"][1], gt)
    if "TG-MSGF ensemble" in S:
        res["paired_tests"]["TG-MSGF vs handcrafted_growth"] = paired(yt, S["handcrafted_growth"][1], S["TG-MSGF ensemble"][1], gt)
    te.to_csv(os.path.join(OUT, "test_pair_predictions.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "temporal_metrics.json"), "w"), indent=2)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "temporal_table.csv"), index=False)
    print(f"\ntest: {len(te)} nodules that have a prior scan ({int(yt.sum())} malignant, {len(np.unique(gt))} patients)\n")
    print(df.to_string(index=False))
    print()
    for k, v in res["paired_tests"].items():
        print(f"{k:50s} delta {v['delta']:+.4f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p={v['p']:.3f}")
    if "stacked: single + TG-MSGF + growth" in S:
        res["paired_tests"]["stacked vs handcrafted_growth"] = paired(yt, S["handcrafted_growth"][1], S["stacked: single + TG-MSGF + growth"][1], gt)
        v = res["paired_tests"]["stacked vs handcrafted_growth"]
        print(f"{'stacked vs handcrafted_growth':50s} delta {v['delta']:+.4f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p={v['p']:.3f}")
    for k in ("handcrafted_growth_coefficients", "stacked_coefficients", "gate_beta_vs_diameter_growth_spearman", "gate_beta_mean_malignant_vs_benign"):
        if k in res:
            print(k, res[k])
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "train", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples-per-epoch", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()
    {"pairs": lambda _: build_pairs(), "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
