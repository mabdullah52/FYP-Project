"""Paired bootstrap test of CPM differences between runs (same scans resampled for both).

    python stats_compare.py                 -> WORK_DIR/significance_table.csv
Answers "is variant B really better/worse than A, or is it noise?" for the paper.
No GPU needed; uses the saved predictions.csv files.
"""
import os

import numpy as np
import pandas as pd

import config as C
from luna_io import index_scans
from metrics import froc_from_hits, per_scan_hits

# (label, run A, eval A, run B, eval B) -> difference reported as B - A
COMPARISONS = [
    ("Ablation: no context stream", "abl_full", "eval_subset9_plain", "abl_nocontext", "eval_subset9_plain"),
    ("Ablation: concat instead of gated fusion", "abl_full", "eval_subset9_plain", "abl_concat", "eval_subset9_plain"),
    ("Ablation: single HU window", "abl_full", "eval_subset9_plain", "abl_1window", "eval_subset9_plain"),
    ("Ablation: no diameter task", "abl_full", "eval_subset9_plain", "abl_noaux", "eval_subset9_plain"),
    ("Ablation: BCE instead of focal", "abl_full", "eval_subset9_plain", "abl_bce", "eval_subset9_plain"),
    ("TTA + MC dropout (test)", "msgf_f9", "eval_subset9_plain", "msgf_f9", "eval_subset9"),
    ("TTA + MC dropout (validation)", "msgf_f9", "eval_subset8_plain", "msgf_f9", "eval_subset8"),
    ("Round-2 hard negatives (validation)", "msgf_f9", "eval_subset8_plain", "msgf_f9_r2", "eval_subset8_plain"),
    ("Round-2 hard negatives (test)", "msgf_f9", "eval_subset9_plain", "msgf_f9_r2", "eval_subset9_plain"),
    ("Training length 10 vs 30 epochs (test)", "msgf_f9", "eval_subset9_plain", "abl_full", "eval_subset9_plain"),
]


def load_hits(run, ev, ann, excl, scan_index):
    p = os.path.join(C.RUNS_DIR, run, ev, "predictions.csv")
    if not os.path.exists(p):
        return None, None
    subset = int(ev[len("eval_subset"):].split("_")[0])
    uids = sorted(u for u, (_, s) in scan_index.items() if s == subset)
    return per_scan_hits(pd.read_csv(p), ann, excl, uids), uids


def paired_cpm(eval_dir_a, eval_dir_b, n_boot=2000, seed=0):
    """CPM(b) - CPM(a) for two evaluation folders on the same subset, scan-level paired bootstrap."""
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if C.EXCLUDED_CSV and os.path.exists(C.EXCLUDED_CSV) else None
    scan_index = index_scans()
    subset = int(os.path.basename(os.path.normpath(eval_dir_a))[len("eval_subset"):].split("_")[0])
    uids = sorted(u for u, (_, s) in scan_index.items() if s == subset)
    ha = per_scan_hits(pd.read_csv(os.path.join(eval_dir_a, "predictions.csv")), ann, excl, uids)
    hb = per_scan_hits(pd.read_csv(os.path.join(eval_dir_b, "predictions.csv")), ann, excl, uids)
    cpm = lambda h: froc_from_hits(h)[0].mean()
    rng, n = np.random.RandomState(seed), len(ha)
    d = np.array([cpm([hb[j] for j in idx]) - cpm([ha[j] for j in idx]) for idx in (rng.randint(0, n, n) for _ in range(n_boot))])
    return {"delta": float(cpm(hb) - cpm(ha)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def main(n_boot=2000, seed=0):
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if os.path.exists(C.EXCLUDED_CSV) else None
    scan_index = index_scans()
    rng = np.random.RandomState(seed)
    rows = []
    for label, ra, ea, rb, eb in COMPARISONS:
        ha, ua = load_hits(ra, ea, ann, excl, scan_index)
        hb, ub = load_hits(rb, eb, ann, excl, scan_index)
        if ha is None or hb is None or ua != ub:
            print(f"skip (missing predictions): {label}")
            continue
        cpm = lambda h: froc_from_hits(h)[0].mean()
        a, b = cpm(ha), cpm(hb)
        n = len(ha)
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.randint(0, n, n)
            diffs[i] = cpm([hb[j] for j in idx]) - cpm([ha[j] for j in idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        rows.append({"comparison": label, "CPM_A": round(a, 4), "CPM_B": round(b, 4), "delta_B_minus_A": round(b - a, 4),
                     "delta_95CI": f"{lo:+.3f} to {hi:+.3f}", "p_value": round(p, 3),
                     "significant_at_0.05": bool(p < 0.05)})
        print(f"{label:45s} {a:.4f} -> {b:.4f}  delta {b - a:+.4f}  [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    out = os.path.join(C.WORK_DIR, "significance_table.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
