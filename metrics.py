"""LUNA16 FROC / CPM (re-implementation of the official evaluation logic), calibration and
uncertainty-triage metrics.

FROC rules (Setio et al., 2017, Sec. 3.3):
  * a candidate is a TP if it lies within radius r = diameter/2 of a reference nodule;
    a nodule hit by several candidates takes the highest score;
  * candidates hitting an "irrelevant finding" (annotations_excluded.csv) are ignored;
  * everything else is a false positive;
  * CPM = mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FPs per scan;
  * 95% CI by bootstrapping scans.
"""
import numpy as np
import pandas as pd

import config as C

EXCL_DEFAULT_RADIUS_MM = 5.0   # excluded findings without a diameter (-1 in the csv)


def per_scan_hits(pred, ann, excl, uids):
    """Returns list of (nodule_scores, fp_scores) per scan. Unhit nodules score -inf."""
    out = []
    P = {u: g for u, g in pred.groupby("seriesuid")}
    A = {u: g for u, g in ann.groupby("seriesuid")}
    E = {u: g for u, g in excl.groupby("seriesuid")} if excl is not None else {}
    for u in uids:
        p = P.get(u)
        a = A.get(u)
        pxyz = p[["coordX", "coordY", "coordZ"]].to_numpy() if p is not None else np.zeros((0, 3))
        ps = p["probability"].to_numpy() if p is not None else np.zeros(0)
        used = np.zeros(len(ps), bool)
        nod = []
        if a is not None:
            for _, r in a.iterrows():
                d = np.linalg.norm(pxyz - r[["coordX", "coordY", "coordZ"]].to_numpy(float), axis=1)
                m = d <= r.diameter_mm / 2.0
                used |= m
                nod.append(ps[m].max() if m.any() else -np.inf)
        e = E.get(u)
        if e is not None and len(ps):
            for _, r in e.iterrows():
                rad = r.diameter_mm / 2.0 if r.diameter_mm > 0 else EXCL_DEFAULT_RADIUS_MM
                d = np.linalg.norm(pxyz - r[["coordX", "coordY", "coordZ"]].to_numpy(float), axis=1)
                used |= d <= rad
        out.append((np.array(nod, float), ps[~used]))
    return out


def froc_from_hits(hits, fp_rates=C.FROC_FP_RATES):
    nod = np.concatenate([h[0] for h in hits]) if hits else np.zeros(0)
    fps = np.concatenate([h[1] for h in hits]) if hits else np.zeros(0)
    n_scans, n_nod = len(hits), len(nod)
    thr = np.unique(np.concatenate([nod[np.isfinite(nod)], fps]))[::-1]
    if n_nod == 0 or len(thr) == 0:
        return np.zeros(len(fp_rates)), np.zeros(0), np.zeros(0)
    nod_s = np.sort(nod)
    fps_s = np.sort(fps)
    sens = (n_nod - np.searchsorted(nod_s, thr, side="left")) / n_nod
    fppi = (len(fps_s) - np.searchsorted(fps_s, thr, side="left")) / n_scans
    at = []
    for r in fp_rates:
        ok = fppi <= r
        at.append(sens[ok].max() if ok.any() else 0.0)
    return np.array(at), fppi, sens


def cpm_with_ci(pred, ann, excl, uids, n_boot=1000, seed=0):
    hits = per_scan_hits(pred, ann, excl, uids)
    at, fppi, sens = froc_from_hits(hits)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(hits), len(hits))
        b, _, _ = froc_from_hits([hits[i] for i in idx])
        boots.append(b)
    boots = np.array(boots) if boots else np.zeros((1, len(at)))
    cpm_b = boots.mean(1)
    return {
        "cpm": float(at.mean()),
        "cpm_ci95": [float(np.percentile(cpm_b, 2.5)), float(np.percentile(cpm_b, 97.5))],
        "sensitivity_at_fp": {str(r): float(s) for r, s in zip(C.FROC_FP_RATES, at)},
        "n_nodules": int(sum(len(h[0]) for h in hits)),
        "n_nodules_reachable": int(sum(np.isfinite(h[0]).sum() for h in hits)),
        "n_scans": len(hits),
    }, (fppi, sens)


def expected_calibration_error(p, y, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf, acc = p[m].mean(), y[m].mean()
        ece += m.mean() * abs(conf - acc)
        rows.append((conf, acc, int(m.sum())))
    return float(ece), rows


def triage_curve(p, y, unc, thr=0.5, fractions=np.linspace(0, 0.3, 16)):
    from sklearn.metrics import roc_auc_score
    """Refer the most uncertain fraction of candidates to the radiologist and measure the
    automatic performance on the remaining ones (selective prediction)."""
    order = np.argsort(-unc)
    rows = []
    for f in fractions:
        keep = np.ones(len(p), bool)
        keep[order[: int(round(f * len(p)))]] = False
        pk, yk = p[keep], y[keep]
        tp = ((pk >= thr) & (yk == 1)).sum()
        fn = ((pk < thr) & (yk == 1)).sum()
        fp = ((pk >= thr) & (yk == 0)).sum()
        tn = ((pk < thr) & (yk == 0)).sum()
        rows.append(dict(referred_fraction=float(f),
                         auc=float(roc_auc_score(yk, pk)) if 0 < yk.sum() < len(yk) else float("nan"),
                         sensitivity=float(tp / (tp + fn)) if tp + fn else float("nan"),
                         specificity=float(tn / max(1, tn + fp)),
                         precision=float(tp / max(1, tp + fp)),
                         positives_referred=int((y == 1).sum() - (yk == 1).sum())))
    return pd.DataFrame(rows)
