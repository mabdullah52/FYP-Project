"""Extra analyses for Paper 2 (no training needed): clinical baselines, calibration,
decision curves and prior-scan gain by size, all on the pooled five-fold LUNA25 predictions.

Inputs (already produced by luna25_cv.py):
  work/luna25_cv/cv_nodule_predictions.csv   one row per nodule, out-of-fold scores
  work/luna25_cv/cv_pair_predictions.csv     one row per current/prior pair, out-of-fold scores
  work/luna25/meta.csv                        age, sex, series of every nodule
Outputs: work/paper2_extra/*.csv, extra.json, dca.png, calibration.png
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from brock import brock_risk

W = 'work'
OUT = os.path.join(W, 'paper2_extra')
B = 2000
RNG = np.random.RandomState(2026)


# ---------------------------------------------------------------- data
def load():
    n = pd.read_csv(f'{W}/luna25_cv/cv_nodule_predictions.csv')
    p = pd.read_csv(f'{W}/luna25_cv/cv_pair_predictions.csv')
    m = pd.read_csv(f'{W}/luna25/meta.csv')
    count = m.groupby('seriesuid').size().rename('n_in_scan')
    m = m.join(count, on='seriesuid')
    cols = ['row', 'age', 'gender', 'n_in_scan']
    n = n.merge(m[cols], on='row', how='left')
    p = p.merge(m[cols].rename(columns={'row': 'row_cur'}), on='row_cur', how='left')
    for d, size in ((n, 'diameter_pred_mm'), (p, 'diam_cur')):
        d['brock'] = [brock_risk(a, g == 'Female', False, False, s, 'solid', False, c, False)
                      for a, g, s, c in zip(d.age, d.gender, d[size], d.n_in_scan)]
    n['lungrads'] = n.diameter_pred_mm.map(lungrads_baseline)
    p['lungrads'] = [lungrads_followup(c, q, t) for c, q, t in zip(p.diam_cur, p.diam_prior, p.dt_years)]
    return n, p


def lungrads_baseline(d):
    """Lung-RADS v2022, solid nodule at baseline, by mean diameter (mm)."""
    return 2 if d < 6 else 3 if d < 8 else 4 if d < 15 else 5          # 4 = 4A, 5 = 4B


def lungrads_followup(cur, prior, dt):
    """Lung-RADS v2022, solid nodule seen before. Growth = > 1.5 mm increase in mean diameter
    within 12 months (scaled to the interval). Growing: < 8 mm -> 4A, >= 8 mm -> 4B; otherwise 2."""
    growing = (cur - prior) > 1.5 * max(dt, 1.0)
    if not growing:
        return 2
    return 4 if cur < 8 else 5


# ---------------------------------------------------------------- metrics
def ece(y, s, bins=10):
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - s[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def cal_slope_intercept(y, s):
    from sklearn.linear_model import LogisticRegression
    lo = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6).fit(lo[:, None], y)
    slope = lr.coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1
    from scipy.optimize import brentq
    f = lambda a: (1 / (1 + np.exp(-(lo + a)))).mean() - y.mean()
    return slope, brentq(f, -10, 10)


def spec_at_sens(y, s, target):
    fpr, tpr, thr = roc_curve(y, s)
    i = np.argmax(tpr >= target)
    return 1 - fpr[i], thr[i]


def sens_at_spec(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = (1 - fpr) >= target
    return tpr[ok].max()


def net_benefit(y, s, t):
    pos = s >= t
    n = len(y)
    tp = (pos & (y == 1)).sum() / n
    fp = (pos & (y == 0)).sum() / n
    return tp - fp * t / (1 - t)


def boot(df, fn, b=B):
    """participant-level bootstrap of a statistic fn(df) -> float"""
    groups = df.groupby('patient').indices
    keys = np.array(list(groups))
    vals = []
    for _ in range(b):
        take = RNG.choice(keys, len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in take])
        sub = df.iloc[idx]
        if sub.label.nunique() < 2:
            continue
        vals.append(fn(sub))
    vals = np.array(vals)
    return np.percentile(vals, [2.5, 97.5]), vals


def paired(df, a, b, fn=lambda d, c: roc_auc_score(d.label, d[c])):
    est = fn(df, a) - fn(df, b)
    _, v = boot(df, lambda d: fn(d, a) - fn(d, b))
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return est, max(p, 1 / len(v))


def auc_row(df, col, ref=None):
    y = df.label.values
    a = roc_auc_score(y, df[col])
    ci, _ = boot(df, lambda d: roc_auc_score(d.label, d[col]))
    row = {'model': col, 'auc': a, 'lo': ci[0], 'hi': ci[1]}
    if ref:
        d, p = paired(df, col, ref)
        row.update({'delta_vs_' + ref: d, 'p_vs_' + ref: p})
    return row


# ---------------------------------------------------------------- analyses
def main():
    os.makedirs(OUT, exist_ok=True)
    n, p = load()
    res = {}

    # 1. clinical baselines, all nodules
    rows = [auc_row(n, c, 'model_3d') for c in ('lungrads', 'size_only', 'brock')] + [auc_row(n, 'model_3d')]
    pd.DataFrame(rows).to_csv(f'{OUT}/baselines_all.csv', index=False)
    res['baselines_all'] = rows

    # 2. clinical baselines and prior scan, followed nodules
    rows = [auc_row(p, c, 'stacked') for c in ('lungrads', 'brock', 'single', 'growth', 'tg_msgf')] + [auc_row(p, 'stacked')]
    pd.DataFrame(rows).to_csv(f'{OUT}/baselines_pairs.csv', index=False)
    res['baselines_pairs'] = rows

    # 3. operating points: Lung-RADS 4A+ versus the model at the same specificity / sensitivity
    ops = []
    for name, df, ref_col, models in (('all', n, 'lungrads', ['size_only', 'brock', 'model_3d']),
                                      ('pairs', p, 'lungrads', ['brock', 'single', 'stacked'])):
        y = df.label.values
        pos = df[ref_col] >= 4
        se, sp = pos[y == 1].mean(), (~pos[y == 0]).mean()
        flagged_benign = int((pos & (y == 0)).sum())
        ops.append({'set': name, 'model': 'Lung-RADS 4A+', 'sens': se, 'spec': sp, 'benign_flagged': flagged_benign,
                    'malignant': int(y.sum()), 'benign': int((y == 0).sum())})
        for c in models:
            s = df[c].values
            sens_same_spec = sens_at_spec(y, s, sp)
            spec_same_sens, thr = spec_at_sens(y, s, se)
            ops.append({'set': name, 'model': c, 'sens_at_lungrads_spec': sens_same_spec,
                        'spec_at_lungrads_sens': spec_same_sens,
                        'benign_flagged_at_lungrads_sens': int(((s >= thr) & (y == 0)).sum()),
                        'spec_at_95_sens': spec_at_sens(y, s, 0.95)[0]})
    pd.DataFrame(ops).to_csv(f'{OUT}/operating_points.csv', index=False)
    res['operating_points'] = ops

    # specificity at 95% sensitivity with CIs (all nodules and pairs)
    sp95 = []
    for name, df, cols in (('all', n, ['size_only', 'brock', 'model_3d']), ('pairs', p, ['single', 'stacked'])):
        for c in cols:
            est = spec_at_sens(df.label.values, df[c].values, 0.95)[0]
            ci, _ = boot(df, lambda d: spec_at_sens(d.label.values, d[c].values, 0.95)[0], b=1000)
            sp95.append({'set': name, 'model': c, 'spec_at_95_sens': est, 'lo': ci[0], 'hi': ci[1]})
    pd.DataFrame(sp95).to_csv(f'{OUT}/spec_at_95_sens.csv', index=False)
    res['spec_at_95_sens'] = sp95

    # 4. calibration
    cal = []
    for name, df, cols in (('all', n, ['size_only', 'brock', 'model_3d']), ('pairs', p, ['brock', 'single', 'stacked'])):
        y = df.label.values
        for c in cols:
            s = df[c].values
            sl, ic = cal_slope_intercept(y, s)
            cal.append({'set': name, 'model': c, 'brier': float(((s - y) ** 2).mean()), 'ece': ece(y, s),
                        'slope': sl, 'intercept': ic, 'mean_pred': s.mean(), 'prevalence': y.mean()})
    pd.DataFrame(cal).to_csv(f'{OUT}/calibration.csv', index=False)
    res['calibration'] = cal

    # 5. decision curves
    ts = np.round(np.arange(0.01, 0.301, 0.01), 3)
    dca = {'t': ts.tolist()}
    for name, df, cols in (('all', n, ['size_only', 'brock', 'model_3d']), ('pairs', p, ['single', 'stacked'])):
        y = df.label.values
        dca[name + '_treat_all'] = [y.mean() - (1 - y.mean()) * t / (1 - t) for t in ts]
        for c in cols:
            dca[f'{name}_{c}'] = [net_benefit(y, df[c].values, t) for t in ts]
    pd.DataFrame(dca).to_csv(f'{OUT}/decision_curves.csv', index=False)

    # 6. prior-scan gain by current size group
    grp = []
    bins = [(0, 6, '<6 mm'), (6, 8, '6-8 mm'), (8, 15, '8-15 mm'), (15, 1e9, '>=15 mm')]
    for lo, hi, lab in bins:
        d = p[(p.diam_cur >= lo) & (p.diam_cur < hi)]
        if d.label.sum() < 5 or (d.label == 0).sum() < 5:
            grp.append({'group': lab, 'n': len(d), 'malignant': int(d.label.sum())}); continue
        a1, a2 = roc_auc_score(d.label, d.single), roc_auc_score(d.label, d.stacked)
        delta, pv = paired(d, 'stacked', 'single')
        grp.append({'group': lab, 'n': len(d), 'malignant': int(d.label.sum()), 'auc_single': a1, 'auc_stacked': a2,
                    'delta': delta, 'p': pv})
    pd.DataFrame(grp).to_csv(f'{OUT}/prior_gain_by_size.csv', index=False)
    res['prior_gain_by_size'] = grp

    json.dump(res, open(f'{OUT}/extra.json', 'w'), indent=1, default=float)
    figures(n, p, pd.DataFrame(dca))
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))


def figures(n, p, dca):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
    lab = {'size_only': 'Predicted size', 'brock': 'Brock (reduced)', 'model_3d': '3D network (single scan)',
           'single': 'Single scan', 'stacked': 'With prior scan (stacked)'}
    sty = {'size_only': ('#7f7f7f', '--'), 'brock': ('#d62728', ':'), 'model_3d': ('#1f77b4', '-'),
           'single': ('#1f77b4', '-'), 'stacked': ('#2ca02c', '-')}

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    t = dca.t * 100
    for k, name, cols in ((0, 'all', ['size_only', 'brock', 'model_3d']), (1, 'pairs', ['single', 'stacked'])):
        ax[k].plot(t, dca[name + '_treat_all'], color='k', lw=0.8, label='Refer all')
        ax[k].axhline(0, color='k', lw=0.8, ls='-.', label='Refer none')
        for c in cols:
            ax[k].plot(t, dca[f'{name}_{c}'], color=sty[c][0], ls=sty[c][1], lw=1.4, label=lab[c])
        ax[k].set_ylim(-0.01, max(dca[name + '_treat_all'][0], 0.1) * 1.05)
        ax[k].set_xlabel('Threshold probability (%)'); ax[k].set_ylabel('Net benefit')
        ax[k].set_title('All nodules (6,163)' if name == 'all' else 'Followed nodules (2,401)', fontsize=9)
        ax[k].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/dca.png', dpi=300); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, df, cols, title in ((0, n, ['size_only', 'brock', 'model_3d'], 'All nodules'),
                               (1, p, ['single', 'stacked'], 'Followed nodules')):
        ax[k].plot([0, 0.6], [0, 0.6], color='k', lw=0.8)
        for c in cols:
            s, y = df[c].values, df.label.values
            edges = np.quantile(s, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, 9)
            xs = [s[idx == b].mean() for b in range(10)]; ys = [y[idx == b].mean() for b in range(10)]
            ax[k].plot(xs, ys, marker='o', ms=3, color=sty[c][0], ls=sty[c][1], lw=1.2, label=lab[c])
        ax[k].set_xlabel('Predicted risk'); ax[k].set_ylabel('Observed fraction malignant')
        ax[k].set_title(title, fontsize=9); ax[k].legend(fontsize=7, frameon=False)
        ax[k].set_xlim(0, 0.6); ax[k].set_ylim(0, 0.8)
    fig.tight_layout(); fig.savefig(f'{OUT}/calibration.png', dpi=300); plt.close(fig)


if __name__ == '__main__':
    main()
