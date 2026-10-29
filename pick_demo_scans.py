"""Lists good scans to demo the apps (all from data the models never trained on):
  * LUNA16 test subset 9 scans WITH nodules (largest first)   -> .mhd paths
  * LUNA25 test-fold scans with a MALIGNANT / clearly benign nodule -> .mha paths
Also tells you whether given scans (e.g. the ones you already tried) contain nodules:
    python pick_demo_scans.py [seriesuid ...]
"""
import os
import sys

import pandas as pd

import config as C
from luna_io import index_scans

idx = index_scans()
ann = pd.read_csv(C.ANNOTATIONS_CSV)
print("=" * 70)
print("LUNA16 TEST SCANS (subset 9) WITH NODULES - copy a path into 'Add scan from a path on this PC'")
print("=" * 70)
s9 = ann[ann.seriesuid.isin([u for u, (_, s) in idx.items() if s == C.DEFAULT_TEST_SUBSET])]
best = s9.groupby("seriesuid").diameter_mm.agg(["max", "count"]).sort_values("max", ascending=False)
for uid, r in best.head(8).iterrows():
    print(f"  largest nodule {r['max']:4.1f} mm, {int(r['count'])} nodule(s):\n    {idx[uid][0]}")
print(f"\n({len(best)} of {sum(1 for _, s in idx.values() if s == C.DEFAULT_TEST_SUBSET)} subset-9 scans have nodules; the others are normal)")

for uid in sys.argv[1:]:
    uid = os.path.splitext(os.path.basename(uid))[0]
    a = ann[ann.seriesuid == uid]
    sub = idx.get(uid, (None, "?"))[1]
    print(f"\n{uid}: subset {sub}, {len(a)} nodule(s)" + (f", sizes {', '.join(f'{d:.1f}' for d in a.diameter_mm)} mm" if len(a) else " - a normal scan, 0 findings is CORRECT"))

try:
    from luna25_common import L25_DIR, index_images
    f = os.path.join(C.WORK_DIR, "luna25_final", "test_predictions.csv")
    meta = pd.read_csv(f if os.path.exists(f) else os.path.join(L25_DIR, "meta.csv"), dtype={"seriesuid": str})
    te = meta[meta.role == "test"]
    imgs = index_images(C.find_luna25())
    print("\n" + "=" * 70)
    print("LUNA25 TEST SCANS (.mha) - for the cancer-risk demo (never used in training)")
    print("=" * 70)
    mal = te[te.label == 1].sort_values("diameter_pred_mm", ascending=False).drop_duplicates("seriesuid").head(4)
    ben = te[(te.label == 0) & (te.diameter_pred_mm > 6)].sort_values("diameter_pred_mm", ascending=False).drop_duplicates("seriesuid").head(3)
    for name, d in (("MALIGNANT (cancer confirmed)", mal), ("BENIGN", ben)):
        print(f"\n {name}:")
        for _, r in d.iterrows():
            print(f"  nodule ~{r.diameter_pred_mm:.0f} mm at x={r.coordX:.0f} y={r.coordY:.0f} z={r.coordZ:.0f} mm:\n    {imgs.get(r.seriesuid, '?')}")
except Exception as e:
    print("\n(LUNA25 list skipped:", e, ")")
