"""Full-system test on LUNA16 test subset 9, exactly as the doctor app works on a NEW scan:
own candidate finder (candidate_gen.py) -> MSGF-Net (TTA + MC dropout + calibration) -> 6 mm NMS.
No LUNA16 candidate list is used. Resumable (per-scan results are appended).

    python eval_end_to_end.py            -> work/end_to_end/{metrics.json, predictions.csv, per_nodule.csv}
Reports: candidate-finder recall (overall and by nodule size), end-to-end FROC / CPM with 95% CI.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import config as C
from candidate_gen import FINDER_VERSION, generate_candidates
from infer import nms, score_two_stage
from luna_io import Scan, index_scans
from metrics import cpm_with_ci
from predictor import Predictor



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=C.DEFAULT_TEST_SUBSET,
                    help="LUNA16 subset to test on (9 = test; 8 = validation, a check the finder was not tuned on)")
    ap.add_argument("--fast", action="store_true", help="doctor-app fast mode (quick pass + full TTA on top candidates)")
    args = ap.parse_args()
    OUT = os.path.join(C.WORK_DIR, "end_to_end", (FINDER_VERSION if args.subset == C.DEFAULT_TEST_SUBSET
                       else f"{FINDER_VERSION}_subset{args.subset}") + ("_fast" if args.fast else ""))
    os.makedirs(OUT, exist_ok=True)
    idx = index_scans()
    uids = sorted(u for u, (_, s) in idx.items() if s == args.subset)
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    excl = pd.read_csv(C.EXCLUDED_CSV) if C.EXCLUDED_CSV and os.path.exists(C.EXCLUDED_CSV) else None
    pred_path, nod_path = os.path.join(OUT, "predictions.csv"), os.path.join(OUT, "per_nodule.csv")
    done = set(pd.read_csv(nod_path, dtype=str).seriesuid) if os.path.exists(nod_path) else set()
    done_scans_path = os.path.join(OUT, "done_scans.txt")
    done |= set(open(done_scans_path).read().split()) if os.path.exists(done_scans_path) else set()
    model = os.path.join(C.RUNS_DIR, C.FINAL_RUN, "best.pt")
    pred = Predictor(model)
    print(f"model {model} | candidate finder {FINDER_VERSION} | {'FAST' if args.fast else 'full'} scoring | subset {args.subset}: {len(uids)} scans, {len(done)} already done")
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for k, u in enumerate([u for u in uids if u not in done], 1):
            ts = time.time()
            scan = Scan(idx[u][0])
            cands, mask = generate_candidates(scan.image)
            mask_ml = float(mask.sum()) / 1000
            patches = scan.extract_many(cands[["coordX", "coordY", "coordZ"]].to_numpy(), executor=ex)
            s = score_two_stage(pred, patches) if args.fast else pred.score(patches)
            df = cands.copy().reset_index(drop=True)
            df["probability"] = s["prob"]
            df = nms(df)
            df["seriesuid"] = u
            df[["seriesuid", "coordX", "coordY", "coordZ", "probability"]].to_csv(
                pred_path, mode="a", header=not os.path.exists(pred_path), index=False)
            cx = cands[["coordX", "coordY", "coordZ"]].to_numpy()
            rows = []
            for _, a in ann[ann.seriesuid == u].iterrows():
                d = np.linalg.norm(cx - a[["coordX", "coordY", "coordZ"]].to_numpy(float), axis=1) if len(cx) else np.array([np.inf])
                rows.append({"seriesuid": u, "diameter_mm": a.diameter_mm, "candidate_hit": bool((d <= a.diameter_mm / 2).any()),
                             "nearest_candidate_mm": float(d.min()), "n_candidates": len(cands), "mask_ml": mask_ml})
            if rows:
                pd.DataFrame(rows).to_csv(nod_path, mode="a", header=not os.path.exists(nod_path), index=False)
            open(done_scans_path, "a").write(u + "\n")
            el = time.time() - t0
            hit = sum(r["candidate_hit"] for r in rows)
            print(f"  {k}: {len(cands)} candidates, lung mask {mask_ml / 1000:.1f} L{' (lungs not found, searched whole scan)' if mask_ml > 15000 else ''}, nodules found by finder {hit}/{len(rows)}"
                  f" | {time.time() - ts:.0f}s, ~{el / k * (len(uids) - len(done) - k) / 60:.0f} min left")
    nod = pd.read_csv(nod_path) if os.path.exists(nod_path) else pd.DataFrame(columns=["seriesuid", "diameter_mm", "candidate_hit"])
    nod["candidate_hit"] = nod.candidate_hit.astype(str).str.lower().isin(["true", "1"])
    preds = pd.read_csv(pred_path)
    res = {"finder_version": FINDER_VERSION, "subset": args.subset, "n_scans": len(uids),
           "scans_where_lungs_not_found": int((nod.groupby("seriesuid").mask_ml.first() > 15000).sum()) if "mask_ml" in nod else None, "n_nodules": int(len(nod)), "candidate_recall": float(nod.candidate_hit.mean()),
           "mean_candidates_per_scan": float(preds.groupby("seriesuid").size().mean())}
    bins = pd.cut(nod.diameter_mm, [0, 6, 10, 20, 100], right=False, labels=["3-6", "6-10", "10-20", ">=20"])
    res["candidate_recall_by_size_mm"] = {str(b): [int(nod.candidate_hit[bins == b].sum()), int((bins == b).sum())]
                                          for b in bins.cat.categories}
    m, _ = cpm_with_ci(preds, ann, excl, uids)
    res["end_to_end"] = m
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    if args.subset == C.DEFAULT_TEST_SUBSET:
        print("\nFor comparison: with the official LUNA16 candidate list the model scored CPM 0.903 on the same scans.")
    full = os.path.join(C.WORK_DIR, "end_to_end", FINDER_VERSION, "metrics.json")
    if args.fast and args.subset == C.DEFAULT_TEST_SUBSET and os.path.exists(full):
        f = json.load(open(full))["end_to_end"]
        print(f"\nFAST mode CPM {m['cpm']:.4f}  vs  full scoring CPM {f['cpm']:.4f}  (difference {m['cpm'] - f['cpm']:+.4f})")
        el = (time.time() - t0) / max(1, len(uids) - len(done))
        print(f"average time per scan in this run: {el:.0f} s")
    print("->", OUT)


if __name__ == "__main__":
    main()
