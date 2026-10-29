#!/usr/bin/env bash
# Full experiment for the paper, in order. Each line can be run on its own.
# Set these two first:
#   export LUNA16_ROOT=/path/to/LUNA16      (subset0..subset9, annotations.csv, candidates_V2.csv)
#   export WORK_DIR=/path/with/15GB/free
set -e
cd "$(dirname "$0")"

# ---- Day 1: data --------------------------------------------------------------
python preprocess.py --audit            # read the report, check every number
python preprocess.py                    # ~8 GB of patches, resumable if interrupted

# ---- Day 2: proposed model, fold 9 (train 0-7, val 8, test 9) --------------------
python train.py --name msgf_f9 --test-subset 9 --val-subset 8
python mine_hard_negatives.py --run msgf_f9 --top-k 20
python train.py --name msgf_f9_r2 --test-subset 9 --val-subset 8 \
    --init-from msgf_f9 --hard-meta "$WORK_DIR/patches/hard_msgf_f9.csv" --epochs 10 --lr 3e-4
python evaluate.py --run msgf_f9_r2 --calibrate            # MAIN RESULT (CPM, FROC, ECE, triage)

# ---- Day 3: ablations (same fold; single pass so the comparison is architecture-only)
python train.py --name abl_nocontext --no-context
python train.py --name abl_concat    --fusion concat
python train.py --name abl_1window   --windows single
python train.py --name abl_noaux     --aux-weight 0
python train.py --name abl_bce       --loss bce
for r in msgf_f9 abl_nocontext abl_concat abl_1window abl_noaux abl_bce; do
  python evaluate.py --run $r --no-tta --mc 1 --tag _plain --n-boot 200
done
python evaluate.py --run msgf_f9_r2 --no-tta --mc 1 --tag _plain --n-boot 200   # effect of round 2
python evaluate.py --run msgf_f9_r2 --tag _full                                 # effect of TTA+MC

# ---- Day 4 (if GPU time allows): more folds for mean +- std --------------------------
for f in 0 3 6; do
  v=$(( (f + 1) % 10 ))
  python train.py --name msgf_f$f --test-subset $f --val-subset $v
  python evaluate.py --run msgf_f$f --calibrate
done

# ---- candidate detector recall + app smoke test ------------------------------------
python candidate_gen.py --eval-subset 9 --max-scans 30
python collect_results.py
