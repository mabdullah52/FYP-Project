#!/usr/bin/env bash
# Builds the Hugging Face Space folder from the repository:  bash deploy/huggingface/assemble.sh <out_dir>
set -euo pipefail
OUT=${1:-space}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$OUT" && mkdir -p "$OUT"
cd "$ROOT"
cp deploy/huggingface/hf_app.py deploy/huggingface/Dockerfile deploy/huggingface/README.md deploy/huggingface/.gitattributes "$OUT/"
cp deploy/huggingface/requirements_space.txt deploy/huggingface/persist.py "$OUT/"
for f in config.py luna_io.py candidate_gen.py predictor.py dataset.py model.py baselines.py infer.py explain.py \
         risk_model.py brock.py metrics.py; do cp "$f" "$OUT/"; done
cp -r doctor_app patient_app "$OUT/"
rm -f "$OUT/patient_app/claude_api_key.txt"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$OUT/work/runs"
for r in msgf_f9 l25_risk l25_scratch l25_scratch_s1 l25_scratch_s2; do
  if [ -d "work/runs/$r" ]; then
    mkdir -p "$OUT/work/runs/$r"
    find "work/runs/$r" -maxdepth 1 -type f \( -name '*.json' -o -name 'best.pt' \) -exec cp {} "$OUT/work/runs/$r/" \;
  fi
done
if [ -d doctor_flutter/build/web ]; then
  mkdir -p "$OUT/doctor_flutter/build" && cp -r doctor_flutter/build/web "$OUT/doctor_flutter/build/web"
fi
echo "Space folder ready: $OUT ($(find "$OUT" -type f | wc -l) files)"
