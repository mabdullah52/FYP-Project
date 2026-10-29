"""Redraw the triage figure from saved test predictions (no re-scoring, takes seconds).

    python replot_triage.py                 -> work/runs/<final run>/eval_subset9/triage.png
"""
import json
import os
import sys

import pandas as pd

import config as C
from evaluate import plot_triage

run = sys.argv[1] if len(sys.argv) > 1 else C.FINAL_RUN
d = os.path.join(C.RUNS_DIR, run, f"eval_subset{C.DEFAULT_TEST_SUBSET}")
df = pd.read_csv(os.path.join(d, "predictions.csv"))
res = plot_triage(df.probability.to_numpy(), df["class"].to_numpy(), df.uncertainty.to_numpy(), os.path.join(d, "triage.png"))
print(json.dumps(res, indent=2))
print("->", os.path.join(d, "triage.png"))
