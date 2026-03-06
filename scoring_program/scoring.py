"""
scoring.py
==========
Codabench scoring program for the EuroZone GDP Nowcasting challenge.

Metrics reported:
  - RMSE  (root mean squared error)  – primary leaderboard metric (lower is better)
  - MAE   (mean absolute error)
  - R²    (coefficient of determination)

Usage (local test):
    python scoring_program/scoring.py \
        --reference-dir dev_phase/reference_data \
        --prediction-dir ingestion_res \
        --output-dir scoring_res
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--reference-dir",  default="dev_phase/reference_data")
parser.add_argument("--prediction-dir", default="ingestion_res")
parser.add_argument("--output-dir",     default="scoring_res")
args, _ = parser.parse_known_args()

REF_DIR  = args.reference_dir
PRED_DIR = args.prediction_dir
OUT_DIR  = args.output_dir

os.makedirs(OUT_DIR, exist_ok=True)


# ── Metric helpers ────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── Score each split ──────────────────────────────────────────────────────────

scores = {}

for split in ["test", "private_test"]:
    ref_path  = os.path.join(REF_DIR,  f"{split}_labels.csv")
    pred_path = os.path.join(PRED_DIR, f"{split}_predictions.csv")

    if not os.path.exists(ref_path):
        print(f"[scoring] Reference file not found for {split}: {ref_path}")
        continue
    if not os.path.exists(pred_path):
        print(f"[scoring] Prediction file not found for {split}: {pred_path}")
        continue

    y_true = pd.read_csv(ref_path)["GDP_growth"].values.astype(float)
    y_pred = pd.read_csv(pred_path)["GDP_growth_pred"].values.astype(float)

    if len(y_true) != len(y_pred):
        print(f"[scoring] ERROR: length mismatch on {split}: "
              f"expected {len(y_true)}, got {len(y_pred)}")
        sys.exit(1)

    split_rmse = rmse(y_true, y_pred)
    split_mae  = mae(y_true, y_pred)
    split_r2   = r2(y_true, y_pred)

    print(f"[scoring] {split:>13} | RMSE={split_rmse:.4f}  MAE={split_mae:.4f}  R²={split_r2:.4f}")

    scores[f"{split}_RMSE"] = round(split_rmse, 6)
    scores[f"{split}_MAE"]  = round(split_mae,  6)
    scores[f"{split}_R2"]   = round(split_r2,   6)

# ── Primary leaderboard score (test RMSE) ────────────────────────────────────
# Codabench uses the key "score" as the primary column
if "test_RMSE" in scores:
    scores["score"] = scores["test_RMSE"]
elif "private_test_RMSE" in scores:
    scores["score"] = scores["private_test_RMSE"]
else:
    scores["score"] = 999.0

# ── Write scores.json ─────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "scores.json")
with open(out_path, "w") as fh:
    json.dump(scores, fh, indent=2)
print(f"[scoring] scores.json saved to {out_path}")
print(f"[scoring] Primary score (RMSE) = {scores['score']:.4f}")
