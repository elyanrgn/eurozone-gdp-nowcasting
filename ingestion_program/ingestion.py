"""
ingestion.py
============
Codabench ingestion program for the EuroZone GDP Nowcasting challenge.

Steps:
1. Load train data (features + labels).
2. Import participant's ``get_model()`` from submission.py.
3. Fit the model on train data.
4. Predict on test and private_test sets.
5. Save predictions as CSV files + metadata JSON.

Usage (local test):
    python ingestion_program/ingestion.py \
        --data-dir    dev_phase/input_data \
        --output-dir  ingestion_res \
        --submission-dir solution
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

# Make bench_utils importable from both ingestion_program/ and submission/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_utils import load_train_data, load_test_data

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",       default="dev_phase/input_data")
parser.add_argument("--output-dir",     default="ingestion_res")
parser.add_argument("--submission-dir", default="solution")
args, _ = parser.parse_known_args()

DATA_DIR       = args.data_dir
OUTPUT_DIR     = args.output_dir
SUBMISSION_DIR = args.submission_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load submission ────────────────────────────────────────────────────────────

print(f"[ingestion] Loading submission from: {SUBMISSION_DIR}")
sys.path.insert(0, SUBMISSION_DIR)
try:
    from submission import get_model
except ImportError as e:
    print(f"[ingestion] ERROR: Could not import get_model from submission.py\n  {e}")
    sys.exit(1)

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"[ingestion] Loading train data from: {DATA_DIR}/train")
train_dir = os.path.join(DATA_DIR, "train")
X_train, y_train = load_train_data(train_dir)
print(f"  X_train: {X_train.shape},  y_train: {y_train.shape}")

# ── Train ─────────────────────────────────────────────────────────────────────

print("[ingestion] Calling get_model() and fitting ...")
t0 = time.time()
try:
    model = get_model()
    model.fit(X_train, y_train)
except Exception:
    print("[ingestion] ERROR during model training:")
    traceback.print_exc()
    sys.exit(1)
train_time = time.time() - t0
print(f"  Training done in {train_time:.2f}s")

# ── Predict on test and private_test ─────────────────────────────────────────

metadata = {"train_time_s": round(train_time, 3)}

for split in ["test", "private_test"]:
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"[ingestion] Skipping {split} (directory not found)")
        continue

    print(f"[ingestion] Predicting on {split} ...")
    t1 = time.time()
    X = load_test_data(DATA_DIR, split=split)
    try:
        preds = model.predict(X)
    except Exception:
        print(f"[ingestion] ERROR during prediction on {split}:")
        traceback.print_exc()
        sys.exit(1)
    pred_time = time.time() - t1

    out_path = os.path.join(OUTPUT_DIR, f"{split}_predictions.csv")
    pd.DataFrame({"GDP_growth_pred": preds}).to_csv(out_path, index=False)
    print(f"  Saved {out_path}  ({len(preds)} rows, {pred_time:.2f}s)")
    metadata[f"{split}_predict_time_s"] = round(pred_time, 3)

# ── Save metadata ─────────────────────────────────────────────────────────────

meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(meta_path, "w") as fh:
    json.dump(metadata, fh, indent=2)
print(f"[ingestion] Metadata saved to {meta_path}")
print("[ingestion] Done.")
