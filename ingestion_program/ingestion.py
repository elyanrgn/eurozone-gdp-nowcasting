"""
ingestion.py — Codabench ingestion program for EuroZone GDP Nowcasting.

Calls participant's get_predictions(X_train_raw, y_train, X_test_raw, label_skeleton).

Usage (local):
    python ingestion_program/ingestion.py \
        --data-dir dev_phase/input_data --output-dir ingestion_res --submission-dir solution
"""

import argparse, json, os, sys, time, traceback
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_utils import load_train_data, load_test_features

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",       default="dev_phase/input_data")
parser.add_argument("--output-dir",     default="ingestion_res")
parser.add_argument("--submission-dir", default="solution")
args, _ = parser.parse_known_args()
os.makedirs(args.output_dir, exist_ok=True)

print(f"[ingestion] Loading submission from: {args.submission_dir}")
sys.path.insert(0, args.submission_dir)
try:
    from submission import get_predictions
except ImportError as e:
    print(f"[ingestion] ERROR: Cannot import get_predictions\n  {e}"); sys.exit(1)

print(f"[ingestion] Loading train data from: {args.data_dir}/train")
X_train_raw, y_train = load_train_data(os.path.join(args.data_dir, "train"))
print(f"  X_train_raw: {X_train_raw.shape}  |  y_train: {y_train.shape}")

metadata = {}

for split in ["test", "private_test"]:
    split_dir = os.path.join(args.data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[ingestion] Skipping {split} (directory not found)"); continue

    print(f"[ingestion] Predicting on {split} ...")
    X_test_raw = load_test_features(args.data_dir, split=split)
    print(f"  X_test_raw: {X_test_raw.shape}")

    # Label skeleton: tells participants which (country, quarter) pairs to predict
    skeleton_path = os.path.join(split_dir, f"{split}_labels_skeleton.csv")
    label_skeleton = pd.read_csv(skeleton_path)
    label_skeleton["Time"] = pd.to_datetime(label_skeleton["Time"])
    label_skeleton = label_skeleton.sort_values(["country", "Time"]).reset_index(drop=True)
    print(f"  Expecting {len(label_skeleton)} predictions")

    t0 = time.time()
    try:
        preds = get_predictions(X_train_raw.copy(), y_train.copy(),
                                X_test_raw.copy(), label_skeleton.copy())
    except Exception:
        print(f"[ingestion] ERROR during get_predictions on {split}:")
        traceback.print_exc(); sys.exit(1)
    elapsed = time.time() - t0

    preds = np.asarray(preds, dtype=float).ravel()
    if len(preds) != len(label_skeleton):
        print(f"[ingestion] ERROR: expected {len(label_skeleton)} predictions, got {len(preds)}")
        sys.exit(1)

    out = os.path.join(args.output_dir, f"{split}_predictions.csv")
    pd.DataFrame({"GDP_growth_pred": preds}).to_csv(out, index=False)
    print(f"  Saved {out}  ({len(preds)} rows, {elapsed:.2f}s)")
    metadata[f"{split}_time_s"] = round(elapsed, 3)

with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("[ingestion] Done.")
