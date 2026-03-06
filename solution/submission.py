"""
submission.py  –  Baseline solution
=====================================
EuroZone GDP Nowcasting Challenge

This file is the only file you need to submit.
It must expose a ``get_model()`` function that returns a scikit-learn
compatible estimator with ``.fit(X, y)`` and ``.predict(X)`` methods.

Baseline: Gradient Boosting Regressor with light feature engineering.
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_model():
    """
    Return an untrained scikit-learn compatible model.

    The ingestion program will call:
        model = get_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    Returns
    -------
    model : sklearn estimator
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])
    return model


# ── Local test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion_program"))
    from bench_utils import load_train_data, load_test_data
    from sklearn.metrics import mean_squared_error
    import math

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dev_phase", "input_data")
    X_train, y_train = load_train_data(os.path.join(DATA_DIR, "train"))
    X_test  = load_test_data(DATA_DIR, split="test")

    model = get_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # Load ground truth for local eval
    import pandas as pd
    y_true = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "dev_phase", "reference_data", "test_labels.csv")
    )["GDP_growth"].values

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Local test RMSE: {rmse:.4f}")
