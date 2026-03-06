"""
bench_utils
===========
Shared utilities between ingestion_program and participant submissions.

Exposes:
- load_train_data(data_dir)   -> (X_train, y_train)
- load_test_data(data_dir)    -> X_test
- FEATURE_COLS                -> ordered list of feature column names
- CATEGORICAL_COLS            -> columns that need encoding
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Ordered list of all feature columns (matches train_features.csv layout)
ID_COLS = ["country", "year", "quarter_end"]

MONTHLY_FEATURES = ["BCI", "CCI", "SHIX", "HICPOV", "UNETOT", "LTIRT", "REER42"]

NUMERIC_COLS = [f"{feat}_m{i}" for feat in MONTHLY_FEATURES for i in range(1, 4)]

CATEGORICAL_COLS = ["country"]

FEATURE_COLS = ID_COLS + NUMERIC_COLS  # all columns in features CSV


def _encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode country as integer and drop non-predictive ID cols."""
    X = X.copy()
    le = LabelEncoder()
    X["country_enc"] = le.fit_transform(X["country"].astype(str))
    X = X.drop(columns=["country", "quarter_end"])  # drop string cols
    return X


def load_train_data(data_dir: str):
    """
    Load training features and labels.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing train_features.csv and train_labels.csv.

    Returns
    -------
    X_train : pd.DataFrame  (encoded, ready for sklearn)
    y_train : np.ndarray    (GDP_growth float values)
    """
    X = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    y_df = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    y = y_df["GDP_growth"].values
    X = _encode_features(X)
    return X, y


def load_test_data(data_dir: str, split: str = "test") -> pd.DataFrame:
    """
    Load test or private_test features.

    Parameters
    ----------
    data_dir : str   Path to the *parent* input_data directory.
    split    : str   Either 'test' or 'private_test'.

    Returns
    -------
    X : pd.DataFrame (encoded, ready for sklearn)
    """
    fname = f"{split}_features.csv"
    folder = os.path.join(data_dir, split)
    X = pd.read_csv(os.path.join(folder, fname))
    return _encode_features(X)
