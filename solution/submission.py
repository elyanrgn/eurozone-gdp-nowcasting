"""
submission.py — Baseline solution
===================================
EuroZone GDP Nowcasting Challenge

This is the ONLY file you submit. It must expose:

    get_predictions(X_train_raw, y_train, X_test_raw) -> array-like

Parameters
----------
X_train_raw : pd.DataFrame
    Raw monthly panel for the training period.
    Shape ≈ (1620, 113). Columns: Time, country, + all macro variables.
    GDP is non-null only on quarter-end months (months 3, 6, 9, 12).

y_train : pd.DataFrame
    Ground-truth labels for the training period.
    Columns: country, Time, GDP_growth (QoQ %).
    One row per (country, quarter-end month).  Shape ≈ (630, 3).

X_test_raw : pd.DataFrame
    Raw monthly panel for the test period. Same structure as X_train_raw.
    Shape ≈ (360, 113) for the dev test split.

Returns
-------
predictions : 1-D array-like of float
    GDP_growth predictions, one value per (country, quarter-end month) pair
    in the test period. Must be in the same order as quarter-end rows sorted
    by (country, Time).

Baseline strategy
-----------------
For each quarter, aggregate the 3 monthly observations into summary stats
(mean of each indicator over the quarter), then train a GradientBoostingRegressor.
This is intentionally simple — participants are expected to do much better with:
  - Better feature engineering (lags, diffs, rolling windows)
  - Variable selection (many columns are redundant or missing)
  - Country fixed effects, panel-aware cross-validation
  - Models robust to the COVID-19 shock (2020 Q1-Q2)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Feature engineering ───────────────────────────────────────────────────────

# Monthly variables to aggregate across the 3 months of each quarter
MONTHLY_VARS = [
    "BCI", "CCI", "SHIX",
    "HICPOV", "HICPG", "HICPIN",
    "UNETOT", "UNEO25",
    "LTIRT", "REER42",
    "ESENTIX", "ICONFIX", "CCONFIX", "KCONFIX", "RTCONFIX", "SCONFIX",
]

# Quarterly variables (already one value per quarter, take the quarter-end value)
QUARTERLY_VARS = [
    "EXPGS", "IMPGS", "GFCE", "HFCE", "GFCF",
    "EMP", "SEMP", "ULCIN", "ULCMN", "ULCFC", "ULCPR", "ULCRT",
    "WS", "ESC", "TEMP", "DFGDP", "RPRP",
    "GGASS", "GGLB", "HHASS", "HHLB", "NFCASS", "NFCLB",
]


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw monthly panel into one row per (country, quarter).

    For monthly variables: compute mean and last-minus-first diff across the quarter.
    For quarterly variables: take the value at the quarter-end month.

    Returns a DataFrame with columns: country, Time (=quarter-end), + all features.
    Sorted by (country, Time).
    """
    panel = panel.copy()
    panel["Time"] = pd.to_datetime(panel["Time"])

    def to_quarter_end(t):
        qe_month = ((t.month - 1) // 3 + 1) * 3
        return t.replace(month=qe_month, day=1)

    panel["quarter_end"] = panel["Time"].apply(to_quarter_end)

    records = []
    for (country, qe), grp in panel.groupby(["country", "quarter_end"]):
        row = {"country": country, "Time": qe}
        # Monthly aggregation
        for var in MONTHLY_VARS:
            if var in grp.columns:
                vals = pd.to_numeric(grp[var], errors="coerce").dropna()
                row[f"{var}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{var}_diff"] = (vals.iloc[-1] - vals.iloc[0]) if len(vals) >= 2 else np.nan
        # Quarterly: take the quarter-end month value
        qend_row = grp[grp["Time"] == qe]
        if len(qend_row) == 0:
            qend_row = grp.iloc[[-1]]
        for var in QUARTERLY_VARS:
            if var in qend_row.columns:
                row[f"{var}"] = pd.to_numeric(qend_row[var].values[0], errors="coerce") \
                                if len(qend_row) else np.nan
        records.append(row)

    feat_df = pd.DataFrame(records).sort_values(["country", "Time"]).reset_index(drop=True)
    return feat_df  # country and Time are kept for merging


def get_X_y(feat_df: pd.DataFrame, labels: pd.DataFrame):
    """Merge engineered features with labels on (country, Time), return X and y."""
    labels = labels.copy()
    labels["Time"] = pd.to_datetime(labels["Time"])
    feat_df = feat_df.copy()
    feat_df["Time"] = pd.to_datetime(feat_df["Time"])
    merged = feat_df.merge(labels, on=["country", "Time"], how="inner")
    y = merged["GDP_growth"].values
    X = merged.drop(columns=["Time", "GDP_growth", "country"])
    # Encode country (it was dropped above — re-add encoded version)
    le = LabelEncoder()
    X = merged.drop(columns=["Time", "GDP_growth"])
    X["country_enc"] = le.fit_transform(X["country"].astype(str))
    X = X.drop(columns=["country"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return X, y


# ── Main entry point ──────────────────────────────────────────────────────────

def get_predictions(X_train_raw: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_test_raw: pd.DataFrame,
                    label_skeleton: pd.DataFrame) -> np.ndarray:
    """
    Full pipeline: feature engineering → model fit → predict.

    Parameters
    ----------
    X_train_raw     : raw monthly panel for training (shape ~1890 × 113)
    y_train         : quarterly labels (country, Time, GDP_growth)
    X_test_raw      : raw monthly panel for test period (shape ~480 × 113)
    label_skeleton  : DataFrame with columns [country, Time] listing the
                      (country, quarter-end) pairs to predict, in order.
                      Your predictions must match this ordering exactly.

    Returns
    -------
    1-D array of float, length = len(label_skeleton)
    """
    # --- Engineer features ---
    train_feats = engineer_features(X_train_raw)
    test_feats  = engineer_features(X_test_raw)

    # --- Build train X/y ---
    X_train, y = get_X_y(train_feats, y_train)
    feat_cols = X_train.columns.tolist()

    # --- Build test X aligned to label_skeleton order ---
    label_skeleton = label_skeleton.copy()
    label_skeleton["Time"] = pd.to_datetime(label_skeleton["Time"])
    test_feats["Time"] = pd.to_datetime(test_feats["Time"])

    # Merge on (country, Time) to get one row per skeleton entry, in skeleton order
    test_merged = label_skeleton.merge(test_feats, on=["country", "Time"], how="left")

    # Encode country
    le = LabelEncoder()
    le.fit(list(X_train_raw["country"].unique()) + list(X_test_raw["country"].unique()))
    test_merged["country_enc"] = le.transform(test_merged["country"].astype(str))

    X_test = test_merged.drop(columns=["Time", "country", "GDP_growth"], errors="ignore")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        if c not in X_test.columns:
            X_test[c] = X_train[c].median()
    X_test = X_test[feat_cols].fillna(X_train.median())

    # --- Model ---
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])
    model.fit(X_train, y)
    return model.predict(X_test)


# ── Local test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys, math
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion_program"))
    from bench_utils import load_train_data, load_test_features, load_labels
    from sklearn.metrics import mean_squared_error

    root = os.path.join(os.path.dirname(__file__), "..")
    DATA_DIR = os.path.join(root, "dev_phase", "input_data")
    REF_DIR  = os.path.join(root, "dev_phase", "reference_data")

    X_train_raw, y_train = load_train_data(os.path.join(DATA_DIR, "train"))
    X_test_raw = load_test_features(DATA_DIR, split="test")
    y_test_true = load_labels(REF_DIR, "test")

    label_skeleton = y_test_true[["country", "Time"]].sort_values(["country", "Time"]).reset_index(drop=True)
    preds = get_predictions(X_train_raw, y_train, X_test_raw, label_skeleton)

    # Order must match labels sorted by (country, Time)
    y_true = y_test_true.sort_values(["country", "Time"])["GDP_growth"].values
    rmse = math.sqrt(mean_squared_error(y_true, preds))
    print(f"Local test RMSE: {rmse:.4f}")
