"""
bench_utils
===========
Shared utilities for the EuroZone GDP Nowcasting challenge.

The dataset is a RAW monthly panel. Each row is one country × one month.
GDP is only non-null on quarter-end months (March, June, September, December).

Participants are responsible for:
  1. Feature engineering / aggregation from the monthly raw data
  2. Variable selection (111 columns, many sparse)
  3. Handling mixed-frequency data

This module only provides I/O helpers — intentionally minimal.

Functions
---------
load_train_data(train_dir)
    Returns (X_raw, y) where X_raw is the raw monthly features DataFrame
    and y is a DataFrame with columns [country, Time, GDP_growth].

load_test_features(input_data_dir, split)
    Returns the raw monthly features DataFrame for 'test' or 'private_test'.

load_labels(reference_dir, split)
    Returns ground-truth labels (for local evaluation only).

Column reference
----------------
Time        : Month timestamp (YYYY-MM-DD), one row per month
country     : ISO country code (AT, BE, DE, EL, ES, FR, IE, IT, NL, PT)

Monthly variables (non-null every month):
  BCI, CCI, SHIX             — confidence / sentiment indices
  HICPOV, HICPG, HICPIN,
  HICPNEF, HICPSV, HICPNG   — HICP inflation components
  UNETOT, UNEO25, UNEU25     — unemployment rates
  LTIRT                      — long-term interest rate
  REER42                     — real effective exchange rate
  ESENTIX, ICONFIX, CCONFIX,
  KCONFIX, RTCONFIX, SCONFIX — economic sentiment sub-indices
  IPMN, IPCAG, IPCOG, ...    — industrial production indices
  TRNMN, TRNCAG, TRNCOG, ... — industrial turnover indices
  PPICAG, PPICOG, ...        — producer price indices

Quarterly variables (non-null only on months 3, 6, 9, 12):
  GDP, EXPGS, IMPGS, GFCE, HFCE, GFCF, GCF
  EMP, SEMP, UNETOT (also monthly), ...
  GGASS, GGLB, HHASS, HHLB, NFCASS, NFCLB, TASS, TLB
  ULC*, RPRP, WS, ESC, TEMP, DFGDP, HPRC, ...

Target
------
GDP_growth : Quarter-on-quarter % change in real GDP
             Defined for each (country, quarter-end month) pair
"""

import os
import pandas as pd
import numpy as np


def load_train_data(train_dir: str):
    """
    Load raw training panel and labels.

    Parameters
    ----------
    train_dir : str
        Path to the folder containing train_features.csv and train_labels.csv.

    Returns
    -------
    X_raw : pd.DataFrame
        Raw monthly panel. Shape ≈ (1620, 113).
        Columns: Time, country, + all macro variables.
    y : pd.DataFrame
        Quarterly labels. Shape ≈ (630, 3).
        Columns: country, Time, GDP_growth.
    """
    X_raw = pd.read_csv(os.path.join(train_dir, "train_features.csv"))
    y     = pd.read_csv(os.path.join(train_dir, "train_labels.csv"))
    X_raw["Time"] = pd.to_datetime(X_raw["Time"])
    y["Time"]     = pd.to_datetime(y["Time"])
    return X_raw, y


def load_test_features(input_data_dir: str, split: str = "test") -> pd.DataFrame:
    """
    Load raw test features.

    Parameters
    ----------
    input_data_dir : str
        Path to the input_data/ directory (parent of train/, test/, private_test/).
    split : str
        'test' or 'private_test'.

    Returns
    -------
    X_raw : pd.DataFrame  — raw monthly panel for the requested split.
    """
    fname = f"{split}_features.csv"
    X_raw = pd.read_csv(os.path.join(input_data_dir, split, fname))
    X_raw["Time"] = pd.to_datetime(X_raw["Time"])
    return X_raw


def load_labels(reference_dir: str, split: str) -> pd.DataFrame:
    """Load ground-truth labels (for local evaluation only, not available on platform)."""
    df = pd.read_csv(os.path.join(reference_dir, f"{split}_labels.csv"))
    df["Time"] = pd.to_datetime(df["Time"])
    return df
