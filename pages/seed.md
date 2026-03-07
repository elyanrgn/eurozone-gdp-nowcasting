# Starter Code

## Minimal working submission

This is the simplest valid `submission.py`. It aggregates each quarter's monthly data into mean values and trains a Ridge regression.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline


MONTHLY_VARS = [
    "BCI", "CCI", "HICPOV", "UNETOT", "LTIRT",
    "REER42", "SHIX", "ESENTIX",
]


def to_quarter_end(t):
    qe_month = ((t.month - 1) // 3 + 1) * 3
    return t.replace(month=qe_month, day=1)


def aggregate_to_quarters(panel: pd.DataFrame) -> pd.DataFrame:
    """Average monthly variables over each quarter, per country."""
    panel = panel.copy()
    panel["Time"] = pd.to_datetime(panel["Time"])
    panel["quarter_end"] = panel["Time"].apply(to_quarter_end)

    records = []
    for (country, qe), grp in panel.groupby(["country", "quarter_end"]):
        row = {"country": country, "Time": qe}
        for var in MONTHLY_VARS:
            if var in grp.columns:
                vals = pd.to_numeric(grp[var], errors="coerce").dropna()
                row[var] = vals.mean() if len(vals) else np.nan
        records.append(row)
    return pd.DataFrame(records)


def get_predictions(X_train_raw, y_train, X_test_raw, label_skeleton):
    # Aggregate to quarterly
    train_q = aggregate_to_quarters(X_train_raw)
    test_q  = aggregate_to_quarters(X_test_raw)

    # Merge train with labels
    y_train["Time"] = pd.to_datetime(y_train["Time"])
    train_q["Time"] = pd.to_datetime(train_q["Time"])
    merged = train_q.merge(y_train, on=["country", "Time"])
    
    # Encode country
    le = LabelEncoder()
    merged["country_enc"] = le.fit_transform(merged["country"])

    feat_cols = MONTHLY_VARS + ["country_enc"]
    X_tr = merged[feat_cols].fillna(0)
    y_tr = merged["GDP_growth"].values

    # Align test to label_skeleton
    label_skeleton = label_skeleton.copy()
    label_skeleton["Time"] = pd.to_datetime(label_skeleton["Time"])
    test_q["Time"] = pd.to_datetime(test_q["Time"])
    test_merged = label_skeleton.merge(test_q, on=["country", "Time"], how="left")
    test_merged["country_enc"] = le.transform(
        test_merged["country"].map(lambda c: c if c in le.classes_ else le.classes_[0])
    )
    X_te = test_merged[feat_cols].fillna(0)

    model = Pipeline([("sc", StandardScaler()), ("reg", Ridge())])
    model.fit(X_tr, y_tr)
    return model.predict(X_te)
```

---

## Baseline solution (provided)

The `solution/submission.py` in the repo uses a more complete set of features:

- 16 monthly indicators aggregated as mean + momentum (last − first) per quarter
- 23 quarterly national accounts / labour market variables
- Country label encoding
- GradientBoostingRegressor (200 trees, depth 4, LR 0.05)

**Dev test RMSE: 1.13** — use this as your starting point to beat.

---

## Local testing

Before submitting, always test locally:

```bash
# Run ingestion (trains your model, produces predictions)
python ingestion_program/ingestion.py \
    --data-dir    dev_phase/input_data \
    --output-dir  ingestion_res \
    --submission-dir path/to/your/submission/folder

# Run scoring (computes RMSE against ground truth)
python scoring_program/scoring.py \
    --reference-dir  dev_phase/reference_data \
    --prediction-dir ingestion_res \
    --output-dir     scoring_res

# See your scores
cat scoring_res/scores.json
```

Put your `submission.py` alone in a folder and point `--submission-dir` at it.

---

## Ideas to improve the baseline

### Better feature engineering

```python
# Momentum: end-of-quarter minus start-of-quarter
row[f"{var}_momentum"] = vals.iloc[-1] - vals.iloc[0]

# Year-on-year change (needs panel history)
# Rolling standard deviation (uncertainty signal)
row[f"{var}_vol"] = vals.std()

# Use previous quarter's value as a lag feature
# (merge shifted labels back as a feature)
```

### Use quarterly variables too

Many informative variables (exports, employment, household saving rate) are only available at quarter-end. Extract them directly:

```python
QUARTERLY_VARS = ["EXPGS", "IMPGS", "HFCE", "GFCF", "EMP", "UNETOT", "WS"]

qend_rows = panel[panel["Time"].dt.month.isin([3, 6, 9, 12])]
for var in QUARTERLY_VARS:
    row[var] = pd.to_numeric(qend_row[var].values[0], errors="coerce")
```

### Cross-country features

German BCI is a known leading indicator for the whole Euro Area:

```python
# Add Euro Area average BCI alongside the country's own BCI
ea_bci = panel.groupby("Time")["BCI"].mean().rename("BCI_EA_avg")
panel = panel.merge(ea_bci, on="Time")
```

### Better models

```python
# XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)

# LightGBM
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.03)

# Huber loss (robust to COVID outliers)
from sklearn.linear_model import HuberRegressor
model = HuberRegressor(epsilon=1.5, max_iter=500)
```

### Panel-aware cross-validation

Never use future quarters to validate past ones:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# Split on the quarter index, not randomly
```
