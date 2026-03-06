# Starter Code

Copy the template below into your `submission.py`.

## Minimal baseline (Ridge Regression)

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])
```

## Stronger baseline (Gradient Boosting)

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])
```

## Advanced: with custom feature engineering

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MacroFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add momentum and spread features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = X.columns.tolist()
        # BCI momentum: m3 - m1
        bci_cols = [c for c in cols if "BCI" in str(c)]
        if len(bci_cols) == 3:
            X["BCI_momentum"] = X[bci_cols[2]] - X[bci_cols[0]]
        # CCI momentum
        cci_cols = [c for c in cols if "CCI" in str(c)]
        if len(cci_cols) == 3:
            X["CCI_momentum"] = X[cci_cols[2]] - X[cci_cols[0]]
        # Interest rate spread (long - short not available, use level)
        return X.values


def get_model():
    return Pipeline([
        ("engineer", MacroFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42,
        )),
    ])
```

## Running locally

To test your submission before uploading:

```bash
# 1. Generate the data (only once)
python tools/setup_data.py --data-dir /path/to/xlsx/files

# 2. Run ingestion
python ingestion_program/ingestion.py \
    --data-dir    dev_phase/input_data \
    --output-dir  ingestion_res \
    --submission-dir solution   # or your own folder

# 3. Run scoring
python scoring_program/scoring.py \
    --reference-dir dev_phase/reference_data \
    --prediction-dir ingestion_res \
    --output-dir scoring_res

# 4. Check scores
cat scoring_res/scores.json
```
