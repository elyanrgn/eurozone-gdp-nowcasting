# How to Participate

## 1. Understanding the Task

You must predict the **quarter-on-quarter GDP growth rate** (in %) for 9 Euro Area countries:
Austria (AT), Belgium (BE), Germany (DE), Greece (EL), Spain (ES), Ireland (IE),
Italy (IT), Netherlands (NL), and Portugal (PT).

Each sample in the dataset corresponds to **one country × one quarter**.  
The features are monthly macro indicators observed during the 3 months of that quarter.

**This is a regression task.** The primary metric is **RMSE** (lower is better).

---

## 2. Dataset Description

### Features (24 columns)

| Column group | Description |
|---|---|
| `country` | ISO country code (AT, BE, DE, EL, ES, IE, IT, NL, PT) |
| `year` | Year of the quarter |
| `quarter_end` | Date of the last month of the quarter (YYYY-MM-DD) |
| `BCI_m1/m2/m3` | Business Confidence Index — months 1, 2, 3 of the quarter |
| `CCI_m1/m2/m3` | Consumer Confidence Index |
| `SHIX_m1/m2/m3` | Share price index |
| `HICPOV_m1/m2/m3` | HICP overall inflation index |
| `UNETOT_m1/m2/m3` | Total unemployment rate (%) |
| `LTIRT_m1/m2/m3` | Long-term interest rate (%) |
| `REER42_m1/m2/m3` | Real effective exchange rate (42 partners) |

### Target

`GDP_growth` — quarter-on-quarter % change in real GDP.

### Data splits

| Split | Period | Rows | Purpose |
|---|---|---|---|
| train | 2000 Q2 – 2015 Q4 | 567 | Model training |
| test | 2016 Q1 – 2019 Q4 | 144 | Dev-phase leaderboard |
| private_test | 2020 Q1 – 2025 Q3 | 204 | Final scoring (includes COVID shock) |

---

## 3. Submission Format

Submit a single Python file named **`submission.py`** that exposes one function:

```python
def get_model():
    """Return an untrained scikit-learn compatible estimator."""
    ...
    return model
```

The model must implement `.fit(X, y)` and `.predict(X)`.

---

## 4. Submitting

1. Write your `submission.py`.
2. Zip it (just the file, not a folder): `zip my_submission.zip submission.py`
3. Upload on the *My Submissions* tab.

---

## 5. Tips for Good Performance

- Feature engineering: try rolling statistics, quarter-over-quarter differences, or country interaction terms.
- The private test set includes the COVID-19 shock (2020 Q1–Q2). Models that generalise to structural breaks will score better.
- Ensemble methods (Random Forest, Gradient Boosting, XGBoost) tend to work well on this kind of tabular panel data.
- Country fixed effects (one-hot encoding of `country`) can capture structural differences.

Good luck!
