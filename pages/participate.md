# How to Participate

## 1. Understand the data format

You receive the **raw monthly macroeconomic panel** for 10 Euro Area countries. Each CSV file has:

- `Time` — month timestamp (YYYY-MM-DD, always the 1st of the month)
- `country` — ISO country code (AT, BE, DE, EL, ES, FR, IE, IT, NL, PT)
- 111 macroeconomic variables

**Key property:** quarterly variables (GDP, employment, national accounts, balance sheets) are only populated on quarter-end months (March, June, September, December). All other months have `NaN` for these columns. Monthly variables (confidence, inflation, unemployment, etc.) are populated every month.

### File layout

```
dev_phase/input_data/train/
    train_features.csv          # raw monthly panel, 1890 rows × 113 cols
    train_labels.csv            # columns: country, Time, GDP_growth (630 rows)
    
dev_phase/input_data/test/
    test_features.csv           # raw monthly panel, 480 rows × 113 cols
    test_labels_skeleton.csv    # columns: country, Time — defines prediction order
```

The `test_labels_skeleton.csv` tells you which `(country, quarter-end month)` pairs you must predict and in what order. **Your predictions must match this ordering exactly.**

---

## 2. Understand the submission API

You submit a single file `submission.py` that must expose this function:

```python
def get_predictions(
    X_train_raw:    pd.DataFrame,   # raw monthly panel, train period
    y_train:        pd.DataFrame,   # quarterly labels (country, Time, GDP_growth)
    X_test_raw:     pd.DataFrame,   # raw monthly panel, test period
    label_skeleton: pd.DataFrame,   # (country, Time) pairs to predict — defines order
) -> array-like:
    ...
    return predictions  # 1-D array of float, len == len(label_skeleton)
```

The ingestion program will call your function with the real data and check that you return exactly `len(label_skeleton)` values in the correct order.

---

## 3. Variable reference

### Monthly variables (~40 columns, always populated)

| Code | Description |
|---|---|
| `BCI` | Business Confidence Index |
| `CCI` | Consumer Confidence Index |
| `ESENTIX` | Economic Sentiment Index |
| `ICONFIX`, `CCONFIX`, `KCONFIX`, `RTCONFIX`, `SCONFIX` | Sector confidence indices (industry, construction, retail, services) |
| `HICPOV` | HICP — overall inflation |
| `HICPG`, `HICPIN`, `HICPNEF`, `HICPSV`, `HICPNG` | HICP components (goods, energy, non-energy, services, non-goods) |
| `UNETOT`, `UNEO25`, `UNEU25` | Unemployment rate (total, over-25, under-25) |
| `LTIRT` | Long-term interest rate (%) |
| `REER42` | Real effective exchange rate (42 trading partners) |
| `SHIX` | Share price index |
| `IPMN`, `IPCAG`, `IPCOG`, `IPDCOG`, `IPNDCOG`, `IPING`, `IPNRG` | Industrial production indices |
| `TRNMN`, `TRNCAG`, `TRNCOG`, `TRNDCOG`, `TRNNDCOG`, `TRNING`, `TRNNRG` | Industrial turnover indices |
| `PPICAG`, `PPICOG`, `PPINDCOG`, `PPIDCOG`, `PPIING`, `PPINRG` | Producer price indices |

### Quarterly variables (~71 columns, NaN on non-quarter-end months)

| Code | Description |
|---|---|
| `GDP` | Real GDP (chain-linked, millions €) |
| `EXPGS`, `IMPGS` | Exports / Imports of goods & services |
| `GFCE` | Government final consumption expenditure |
| `HFCE` | Household final consumption expenditure |
| `CONSD`, `CONSND`, `CONSSD`, `CONSSV` | Consumption sub-components (durable, non-durable, semi-durable, services) |
| `GFCF`, `GCF` | Gross fixed capital formation / Gross capital formation |
| `GFACON`, `GFAMG` | Fixed investment sub-components (construction, machinery) |
| `GNFCPS`, `GNFCIR` | Non-financial corporation saving / investment ratio |
| `EMP`, `SEMP` | Total employment / self-employment |
| `EMP*` (10 codes) | Employment by sector (agriculture, industry, manufacturing, construction, retail, IT, finance, real estate, professional, public) |
| `TEMP`, `THOURS` | Temporary employment, total hours worked |
| `ULC*` (8 codes) | Unit labour costs by sector |
| `WS`, `ESC`, `RPRP` | Wage sum, employer social contributions, real property prices |
| `DFGDP`, `GHIR`, `GHSR` | GDP deflator, household investment/saving rates |
| `HPRC` | House price index |
| `TASS.*`, `TLB.*` | Total financial assets/liabilities (short/long-term, loans/deposits) |
| `GGASS.*`, `GGLB.*` | Government financial assets/liabilities |
| `HHASS.*`, `HHLB.*` | Household financial assets/liabilities |
| `NFCASS.*`, `NFCLB.*` | Non-financial corporation financial assets/liabilities |

**Note:** not all variables are available for all 10 countries. Some columns will be `NaN` for specific country×month combinations even at quarter-end dates.

---

## 4. Things to try

**Feature engineering**
- Aggregate monthly data per quarter: mean, last value, first-to-last change (momentum), rolling volatility
- Compute year-on-year changes to remove seasonality
- Lag features: use indicators from the previous quarter(s) for a forecasting component
- Create cross-country features (e.g. German BCI as a leading indicator for smaller economies)

**Variable selection**
- Many columns are highly correlated — use correlation filtering, PCA, or feature importance
- Some columns have >50% NaN across all countries — consider dropping them
- Quarterly variables that appear in only 6/10 countries add noise for country-pooled models

**Modelling**
- GradientBoosting / XGBoost / LightGBM tend to work well on panel tabular data
- Country fixed effects (one-hot encoding) can capture structural level differences
- Train one model per country vs. one pooled model — both have trade-offs
- Panel-aware cross-validation: never use future quarters to validate a past one

**Robustness to COVID**
- The private test includes extreme observations (−17.8% to +22.8%)
- Consider Huber loss / quantile regression to reduce sensitivity to outliers
- Models that incorporate global shock indicators (Euro Area aggregates) may generalise better

---

## 5. Submitting

1. Write your `submission.py`.
2. Test it locally (see the Starter Code tab).
3. Zip the file: `zip my_submission.zip submission.py`
4. Upload on the **My Submissions** tab.

**Limits:** 100 total submissions, 5 per day. Execution timeout: 5 minutes per submission.
