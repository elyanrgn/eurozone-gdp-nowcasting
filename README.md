# EuroZone GDP Nowcasting — Codabench Competition

[![CI](https://github.com/elyanrgn/eurozone-gdp-nowcasting/actions/workflows/ci.yml/badge.svg)](https://github.com/elyanrgn/eurozone-gdp-nowcasting/actions)

> **DataCamp Challenge · HI-Paris**

## Overview

Can a machine learning model anticipate whether an economy is growing or contracting — before official statistics are released?

This competition challenges you to build a **GDP nowcasting model** for 10 Euro Area countries using raw monthly macroeconomic panel data. Unlike typical ML competitions, the dataset is not pre-processed: you receive the raw time series as published by Eurostat/ECB and must make all design decisions yourself — which variables to use, how to handle mixed-frequency data, and how to engineer features that generalise across economic regimes.

---

## Task

**Predict the quarter-on-quarter GDP growth rate (%)** for each `(country, quarter)` pair in the test set.

- **Type:** Regression
- **Primary metric:** RMSE — lower is better
- **Countries:** AT, BE, DE, EL, ES, FR, IE, IT, NL, PT (10 Euro Area members)
- **Training period:** 2000 Q2 → 2015 Q4

---

## Dataset

The dataset is a **raw monthly panel**: one row per country per month, with **111 macroeconomic variables** spanning prices, labour markets, financial conditions, confidence indices, national accounts, and balance sheets.

### Mixed-frequency structure

This is the central challenge of the competition:

- **Monthly variables (~40)** are populated every single month — confidence indices, inflation, unemployment, interest rates, exchange rates, industrial production, etc.
- **Quarterly variables (~71)** are only populated on quarter-end months (March, June, September, December) — GDP, employment, national accounts, balance sheet data, etc.
- All other cells for quarterly variables contain `NaN`.

You must decide how to handle this: aggregate monthly data per quarter, impute, use the raw monthly structure, or something else entirely.

### Splits

| Split | Monthly rows | Labels (country×quarter) | Period |
|---|---|---|---|
| **train** | 1,890 | 630 | 2000-04 → 2015-12 |
| **test** *(dev leaderboard)* | 480 | 160 | 2016-01 → 2019-12 |
| **private_test** *(final scoring)* | 690 | 227 | 2020-01 → 2025-09 |

The private test set includes the **COVID-19 shock** (2020 Q1–Q2, GDP down to −17.8%), the post-COVID rebound (+22.8%), and the 2022 energy crisis. Models trained only on "normal" business cycle variation will generalise poorly.

### Variable groups

**Monthly variables** (non-null every month):

| Group | Variables |
|---|---|
| Confidence & sentiment | `BCI`, `CCI`, `ESENTIX`, `ICONFIX`, `CCONFIX`, `KCONFIX`, `RTCONFIX`, `SCONFIX` |
| Inflation (HICP) | `HICPOV`, `HICPG`, `HICPIN`, `HICPNEF`, `HICPSV`, `HICPNG` |
| Labour market | `UNETOT`, `UNEO25`, `UNEU25` |
| Financial | `LTIRT`, `REER42`, `SHIX` |
| Industrial production | `IPMN`, `IPCAG`, `IPCOG`, `IPDCOG`, `IPNDCOG`, `IPING`, `IPNRG` |
| Industrial turnover | `TRNMN`, `TRNCAG`, `TRNCOG`, `TRNDCOG`, `TRNNDCOG`, `TRNING`, `TRNNRG` |
| Producer prices (PPI) | `PPICAG`, `PPICOG`, `PPINDCOG`, `PPIDCOG`, `PPIING`, `PPINRG` |

**Quarterly variables** (non-null only on months 3, 6, 9, 12):

| Group | Variables |
|---|---|
| National accounts | `GDP`, `EXPGS`, `IMPGS`, `GFCE`, `HFCE`, `GFCF`, `GCF`, `CONSD`, `CONSND`, `CONSSD`, `CONSSV` |
| Investment | `GFACON`, `GFAMG`, `GNFCPS`, `GNFCIR` |
| Labour (quarterly) | `EMP`, `SEMP`, `EMPAG`, `EMPIN`, `EMPMN`, `EMPCON`, `EMPRT`, `EMPIT`, `EMPFC`, `EMPRE`, `EMPPR`, `EMPPA`, `EMPENT`, `TEMP`, `THOURS` |
| Unit labour costs | `ULCIN`, `ULCMN`, `ULCMQ`, `ULCCON`, `ULCRT`, `ULCFC`, `ULCRE`, `ULCPR` |
| Saving & income | `WS`, `ESC`, `RPRP`, `DFGDP`, `GHIR`, `GHSR` |
| Balance sheets | `TASS.*`, `TLB.*`, `GGASS.*`, `GGLB.*`, `HHASS.*`, `HHLB.*`, `NFCASS.*`, `NFCLB.*` |
| House prices | `HPRC` |

Note: not all variables are available for all 10 countries — some columns will be `NaN` for certain country×month combinations even at quarter-end dates.

### Target

`GDP_growth` — QoQ % change in real GDP. Stats across all 1,017 labels:
mean = 0.41%, std = 2.22%, min = −17.8% (Ireland 2008 or COVID), max = +22.8%.

---

## Submission format

Submit a single file **`submission.py`** exposing:

```python
def get_predictions(
    X_train_raw:    pd.DataFrame,   # raw monthly panel, train (~1890 rows × 113 cols)
    y_train:        pd.DataFrame,   # labels: columns [country, Time, GDP_growth]
    X_test_raw:     pd.DataFrame,   # raw monthly panel, test  (~480 rows × 113 cols)
    label_skeleton: pd.DataFrame,   # columns [country, Time] — defines prediction order
) -> array-like:                    # 1-D float array, len == len(label_skeleton)
```

`label_skeleton` lists exactly which `(country, quarter-end month)` pairs to predict and in what order. Your returned array must have **exactly `len(label_skeleton)` values** in the same row order.

---

## Repository structure

```
competition.yaml                     Codabench configuration (version 2)
ingestion_program/
    ingestion.py                     Calls get_predictions(), saves prediction CSVs
    metadata.yaml
    bench_utils/__init__.py          I/O helpers: load_train_data, load_test_features
scoring_program/
    scoring.py                       Computes RMSE / MAE / R², writes scores.json
    metadata.yaml
solution/
    submission.py                    Baseline solution (GradientBoostingRegressor)
pages/
    overview.md  participate.md  seed.md  timeline.md  terms.md
tools/
    setup_data.py                    Builds dataset CSVs from raw xlsx files
    create_bundle.py                 Packages bundle.zip for Codabench upload
    run_docker.py                    Local Docker end-to-end test
    Dockerfile
dev_phase/                           Generated by setup_data.py (git-ignored)
template_starting_kit.ipynb          EDA + baseline walkthrough notebook
requirements.txt
```

---

## Quick start

```bash
# 1. Generate data splits from raw xlsx files
python tools/setup_data.py --data-dir /path/to/xlsx/files/

# 2. Zip the data (required once after generating)
cd dev_phase
zip -r input_data.zip input_data/ && zip -r reference_data.zip reference_data/
cd ..

# 3. Run the full local pipeline
python ingestion_program/ingestion.py \
    --data-dir    dev_phase/input_data \
    --output-dir  ingestion_res \
    --submission-dir solution

python scoring_program/scoring.py \
    --reference-dir  dev_phase/reference_data \
    --prediction-dir ingestion_res \
    --output-dir     scoring_res

cat scoring_res/scores.json

# 4. Build the Codabench upload bundle
python tools/create_bundle.py   # → bundle.zip
```

---

## Baseline results

Baseline strategy: aggregate monthly indicators (mean + diff over the quarter) → GradientBoostingRegressor.

| Split | Period | RMSE | MAE | R² |
|---|---|---|---|---|
| test | 2016–2019 | **1.13** | 0.55 | −0.23 |
| private_test | 2020–2025 | 3.94 | 2.04 | ~0.00 |

The private test RMSE is dominated by the COVID shock. Better feature engineering, variable selection, robust models, and panel-aware cross-validation can substantially improve on both splits.

---

## Data sources

Raw data: ECB / Eurostat harmonised macroeconomic indicators for 10 Euro Area member states, provided as part of the DataCamp course materials at CentraleSupélec / HI-Paris.
