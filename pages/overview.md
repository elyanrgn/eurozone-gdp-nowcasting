# EuroZone GDP Nowcasting

## What is this challenge?

Can a machine learning model predict how fast an economy is growing — before the official statistics are even published?

In this competition, you will build a **nowcasting model** for quarterly GDP growth across **10 Euro Area countries**: Austria (AT), Belgium (BE), Germany (DE), Greece (EL), Spain (ES), France (FR), Ireland (IE), Italy (IT), Netherlands (NL), and Portugal (PT).

GDP figures are published weeks after each quarter ends. But high-frequency monthly data — business confidence, inflation, unemployment, interest rates, stock prices — arrive much sooner. A good nowcasting model exploits these signals to estimate GDP growth in real time.

---

## The challenge

This is **not** a standard tabular ML problem. You receive the raw monthly macroeconomic panel as published by Eurostat and the ECB, and you must make all modelling decisions yourself:

- **Which of the 111 variables are useful?** Many are redundant, highly correlated, or missing for some countries.
- **How do you handle mixed-frequency data?** Monthly indicators arrive every month; GDP and most national accounts variables are only available quarterly (columns are `NaN` in non-quarter-end months).
- **How do you build features?** You might aggregate monthly data per quarter, compute lags, rolling windows, year-on-year changes, or differences between countries.
- **How do you handle the panel structure?** 10 countries share the same model — should you use country fixed effects, train per country, or pool everything?

---

## Dataset at a glance

| | |
|---|---|
| **Countries** | AT, BE, DE, EL, ES, FR, IE, IT, NL, PT |
| **Variables** | 111 macroeconomic indicators |
| **Frequency** | Monthly panel (310 months per country, 2000–2025) |
| **Training period** | 2000 Q2 → 2015 Q4 (630 country×quarter labels) |
| **Dev test period** | 2016 Q1 → 2019 Q4 (160 labels, leaderboard) |
| **Final test period** | 2020 Q1 → 2025 Q3 (227 labels, hidden) |
| **Target** | GDP growth (QoQ %, mean ≈ 0.4%, std ≈ 2.2%) |
| **Metric** | RMSE — lower is better |

The variable groups available to you:

- **Confidence & sentiment** — business confidence index (BCI), consumer confidence (CCI), economic sentiment (ESENTIX), and sector-specific indices
- **Inflation** — HICP overall and components (food, energy, services, non-energy goods)
- **Labour market** — total unemployment, youth unemployment (under/over 25), hours worked, employment by sector
- **Financial** — long-term interest rate, real effective exchange rate, stock price index
- **Industrial activity** — industrial production and turnover indices by product category
- **Producer prices** — PPI by category
- **National accounts** *(quarterly only)* — GDP, exports, imports, household consumption, government spending, investment
- **Balance sheets** *(quarterly only)* — financial assets and liabilities of households, non-financial corporations, government

---

## Why it's hard

1. **Mixed frequencies** — you cannot simply feed the raw panel to a standard ML model. You must decide how to align monthly and quarterly signals.
2. **High dimensionality with sparse coverage** — 111 variables but many are `NaN` for specific country-variable combinations. Variable selection matters.
3. **Regime shifts** — the private test set includes the COVID-19 collapse (2020 Q1–Q2, down to −17.8%), the V-shaped rebound (+22.8%), and the 2022 energy shock. A model trained on 2000–2015 must extrapolate far outside its training distribution.
4. **Panel structure** — 10 countries share one model. Structural differences between economies (e.g. Ireland's high volatility from multinational accounting) can hurt naive pooling.

---

## Evaluation

Your model is scored by **RMSE** (root mean squared error) on the dev test set (2016–2019) during the competition, and on the private test set (2020–2025) for final ranking.

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}$$

The **baseline** (simple quarterly aggregation + GradientBoosting) achieves RMSE ≈ **1.13** on the dev test. The theoretical lower bound is 0 (perfect prediction). A naive "always predict the mean" gives RMSE ≈ 2.2.
