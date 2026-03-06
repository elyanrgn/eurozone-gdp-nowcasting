# EuroZone GDP Nowcasting Challenge

## Overview

Can a machine learning model predict whether an economy is growing or contracting — before the official statistics are released?

In this challenge, you will build a **nowcasting model** for quarterly GDP growth across 9 Euro Area countries: Austria, Belgium, Germany, Greece, Spain, Ireland, Italy, Netherlands, and Portugal.

## The Task

Predict the **quarter-on-quarter GDP growth rate** (%) for each country-quarter combination, using only monthly economic indicators observed *within* that quarter — business confidence, consumer sentiment, inflation, unemployment, interest rates, stock prices, and exchange rates.

This mirrors what economists and central banks do in real time: GDP figures are published weeks after the quarter ends, but high-frequency monthly data arrive much sooner. A good nowcasting model can give an early signal.

## Why It's Interesting

- **Real macroeconomic data** from 9 countries, 2000–2025
- **Temporal challenge**: the private test set includes the **COVID-19 shock** of 2020 Q1–Q2, where GDP fell by up to −18% in a single quarter. Models must generalise beyond normal business cycles.
- **Panel structure**: with 9 countries over 25 years, there's rich cross-country variation to exploit.

## Metric

**RMSE** (Root Mean Squared Error), in percentage points. Lower is better.

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2}$$

The baseline Gradient Boosting model achieves RMSE ≈ 1.32 on the development test set (2016–2019).

## Data Splits

| Split | Period | Rows | Use |
|---|---|---|---|
| Train | 2000 Q2 – 2015 Q4 | 567 | Model fitting |
| Test (dev) | 2016 Q1 – 2019 Q4 | 144 | Leaderboard during competition |
| Private test | 2020 Q1 – 2025 Q3 | 204 | Final scoring |

## How to Submit

Write a `submission.py` file with a `get_model()` function returning a scikit-learn compatible model. See the **Participate** and **Starter Code** tabs for details.
