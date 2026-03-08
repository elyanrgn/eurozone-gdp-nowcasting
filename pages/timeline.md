# Competition Timeline

| Date | Event |
|---|---|
| **March 8, 2026** | Competition opens — development phase begins |
| **March 8 → March 28, 2026** | **Development phase:** submit against the 2016–2019 test set. Leaderboard shows RMSE on dev test. |
| **March 28, 2026 (23:59 UTC)** | Development phase closes — no more submissions accepted |
| **March 28, 2026** | Final scoring: model evaluated on the private test set (2020–2025, includes COVID shock) |
| **April 4, 2026** | Final leaderboard published — winners announced |

---

## Submission limits

- **100 submissions** total per team during the development phase
- **5 submissions per day** per team maximum
- Each submission runs for at most **5 minutes** on the platform

---

## Data splits reminder

| Phase | Features | Labels | Period |
|---|---|---|---|
| Development | `test_features.csv` (480 monthly rows) | Hidden during competition | 2016 Q1 → 2019 Q4 |
| Final | `private_test_features.csv` (690 monthly rows) | Hidden until final scoring | 2020 Q1 → 2025 Q3 |

The private test set includes **the COVID-19 crisis** (2020 Q1–Q2), the post-COVID rebound (2020 Q3–2021), the energy price shock (2022), and the low-growth environment of 2023–2025. Models that learn only from normal business cycles (2000–2015) will struggle.
