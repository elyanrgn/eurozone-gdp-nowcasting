"""
setup_data.py
=============
Builds the competition dataset from the raw Euro Area country Excel files.

Usage:
    python tools/setup_data.py [--data-dir <path_to_xlsx_folder>]

The script:
1. Reads the 9-country panel of monthly macro indicators.
2. Constructs quarterly nowcasting samples (features = 3 monthly obs per quarter).
3. Splits by time into train / test / private_test to avoid leakage.
4. Saves CSVs in the expected Codabench layout:
   dev_phase/input_data/train/     -> train_features.csv, train_labels.csv
   dev_phase/input_data/test/      -> test_features.csv
   dev_phase/input_data/private_test/ -> private_test_features.csv
   dev_phase/reference_data/       -> test_labels.csv, private_test_labels.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

COUNTRIES = {
    "AT": "ATdata.xlsx",
    "BE": "BEdata.xlsx",
    "DE": "DEdata.xlsx",
    "EL": "ELdata.xlsx",
    "ES": "ESdata.xlsx",
    "IE": "IEdata.xlsx",
    "IT": "ITdata.xlsx",
    "NL": "NLdata.xlsx",
    "PT": "PTdata.xlsx",
}

# Monthly macro indicators available in all 9 countries
MONTHLY_FEATURES = ["BCI", "CCI", "SHIX", "HICPOV", "UNETOT", "LTIRT", "REER42"]

# Temporal split (inclusive)
# train: 2000-Q2 → 2015-Q4
# test:  2016-Q1 → 2019-Q4   (used during dev phase)
# private_test: 2020-Q1 → 2025-Q3  (used for final scoring, includes COVID shock)
TRAIN_END   = "2015-12-31"
TEST_END    = "2019-12-31"

RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_country(path: str, country: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Normalise column names: GDP_DE -> GDP, BCI_DE -> BCI, etc.
    df = df.rename(columns={c: c.replace(f"_{country}", "") for c in df.columns})
    df["Time"] = pd.to_datetime(df["Time"])
    df["country"] = country
    return df.sort_values("Time").reset_index(drop=True)


def build_quarterly_samples(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    For each quarter with an observed GDP value, create one sample whose
    features are the 3 monthly observations within that quarter.
    """
    gdp_df = df[["Time", "GDP"]].dropna().copy()
    gdp_df["GDP_growth"] = gdp_df["GDP"].pct_change() * 100  # QoQ %
    gdp_df = gdp_df.dropna().reset_index(drop=True)

    samples = []
    for _, row in gdp_df.iterrows():
        qt = row["Time"]
        m1 = qt - pd.DateOffset(months=2)
        m2 = qt - pd.DateOffset(months=1)
        m3 = qt

        sample = {
            "country": country,
            "year": qt.year,
            "quarter_end": qt.strftime("%Y-%m-%d"),
            "GDP_growth": round(row["GDP_growth"], 6),
        }
        for feat in MONTHLY_FEATURES:
            if feat not in df.columns:
                for i in range(1, 4):
                    sample[f"{feat}_m{i}"] = np.nan
                continue
            for i, m in enumerate([m1, m2, m3], 1):
                vals = df.loc[df["Time"] == m, feat]
                sample[f"{feat}_m{i}"] = vals.values[0] if len(vals) > 0 else np.nan

        samples.append(sample)

    return pd.DataFrame(samples)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str, out_root: str):
    print(f"[setup_data] Reading raw data from: {data_dir}")

    all_samples = []
    for country, filename in COUNTRIES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            # also try with the numeric prefix used in the uploaded files
            candidates = [
                f for f in os.listdir(data_dir)
                if f.endswith(f"_{filename}")
            ]
            if candidates:
                path = os.path.join(data_dir, candidates[0])
            else:
                print(f"  [WARNING] {filename} not found in {data_dir}, skipping.")
                continue
        print(f"  Loading {country} from {os.path.basename(path)} ...", end=" ")
        df = load_country(path, country)
        samples = build_quarterly_samples(df, country)
        print(f"{len(samples)} quarterly samples")
        all_samples.append(samples)

    dataset = pd.concat(all_samples, ignore_index=True)
    dataset["quarter_end"] = pd.to_datetime(dataset["quarter_end"])
    dataset = dataset.sort_values(["country", "quarter_end"]).reset_index(drop=True)

    # Fill the 1-2 missing values with forward fill per country
    feat_cols = [c for c in dataset.columns if c not in ["country", "year", "quarter_end", "GDP_growth"]]
    dataset[feat_cols] = dataset.groupby("country")[feat_cols].transform(lambda x: x.ffill().bfill())

    # ── Temporal split ────────────────────────────────────────────────────────
    train_mask        = dataset["quarter_end"] <= TRAIN_END
    test_mask         = (dataset["quarter_end"] > TRAIN_END) & (dataset["quarter_end"] <= TEST_END)
    private_test_mask = dataset["quarter_end"] > TEST_END

    train        = dataset[train_mask].reset_index(drop=True)
    test         = dataset[test_mask].reset_index(drop=True)
    private_test = dataset[private_test_mask].reset_index(drop=True)

    print(f"\n[setup_data] Split summary:")
    print(f"  train:        {len(train):>4} rows  ({train['quarter_end'].min().date()} → {train['quarter_end'].max().date()})")
    print(f"  test:         {len(test):>4} rows  ({test['quarter_end'].min().date()} → {test['quarter_end'].max().date()})")
    print(f"  private_test: {len(private_test):>4} rows  ({private_test['quarter_end'].min().date()} → {private_test['quarter_end'].max().date()})")

    label_col  = "GDP_growth"
    id_cols    = ["country", "year", "quarter_end"]

    # ── Save ──────────────────────────────────────────────────────────────────
    def save(df, folder, name):
        path = os.path.join(out_root, folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  Saved {path}  ({df.shape})")

    train_features = train.drop(columns=[label_col])
    train_labels   = train[id_cols + [label_col]]

    test_features  = test.drop(columns=[label_col])
    test_labels    = test[id_cols + [label_col]]

    pt_features    = private_test.drop(columns=[label_col])
    pt_labels      = private_test[id_cols + [label_col]]

    print("\n[setup_data] Saving files ...")
    save(train_features, "dev_phase/input_data/train",        "train_features.csv")
    save(train_labels,   "dev_phase/input_data/train",        "train_labels.csv")
    save(test_features,  "dev_phase/input_data/test",         "test_features.csv")
    save(pt_features,    "dev_phase/input_data/private_test", "private_test_features.csv")
    save(test_labels,    "dev_phase/reference_data",          "test_labels.csv")
    save(pt_labels,      "dev_phase/reference_data",          "private_test_labels.csv")

    print("\n[setup_data] Done.")


def generate_synthetic_data(out_root: str, n_per_split: tuple = (200, 60, 80)):
    """
    Generate a small synthetic dataset for CI / testing purposes.
    Mimics the real feature schema but uses random data.
    """
    print("[setup_data] Generating synthetic data for CI ...")
    np.random.seed(42)
    countries = ["AT", "BE", "DE", "EL", "ES", "IE", "IT", "NL", "PT"]
    feat_cols = [f"{feat}_m{i}" for feat in MONTHLY_FEATURES for i in range(1, 4)]

    def _make_split(n, start_year):
        rows = []
        for i in range(n):
            c = countries[i % len(countries)]
            y = start_year + i // (len(countries) * 4)
            rows.append({
                "country": c,
                "year": y,
                "quarter_end": f"{y}-03-01",
                **{f: float(np.random.randn() + 100) for f in feat_cols},
            })
        return pd.DataFrame(rows)

    id_cols = ["country", "year", "quarter_end"]
    label_col = "GDP_growth"

    for split, n, sy in zip(
        ["train", "test", "private_test"], n_per_split, [2000, 2016, 2020]
    ):
        df = _make_split(n, sy)
        df[label_col] = np.random.randn(n) * 2

        features = df[id_cols + feat_cols]
        labels   = df[id_cols + [label_col]]

        if split == "train":
            features.to_csv(os.path.join(out_root, "dev_phase/input_data/train/train_features.csv"), index=False)
            labels.to_csv(  os.path.join(out_root, "dev_phase/input_data/train/train_labels.csv"),   index=False)
        elif split == "test":
            features.to_csv(os.path.join(out_root, "dev_phase/input_data/test/test_features.csv"), index=False)
            labels.to_csv(  os.path.join(out_root, "dev_phase/reference_data/test_labels.csv"),     index=False)
        else:
            features.to_csv(os.path.join(out_root, "dev_phase/input_data/private_test/private_test_features.csv"), index=False)
            labels.to_csv(  os.path.join(out_root, "dev_phase/reference_data/private_test_labels.csv"),            index=False)

    print("[setup_data] Synthetic data generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing the raw *data.xlsx files.",
    )
    parser.add_argument(
        "--out-root",
        default=".",
        help="Root of the competition repo (default: current directory).",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Generate synthetic data instead of reading real xlsx files (for CI).",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    for d in [
        "dev_phase/input_data/train",
        "dev_phase/input_data/test",
        "dev_phase/input_data/private_test",
        "dev_phase/reference_data",
    ]:
        os.makedirs(os.path.join(args.out_root, d), exist_ok=True)

    if args.ci:
        generate_synthetic_data(args.out_root)
    else:
        main(args.data_dir, args.out_root)
