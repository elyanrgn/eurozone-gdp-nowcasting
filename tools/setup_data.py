"""
setup_data.py
=============
Builds the competition dataset from the raw Euro Area country Excel files.

New format (v2): participants receive the RAW monthly panel for all 10 countries.
Each row = one country × one month. GDP is only non-null on quarter-end months.
Participants must handle:
  - Mixed-frequency data (monthly indicators + quarterly GDP)
  - Variable selection (111 columns, many with high missingness)
  - Feature engineering from the raw time series

Target: predict GDP_growth (QoQ %) for each country × quarter observation.

Temporal split (by quarter-end date):
  train:        2000-Q2 → 2015-Q4
  test:         2016-Q1 → 2019-Q4
  private_test: 2020-Q1 → 2025-Q3  (includes COVID shock)

Output structure:
  dev_phase/input_data/train/train_features.csv       raw monthly panel (train period)
  dev_phase/input_data/train/train_labels.csv         GDP_growth per country×quarter
  dev_phase/input_data/test/test_features.csv         raw monthly panel (test period)
  dev_phase/input_data/private_test/...               raw monthly panel (private period)
  dev_phase/reference_data/test_labels.csv            GDP_growth per country×quarter
  dev_phase/reference_data/private_test_labels.csv    GDP_growth per country×quarter
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

COUNTRIES = {
    "AT": "ATdata.xlsx",
    "BE": "BEdata.xlsx",
    "DE": "DEdata.xlsx",
    "EL": "ELdata.xlsx",
    "ES": "ESdata.xlsx",
    "FR": "FRdata.xlsx",
    "IE": "IEdata.xlsx",
    "IT": "ITdata.xlsx",
    "NL": "NLdata.xlsx",
    "PT": "PTdata.xlsx",
}

# Quarter-end months that bound the splits
TRAIN_END = pd.Timestamp("2015-12-01")
TEST_END = pd.Timestamp("2019-12-01")
# private_test: everything after TEST_END

RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────


def find_file(data_dir: str, filename: str) -> str | None:
    """Find a file by exact name or with a numeric prefix (uploaded format)."""
    exact = os.path.join(data_dir, filename)
    if os.path.exists(exact):
        return exact
    candidates = [f for f in os.listdir(data_dir) if f.endswith(f"_{filename}")]
    if candidates:
        return os.path.join(data_dir, candidates[0])
    return None


def load_country(path: str, country: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Normalise column names: GDP_DE → GDP, TASS.LBD_DE → TASS.LBD
    rename = {c: c.replace(f"_{country}", "") for c in df.columns}
    df = df.rename(columns=rename)
    df["Time"] = pd.to_datetime(df["Time"])
    df["country"] = country
    return df.sort_values("Time").reset_index(drop=True)


def build_full_panel(data_dir: str) -> pd.DataFrame:
    """Load all countries and stack into a single monthly panel."""
    all_dfs = []
    for country, filename in COUNTRIES.items():
        path = find_file(data_dir, filename)
        if path is None:
            print(
                f"  [WARNING] {filename} not found in {data_dir}, skipping {country}."
            )
            continue
        print(f"  Loading {country} from {os.path.basename(path)} ...", end=" ")
        df = load_country(path, country)
        print(f"{len(df)} monthly rows, {df.shape[1]} columns")
        all_dfs.append(df)

    # Union of all column names (countries may differ slightly)
    all_cols = set()
    for d in all_dfs:
        all_cols |= set(d.columns)

    panel = pd.concat(
        [d.reindex(columns=sorted(all_cols)) for d in all_dfs], ignore_index=True
    )
    panel = panel.sort_values(["country", "Time"]).reset_index(drop=True)

    # Ensure Time and country come first
    front = ["Time", "country"]
    rest = [c for c in sorted(panel.columns) if c not in front]
    return panel[front + rest]


def build_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GDP_growth (QoQ %) from the raw panel.
    Returns a DataFrame with one row per country × quarter-end month.
    """
    records = []
    for country, grp in panel.groupby("country"):
        grp = grp.sort_values("Time").copy()
        gdp_rows = grp[grp["GDP"].notna()][["Time", "GDP"]].copy()
        gdp_rows["GDP_growth"] = gdp_rows["GDP"].pct_change() * 100
        gdp_rows = gdp_rows.dropna(subset=["GDP_growth"])
        gdp_rows["country"] = country
        records.append(gdp_rows[["country", "Time", "GDP_growth"]])
    return pd.concat(records, ignore_index=True)


def temporal_split(panel: pd.DataFrame, labels: pd.DataFrame):
    """
    Split both panel and labels by the quarter-end cutoff dates.
    The panel for a given split contains all monthly rows up to that split's end.
    """
    # Quarter-end months for each split
    train_qends = labels[labels["Time"] <= TRAIN_END]["Time"].unique()
    test_qends = labels[(labels["Time"] > TRAIN_END) & (labels["Time"] <= TEST_END)][
        "Time"
    ].unique()
    pt_qends = labels[labels["Time"] > TEST_END]["Time"].unique()

    # For the panel: include all months that belong to quarters in the split
    # A quarter's months are: quarter_end - 2 months, quarter_end - 1 month, quarter_end
    def months_for_qends(qends):
        months = set()
        for qe in qends:
            for offset in [0, 1, 2]:
                months.add(qe - pd.DateOffset(months=offset))
        return months

    train_months = months_for_qends(train_qends)
    test_months = months_for_qends(test_qends)
    pt_months = months_for_qends(pt_qends)

    panel["_time_ts"] = panel["Time"]

    panel_train = panel[panel["_time_ts"].isin(train_months)].drop(columns=["_time_ts"])
    panel_test = panel[panel["_time_ts"].isin(test_months)].drop(columns=["_time_ts"])
    panel_pt = panel[panel["_time_ts"].isin(pt_months)].drop(columns=["_time_ts"])
    panel.drop(columns=["_time_ts"], inplace=True)

    lbl_train = labels[labels["Time"] <= TRAIN_END]
    lbl_test = labels[(labels["Time"] > TRAIN_END) & (labels["Time"] <= TEST_END)]
    lbl_pt = labels[labels["Time"] > TEST_END]

    return (
        panel_train,
        panel_test,
        panel_pt,
        lbl_train.reset_index(drop=True),
        lbl_test.reset_index(drop=True),
        lbl_pt.reset_index(drop=True),
    )


# ── CI synthetic data ─────────────────────────────────────────────────────────


def generate_synthetic_data(out_root: str):
    """Generate small synthetic data for CI/testing (no xlsx files needed)."""
    print("[setup_data] Generating synthetic data for CI ...")
    np.random.seed(42)
    countries = ["AT", "BE", "DE", "EL", "ES", "FR", "IE", "IT", "NL", "PT"]
    times = pd.date_range("2000-01-01", periods=90, freq="MS")  # 7.5 years
    all_rows = []
    for ct in countries:
        for t in times:
            row = {"Time": t.strftime("%Y-%m-%d"), "country": ct}
            for col in [
                "BCI",
                "CCI",
                "SHIX",
                "HICPOV",
                "UNETOT",
                "LTIRT",
                "REER42",
                "EXPGS",
                "IMPGS",
                "HFCE",
                "GFCF",
            ]:
                row[col] = round(np.random.randn() + 100, 4)
            if t.month in (3, 6, 9, 12):
                row["GDP"] = round(500000 + np.random.randn() * 5000, 2)
            else:
                row["GDP"] = np.nan
            all_rows.append(row)
    panel = (
        pd.DataFrame(all_rows).sort_values(["country", "Time"]).reset_index(drop=True)
    )

    # Labels
    label_rows = []
    for ct in countries:
        sub = panel[panel["country"] == ct].copy()
        q = sub[sub["GDP"].notna()].copy()
        q["GDP_val"] = pd.to_numeric(q["GDP"], errors="coerce")
        q["GDP_growth"] = q["GDP_val"].pct_change() * 100
        q = q.dropna(subset=["GDP_growth"])
        for _, r in q.iterrows():
            label_rows.append(
                {
                    "country": ct,
                    "Time": r["Time"],
                    "GDP_growth": round(r["GDP_growth"], 4),
                }
            )
    labels = pd.DataFrame(label_rows)

    # Split: first 60% train, next 20% test, last 20% private_test
    times_sorted = sorted(labels["Time"].unique())
    n = len(times_sorted)
    cut1 = times_sorted[int(n * 0.6)]
    cut2 = times_sorted[int(n * 0.8)]

    splits = [
        ("train", panel[panel["Time"] <= cut1], labels[labels["Time"] <= cut1]),
        (
            "test",
            panel[(panel["Time"] > cut1) & (panel["Time"] <= cut2)],
            labels[(labels["Time"] > cut1) & (labels["Time"] <= cut2)],
        ),
        ("private_test", panel[panel["Time"] > cut2], labels[labels["Time"] > cut2]),
    ]
    for split, pnl, lbl in splits:
        _save(pnl, out_root, f"dev_phase/input_data/{split}", f"{split}_features.csv")
        if split == "train":
            _save(lbl, out_root, f"dev_phase/input_data/{split}", "train_labels.csv")
        else:
            _save(lbl, out_root, "dev_phase/reference_data", f"{split}_labels.csv")
            # Label skeleton alongside features
            _save(
                lbl[["country", "Time"]],
                out_root,
                f"dev_phase/input_data/{split}",
                f"{split}_labels_skeleton.csv",
            )
    print("[setup_data] Synthetic data generated.")


# ── Save helpers ──────────────────────────────────────────────────────────────


def _save(df: pd.DataFrame, root: str, folder: str, name: str):
    path = os.path.join(root, folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved {path}  ({df.shape})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(data_dir: str, out_root: str):
    print(f"[setup_data] Reading raw data from: {data_dir}")

    panel = build_full_panel(data_dir)
    labels = build_labels(panel)

    print(
        f"\n[setup_data] Full panel: {panel.shape}  ({panel['country'].nunique()} countries, "
        f"{panel['Time'].nunique()} monthly timestamps)"
    )
    print(f"[setup_data] Labels: {len(labels)} country×quarter observations")

    (p_train, p_test, p_pt, l_train, l_test, l_pt) = temporal_split(panel, labels)

    print(f"\n[setup_data] Splits:")
    print(
        f"  train:        {len(p_train):5d} monthly rows | {len(l_train):4d} quarterly labels "
        f"({l_train['Time'].min().date()} → {l_train['Time'].max().date()})"
    )
    print(
        f"  test:         {len(p_test):5d} monthly rows | {len(l_test):4d} quarterly labels "
        f"({l_test['Time'].min().date()} → {l_test['Time'].max().date()})"
    )
    print(
        f"  private_test: {len(p_pt):5d} monthly rows | {len(l_pt):4d} quarterly labels "
        f"({l_pt['Time'].min().date()} → {l_pt['Time'].max().date()})"
    )

    # Serialise Time as string to avoid timezone issues
    for df in [p_train, p_test, p_pt]:
        df["Time"] = df["Time"].dt.strftime("%Y-%m-%d")
    for df in [l_train, l_test, l_pt]:
        df["Time"] = df["Time"].dt.strftime("%Y-%m-%d")

    print("\n[setup_data] Saving files ...")
    _save(p_train, out_root, "dev_phase/input_data/train", "train_features.csv")
    _save(l_train, out_root, "dev_phase/input_data/train", "train_labels.csv")
    _save(p_test, out_root, "dev_phase/input_data/test", "test_features.csv")
    _save(
        p_pt, out_root, "dev_phase/input_data/private_test", "private_test_features.csv"
    )
    _save(l_test, out_root, "dev_phase/reference_data", "test_labels.csv")
    _save(l_pt, out_root, "dev_phase/reference_data", "private_test_labels.csv")
    # Label skeletons: country + Time only, placed alongside features
    # Used by ingestion to know which (country, quarter) pairs to predict
    _save(
        l_test[["country", "Time"]],
        out_root,
        "dev_phase/input_data/test",
        "test_labels_skeleton.csv",
    )
    _save(
        l_pt[["country", "Time"]],
        out_root,
        "dev_phase/input_data/private_test",
        "private_test_labels_skeleton.csv",
    )
    print("\n[setup_data] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=".", help="Directory with raw *data.xlsx files."
    )
    parser.add_argument("--out-root", default=".", help="Root of the competition repo.")
    parser.add_argument(
        "--ci", action="store_true", help="Generate synthetic data for CI."
    )
    args = parser.parse_args()

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
