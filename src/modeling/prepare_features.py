"""
Day 9: Prepare train/test split for LightGBM.

Loads fct_demand_features (built on Day 8) from the local warehouse,
does a TIME-BASED train/test split (not random shuffle -- this is
time-series data and shuffling would leak future info into training),
sanity-checks the split, and saves train.csv / test.csv for tomorrow's
LightGBM training script.

Day 14 update: merges in daily-aggregate lag/rolling demand features
(from feature_table_with_lags.csv) before splitting, so train/test
carry the new trend features too.
"""

import sqlite3
from pathlib import Path

import pandas as pd

# ---- Paths (match extract.py's layout) ----
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DB_PATH = PROCESSED_DIR / "warehouse.db"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# Fraction of the date range to use for training. The remaining, most
# recent portion becomes the test set -- this mimics how the model will
# actually be used (train on the past, predict the future).
TRAIN_FRACTION = 0.8

# Name of the Day 8 feature table
FEATURE_TABLE = "fct_demand_features"

# Name of the date column in that table -- change this if yours differs
DATE_COL = "date"

# Day 14: daily-aggregate lag/rolling features to merge in
LAG_FEATURES_PATH = PROCESSED_DIR / "feature_table_with_lags.csv"
LAG_COLS = ["daily_sales_lag_7", "daily_sales_lag_14", "daily_rolling_mean_7", "daily_rolling_std_7"]


def load_features(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Could not find {db_path}. Run `dbt run` first so "
            f"{FEATURE_TABLE} exists in the warehouse."
        )
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE}", conn)
    if DATE_COL not in df.columns:
        raise KeyError(
            f"Expected a '{DATE_COL}' column in {FEATURE_TABLE} but got: "
            f"{list(df.columns)}. Update DATE_COL in this script to match."
        )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    if not LAG_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {LAG_FEATURES_PATH}. Run "
            f"src/etl/add_lag_features.py first (Day 14)."
        )
    lag_df = pd.read_csv(LAG_FEATURES_PATH, parse_dates=[DATE_COL])
    lag_df = lag_df[[DATE_COL] + LAG_COLS].drop_duplicates(subset=DATE_COL)
    return df.merge(lag_df, on=DATE_COL, how="left")


def time_based_split(df: pd.DataFrame, train_fraction: float):
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    cutoff_idx = int(len(df) * train_fraction)
    cutoff_date = df.loc[cutoff_idx, DATE_COL]

    train = df[df[DATE_COL] < cutoff_date].copy()
    test = df[df[DATE_COL] >= cutoff_date].copy()
    return train, test, cutoff_date


def sanity_check(train: pd.DataFrame, test: pd.DataFrame):
    print("\n--- Split summary ---")
    print(f"Train: {train.shape[0]:,} rows, {train.shape[1]} columns")
    print(f"  date range: {train[DATE_COL].min()} -> {train[DATE_COL].max()}")
    print(f"Test:  {test.shape[0]:,} rows, {test.shape[1]} columns")
    print(f"  date range: {test[DATE_COL].min()} -> {test[DATE_COL].max()}")

    print("\n--- Null check (top offenders) ---")
    null_counts = train.isnull().sum().sort_values(ascending=False)
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("No nulls found in train set.")
    else:
        print(null_counts.head(10))

    overlap = set(train[DATE_COL]).intersection(set(test[DATE_COL]))
    if overlap:
        print(f"\nWARNING: {len(overlap)} overlapping dates between train/test!")
    else:
        print("\nNo date overlap between train and test. Split is clean.")


def main():
    print(f"Loading {FEATURE_TABLE} from {DB_PATH} ...")
    df = load_features(DB_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")

    print("Merging Day 14 lag/rolling features ...")
    df = add_lag_features(df)
    print(f"After merge: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    train, test, cutoff_date = time_based_split(df, TRAIN_FRACTION)
    print(f"\nSplitting at cutoff date: {cutoff_date}")

    sanity_check(train, test)

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"\nSaved:\n  {TRAIN_PATH}\n  {TEST_PATH}")


if __name__ == "__main__":
    main()