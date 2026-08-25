"""
Day 14: Add lag and rolling-window features based on aggregate daily demand.
Each product-store pair appears too rarely (1-2 times) for its own history,
so we compute trend features at the daily aggregate level instead, and
merge them back onto every row for that date.
"""
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def add_daily_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    # Aggregate total demand per day across all products/stores
    daily = df.groupby("date", as_index=False)["sales_quantity"].sum()
    daily = daily.rename(columns={"sales_quantity": "daily_total_sales"})
    daily = daily.sort_values("date")

    daily["daily_sales_lag_7"] = daily["daily_total_sales"].shift(7)
    daily["daily_sales_lag_14"] = daily["daily_total_sales"].shift(14)
    daily["daily_rolling_mean_7"] = daily["daily_total_sales"].shift(1).rolling(7).mean()
    daily["daily_rolling_std_7"] = daily["daily_total_sales"].shift(1).rolling(7).std()

    # Merge the daily trend features back onto every row for that date
    df = df.merge(
        daily.drop(columns=["daily_total_sales"]),
        on="date",
        how="left",
    )
    return df


def main():
    print("Loading full feature table ...")
    df = pd.read_csv(PROCESSED_DIR / "demand_forecasting_clean.csv", parse_dates=["date"])

    print(f"Rows before: {len(df)}")
    df = add_daily_lag_rolling_features(df)

    n_missing = df["daily_sales_lag_7"].isna().sum()
    print(f"Rows with missing daily_sales_lag_7 (first 7 days only, expected): {n_missing}")

    out_path = PROCESSED_DIR / "feature_table_with_lags.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()