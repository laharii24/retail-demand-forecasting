"""
Categorical encoding for the forecasting feature table (Day 8).

Reads fct_demand_features (built by dbt — run `dbt run` before this script)
from the warehouse, one-hot encodes the text columns models can't use
directly, and writes the result back as fct_demand_features_encoded.

Run order: dbt run  ->  python encode_features.py  ->  (later) train models
on fct_demand_features_encoded.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "warehouse.db"

CATEGORICAL_COLUMNS = [
    "seasonality_factors",
    "external_factors",
    "demand_trend",
    "customer_segments",
]


def run() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("select * from fct_demand_features", conn)
        print(f"Loaded {len(df):,} rows from fct_demand_features")

        before_cols = set(df.columns)
        df_encoded = pd.get_dummies(
            df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS
        )
        new_cols = set(df_encoded.columns) - before_cols
        print(f"One-hot encoded {len(CATEGORICAL_COLUMNS)} columns "
              f"into {len(new_cols)} new binary columns")

        df_encoded.to_sql(
            "fct_demand_features_encoded", conn, if_exists="replace", index=False
        )
        print(f"Loaded {len(df_encoded):,} rows into 'fct_demand_features_encoded'")
    finally:
        conn.close()

    return df_encoded


if __name__ == "__main__":
    run()
