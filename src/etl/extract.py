"""
Extraction & Load script — Retail Demand Forecasting & Inventory Optimization
Week 1, Day 1-3: Load raw CSVs, parse/clean types, load into a warehouse.

NOTE ON WAREHOUSE CHOICE:
The brief specifies Google BigQuery or Snowflake. For local development this
script loads into a local SQLite database (data/processed/warehouse.db) so the
pipeline is runnable without cloud credentials. The schema and load logic are
written so swapping the `sqlite3` connection for a BigQuery/Snowflake client
is a drop-in change later (see load_to_warehouse()).
"""

import sqlite3
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROCESSED_DIR / "warehouse.db"


def extract_demand_forecasting() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "demand_forecasting.csv")
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["sales_quantity"] = pd.to_numeric(df["sales_quantity"], errors="coerce")
    df["promotions"] = df["promotions"].map({"Yes": True, "No": False})
    return df


def extract_inventory_monitoring() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "inventory_monitoring.csv")
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    for col in ["stock_levels", "supplier_lead_time_days", "stockout_frequency",
                "reorder_point", "warehouse_capacity", "order_fulfillment_time_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def extract_pricing_optimization() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "pricing_optimization.csv")
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
    numeric_cols = ["price", "competitor_prices", "discounts", "sales_volume",
                     "customer_reviews", "return_rate_%", "storage_cost", "elasticity_index"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_to_warehouse(tables: dict[str, pd.DataFrame]) -> None:
    """Load cleaned DataFrames into the warehouse (SQLite stand-in for BigQuery/Snowflake)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        for table_name, df in tables.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"Loaded {len(df):,} rows into '{table_name}'")
    finally:
        conn.close()


def run() -> dict[str, pd.DataFrame]:
    tables = {
        "demand_forecasting": extract_demand_forecasting(),
        "inventory_monitoring": extract_inventory_monitoring(),
        "pricing_optimization": extract_pricing_optimization(),
    }
    load_to_warehouse(tables)
    return tables


if __name__ == "__main__":
    run()
    print(f"\nWarehouse written to: {DB_PATH}")
