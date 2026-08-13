"""
Data cleaning — resolves issues surfaced by data_quality.py on Day 1:

1. seasonality_factors / external_factors nulls (33% / 24% in
   demand_forecasting.csv) -> imputed as the literal string "Unknown".
   Decision: impute rather than drop, since dropping ~33% of rows would
   materially shrink the forecasting dataset. "Unknown" is kept as its
   own category rather than e.g. the mode, so a forecasting model can
   still learn a distinct effect for "we don't know the factor" rather
   than silently being told it's the most common factor.

2. Product ID mismatch: 1,941 products appear in demand_forecasting but
   have no matching row in pricing_optimization. Decision: LEFT JOIN and
   keep them, with pricing columns null for those rows, rather than
   dropping the products entirely. A `has_pricing_data` boolean flags
   which rows came from a real match, so downstream analysis can choose
   to exclude them explicitly if a pricing-aware view is needed.

Also adds a reusable join-coverage check to data_quality.py's report so
future extracts catch matching issues automatically.
"""

from pathlib import Path

import pandas as pd

from extract import (
    extract_demand_forecasting,
    extract_pricing_optimization,
    extract_inventory_monitoring,
    load_to_warehouse,
)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def clean_demand_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["seasonality_factors", "external_factors"]:
        n_null = df[col].isnull().sum()
        df[col] = df[col].fillna("Unknown")
        print(f"  Imputed {n_null:,} nulls in '{col}' as 'Unknown'")
    return df


def join_demand_with_pricing(demand: pd.DataFrame, pricing: pd.DataFrame) -> pd.DataFrame:
    # One price row per (product_id, store_id) pair; average if duplicates exist
    pricing_slim = (
        pricing.groupby(["product_id", "store_id"], as_index=False)
        .agg(price_opt=("price", "mean"),
             competitor_prices=("competitor_prices", "mean"),
             elasticity_index=("elasticity_index", "mean"))
    )

    merged = demand.merge(pricing_slim, on=["product_id", "store_id"], how="left")
    merged["has_pricing_data"] = merged["price_opt"].notna()

    n_missing = (~merged["has_pricing_data"]).sum()
    n_products_missing = demand.loc[
        ~demand["product_id"].isin(pricing_slim["product_id"]), "product_id"
    ].nunique()
    print(f"  {n_missing:,} rows have no matching pricing data "
          f"({n_products_missing:,} distinct Product IDs)")
    print("  Kept via LEFT JOIN; flagged with has_pricing_data = False")

    return merged


def run() -> pd.DataFrame:
    print("Cleaning demand_forecasting...")
    demand = clean_demand_forecasting(extract_demand_forecasting())

    print("\nJoining demand_forecasting with pricing_optimization...")
    pricing = extract_pricing_optimization()
    demand_clean = join_demand_with_pricing(demand, pricing)

    inventory = extract_inventory_monitoring()

    load_to_warehouse({
        "demand_forecasting_clean": demand_clean,
        "inventory_monitoring": inventory,
        "pricing_optimization": pricing,
    })

    out_path = PROCESSED_DIR / "demand_forecasting_clean.csv"
    demand_clean.to_csv(out_path, index=False)
    print(f"\nCleaned table written to: {out_path}")

    return demand_clean


if __name__ == "__main__":
    run()
