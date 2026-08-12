"""
Data quality checks — Week 1, Day 4-7:
"Implement data quality checks and format dates, sales volumes, and pricing information."

Runs null/duplicate/range checks on each extracted table and writes a
markdown report to data/processed/data_quality_report.md.
"""

from pathlib import Path

import pandas as pd

from extract import run as extract_all

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "data_quality_report.md"


def check_table(name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## {name}", f"- Rows: {len(df):,}", f"- Columns: {len(df.columns)}"]

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        lines.append("- **Null values found:**")
        for col, count in nulls.items():
            lines.append(f"  - `{col}`: {count} nulls ({count / len(df):.1%})")
    else:
        lines.append("- Null values: none")

    dupes = df.duplicated().sum()
    lines.append(f"- Duplicate rows: {dupes}")

    numeric_df = df.select_dtypes(include="number")
    negatives = (numeric_df < 0).sum()
    negatives = negatives[negatives > 0]
    if len(negatives):
        lines.append("- **Negative values found (check validity):**")
        for col, count in negatives.items():
            lines.append(f"  - `{col}`: {count} negative values")

    date_cols = df.select_dtypes(include="datetime").columns
    for col in date_cols:
        lines.append(f"- `{col}` range: {df[col].min().date()} to {df[col].max().date()}")

    lines.append("")
    return lines


def run() -> None:
    tables = extract_all()
    report = ["# Data Quality Report — Retail Demand Forecasting", ""]

    for name, df in tables.items():
        report.extend(check_table(name, df))

    # Cross-table referential check: do Product IDs / Store IDs line up?
    demand_products = set(tables["demand_forecasting"]["product_id"])
    pricing_products = set(tables["pricing_optimization"]["product_id"])
    missing_in_pricing = demand_products - pricing_products
    report.append("## Cross-table checks")
    report.append(
        f"- Product IDs in demand_forecasting missing from pricing_optimization: "
        f"{len(missing_in_pricing)}"
    )

    REPORT_PATH.write_text("\n".join(report))
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    run()
