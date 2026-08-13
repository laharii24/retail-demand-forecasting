# Data Quality Report — Retail Demand Forecasting

## demand_forecasting
- Rows: 10,000
- Columns: 10
- **Null values found:**
  - `seasonality_factors`: 3315 nulls (33.1%)
  - `external_factors`: 2426 nulls (24.3%)
- Duplicate rows: 0
- `date` range: 2024-01-01 to 2024-12-30

## inventory_monitoring
- Rows: 10,000
- Columns: 9
- Null values: none
- Duplicate rows: 0
- `expiry_date` range: 2024-01-01 to 2024-12-30

## pricing_optimization
- Rows: 10,000
- Columns: 10
- Null values: none
- Duplicate rows: 0

## Cross-table checks
- Product IDs in demand_forecasting missing from pricing_optimization: 1941 (32.0% of distinct products)
  - Resolution: `clean.py` LEFT JOINs demand_forecasting with pricing_optimization and flags matched rows with `has_pricing_data`, rather than dropping unmatched products.

## Resolved issues (see clean.py)
- `seasonality_factors` / `external_factors` nulls imputed as `"Unknown"` category
- Product ID mismatch resolved via LEFT JOIN + `has_pricing_data` flag (see cross-table check above)