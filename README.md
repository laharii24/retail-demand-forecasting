#  Retail Demand Forecasting & Inventory Optimization

An automated analytics platform for retail and supply chain teams. Uses
time-series forecasting on historical sales, promotions, and seasonality to
predict product demand at store/department level and generate optimal
inventory restocking schedules.

## Data Source

Working dataset (in place of the M5/Walmart dataset originally scoped) —
three complementary tables:

- `demand_forecasting.csv` — product/store sales, price, promotions,
  seasonality & external factors, demand trend, customer segment
- `inventory_monitoring.csv` — stock levels, supplier lead time, stockout
  frequency, reorder point, warehouse capacity
- `pricing_optimization.csv` — price, competitor price, discounts, sales
  volume, returns, elasticity index

10,000 rows per table.

## Tech Stack

- **Languages:** Python, SQL
- **Data Warehouse:** SQLite locally for development (`data/processed/warehouse.db`);
  designed to swap in for Google BigQuery/Snowflake for the cloud deployment
  target — see `load_to_warehouse()` in `src/etl/extract.py`
- **Data Transformation:** pandas now, migrating to dbt in Week 2
- **Forecasting Models (planned):** Facebook Prophet, LightGBM
- **Visualization (planned):** Streamlit

## Progress

### Week 1, Day 1-3 — Data Architecture & ETL ✅
- [x] Extraction scripts for all 3 raw tables (`src/etl/extract.py`)
- [x] Type casting, date parsing, boolean normalization
- [x] Load into local warehouse (SQLite stand-in)

### Week 1, Day 4-7 — Data Quality
- [x] Null / duplicate / negative-value checks (`src/etl/data_quality.py`)
- [x] Date range validation
- [x] Cross-table referential check (Product ID coverage between tables)
- [x] Resolve high null rate in `seasonality_factors` / `external_factors` — imputed as "Unknown" in staging    layer (`stg_demand_forecasting.sql`)
- [x] Document data lineage — `dbt docs generate` / lineage graph


### Week 1, Day 7 — dbt Testing & Documentation ✅
- [x] Added not_null tests on `product_id` / `store_id` in mart model
- [x] Added custom test for duplicate product/store rows in joined mart
- [x] Generated dbt docs + lineage graph
- [x] All 6 models rebuild clean (0 errors)
## Data Quality Findings (Day 1)

See `data/processed/data_quality_report.md` for the full report. Headline
issues to address next:

- `seasonality_factors` is 33% null, `external_factors` is 24% null in
  `demand_forecasting.csv` — need a strategy (impute "Unknown" category vs.
  drop) before modeling.
- 1,941 Product IDs appear in `demand_forecasting` but not in
  `pricing_optimization` — need to decide whether to left-join with nulls
  or exclude these products from pricing-aware forecasts.

## Setup

```bash
pip install -r requirements.txt
cd src/etl
python extract.py        # loads raw CSVs into local warehouse
python data_quality.py    # runs checks, writes report
```

## Repo Structure

```
project3/
├── data/
│   ├── raw/            # original CSVs (gitignored if large — see .gitignore)
│   └── processed/      # warehouse.db, data_quality_report.md
├── src/
│   └── etl/
│       ├── extract.py
│       └── data_quality.py
├── requirements.txt
└── README.md
```
