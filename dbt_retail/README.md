# dbt project — retail_forecasting

Transforms the raw/cleaned warehouse tables (loaded by `src/etl/extract.py`
and `src/etl/clean.py`) into staging views and, from Day 4 onward, analysis-
ready marts.

## Setup (run locally — requires internet access)

```bash
pip install dbt-core dbt-sqlite

mkdir -p ~/.dbt
cp profiles.yml.example ~/.dbt/profiles.yml
# edit the `schemas_and_paths` / `schema_directory` paths in profiles.yml
# to point at your local data/processed/warehouse.db

cd dbt_retail
dbt debug     # confirms the connection to warehouse.db works
dbt run       # builds all staging models
```

## What's here so far (Day 3)

- `dbt_project.yml` — project config
- `models/staging/sources.yml` — declares the 3 warehouse tables as dbt sources
- `models/staging/stg_demand_forecasting.sql`
- `models/staging/stg_inventory_monitoring.sql`
- `models/staging/stg_pricing_optimization.sql`

These staging models are thin — just column selection/renaming. The real
cleaning (null imputation, pricing join) already happened in `clean.py`
before the data was loaded into the warehouse, so dbt isn't re-doing that
work.

## Coming next (Day 4+)

- `models/marts/` — a joined, analysis-ready table combining demand,
  inventory, and pricing at the product/store grain
- Weekly/monthly aggregation models
- `dbt docs generate` for lineage documentation
