-- Thin staging layer: rename/select columns, no heavy logic.
-- Heavy cleaning already happened in src/etl/clean.py (Day 2) before load.

select
    product_id,
    date,
    store_id,
    sales_quantity,
    price,
    promotions,
        coalesce(seasonality_factors, 'Unknown') as seasonality_factors,
    coalesce(external_factors, 'Unknown') as external_factors,
    demand_trend,
    customer_segments,
    price_opt,
    competitor_prices,
    elasticity_index,
    has_pricing_data
from {{ source('warehouse', 'demand_forecasting_clean') }}
