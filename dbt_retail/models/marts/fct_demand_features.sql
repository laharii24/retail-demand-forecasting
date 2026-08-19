-- Feature table for demand prediction (LightGBM, Day 11+).
--
-- NOTE ON APPROACH: classic time-series lag/rolling-average features were
-- attempted first, but the underlying data doesn't support them --
-- 9,882 of 9,941 product/store combinations have exactly ONE row each
-- (checked directly against demand_forecasting_clean), so there's no
-- repeated history per product/store to lag over. This also means Prophet
-- (which needs a real time series per entity) isn't viable for this
-- dataset; the forecasting phase will use LightGBM as a cross-sectional
-- regression model instead -- predicting sales_quantity from the features
-- below, not from its own past values.
--
-- Grain: one row per product_id x store_id x date (same as
-- stg_demand_forecasting) -- no aggregation happens here, only new
-- columns derived from existing ones.

select
    product_id,
    store_id,
    date,
    sales_quantity,
    price,
    promotions,
    seasonality_factors,
    external_factors,
    demand_trend,
    customer_segments,
    price_opt,
    competitor_prices,
    elasticity_index,
    has_pricing_data,

    -- Cross-sectional derived features (no time history required)
    case
        when competitor_prices is not null and competitor_prices != 0
            then round((price - competitor_prices) / competitor_prices, 4)
        else null
    end as price_vs_competitor_pct,

    case
        when price_opt is not null and price != 0
            then round((price - price_opt) / price, 4)
        else null
    end as discount_pct,

    (case when promotions then 1 else 0 end) as promo_flag,

    round(sales_quantity * price, 2) as revenue,

    -- Interaction: promo effect isn't uniform across elasticity segments
    (case when promotions then 1 else 0 end) * coalesce(elasticity_index, 0)
        as promo_x_elasticity

from {{ ref('stg_demand_forecasting') }}
