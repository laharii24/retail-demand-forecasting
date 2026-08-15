-- Weekly sales aggregation, grain: product_id x store_id x week.
-- SQLite has no native date_trunc, so week is derived via strftime.
-- '%Y-%W' = ISO-ish year + week-of-year (week starts Monday, weeks
-- 00-53). Good enough for trend analysis; not a strict ISO week number.

with base as (
    select
        product_id,
        store_id,
        date,
        strftime('%Y-%W', date) as sales_week,
        sales_quantity,
        price,
        sales_quantity * price as revenue,
        promotions
    from {{ ref('stg_demand_forecasting') }}
)

select
    product_id,
    store_id,
    sales_week,
    min(date) as week_start_approx,
    sum(sales_quantity) as total_units,
    round(avg(price), 2) as avg_price,
    round(sum(revenue), 2) as total_revenue,
    sum(case when promotions then 1 else 0 end) as promo_days
from base
group by product_id, store_id, sales_week
