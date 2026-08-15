-- Monthly sales aggregation, grain: product_id x store_id x month.

with base as (
    select
        product_id,
        store_id,
        date,
        strftime('%Y-%m', date) as sales_month,
        sales_quantity,
        price,
        sales_quantity * price as revenue,
        promotions,
        demand_trend
    from {{ ref('stg_demand_forecasting') }}
)

select
    product_id,
    store_id,
    sales_month,
    sum(sales_quantity) as total_units,
    round(avg(price), 2) as avg_price,
    round(sum(revenue), 2) as total_revenue,
    sum(case when promotions then 1 else 0 end) as promo_days,
    -- most common demand_trend label that month, as a simple mode
    (
        select demand_trend
        from base b2
        where b2.product_id = base.product_id
          and b2.store_id = base.store_id
          and b2.sales_month = base.sales_month
        group by demand_trend
        order by count(*) desc
        limit 1
    ) as dominant_demand_trend
from base
group by product_id, store_id, sales_month
