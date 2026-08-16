-- Analysis-ready mart: one row per product_id x store_id, combining
-- demand history, inventory status, and pricing.
--
-- Grain note: stg_inventory_monitoring and stg_pricing_optimization each
-- have a small number of duplicate (product_id, store_id) combos (~50 out
-- of ~10,000 rows in each). Both are aggregated with avg() here to keep
-- this mart at a clean one-row-per-product/store grain rather than
-- silently fanning out demand rows on join.

with demand_summary as (
    select
        product_id,
        store_id,
        sum(sales_quantity) as total_units_sold,
        round(avg(price), 2) as avg_demand_price,
        round(sum(sales_quantity * price), 2) as total_revenue,
        round(avg(elasticity_index), 3) as avg_elasticity,
        sum(case when promotions then 1 else 0 end) as promo_days,
        sum(case when has_pricing_data then 1 else 0 end) as rows_with_pricing_data,
        count(*) as total_demand_rows
    from {{ ref('stg_demand_forecasting') }}
    group by product_id, store_id
),

inventory_summary as (
    select
        product_id,
        store_id,
        round(avg(stock_levels), 1) as avg_stock_levels,
        round(avg(supplier_lead_time_days), 1) as avg_supplier_lead_time_days,
        round(avg(stockout_frequency), 2) as avg_stockout_frequency,
        round(avg(reorder_point), 1) as avg_reorder_point,
        round(avg(warehouse_capacity), 1) as avg_warehouse_capacity
    from {{ ref('stg_inventory_monitoring') }}
    group by product_id, store_id
),

pricing_summary as (
    select
        product_id,
        store_id,
        round(avg(price), 2) as avg_listed_price,
        round(avg(competitor_prices), 2) as avg_competitor_price,
        round(avg(discounts), 2) as avg_discount,
        round(avg(elasticity_index), 3) as avg_pricing_elasticity
    from {{ ref('stg_pricing_optimization') }}
    group by product_id, store_id
)

select
    d.product_id,
    d.store_id,
    d.total_units_sold,
    d.avg_demand_price,
    d.total_revenue,
    d.avg_elasticity,
    d.promo_days,
    d.rows_with_pricing_data,
    d.total_demand_rows,
    i.avg_stock_levels,
    i.avg_supplier_lead_time_days,
    i.avg_stockout_frequency,
    i.avg_reorder_point,
    i.avg_warehouse_capacity,
    p.avg_listed_price,
    p.avg_competitor_price,
    p.avg_discount,
    p.avg_pricing_elasticity,
    (i.avg_stock_levels is not null) as has_inventory_data,
    (p.avg_listed_price is not null) as has_pricing_summary_data
from demand_summary d
left join inventory_summary i
    on d.product_id = i.product_id and d.store_id = i.store_id
left join pricing_summary p
    on d.product_id = p.product_id and d.store_id = p.store_id
