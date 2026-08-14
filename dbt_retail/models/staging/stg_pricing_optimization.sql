select
    product_id,
    store_id,
    price,
    competitor_prices,
    discounts,
    sales_volume,
    customer_reviews,
    "return_rate_%" as return_rate_pct,
    storage_cost,
    elasticity_index
from {{ source('warehouse', 'pricing_optimization') }}
