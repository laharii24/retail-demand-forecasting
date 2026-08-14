select
    product_id,
    store_id,
    stock_levels,
    supplier_lead_time_days,
    stockout_frequency,
    reorder_point,
    expiry_date,
    warehouse_capacity,
    order_fulfillment_time_days
from {{ source('warehouse', 'inventory_monitoring') }}
