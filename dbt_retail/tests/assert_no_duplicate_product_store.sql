select
    product_id,
    store_id,
    count(*) as row_count
from {{ ref('mart_product_store_overview') }}
group by product_id, store_id
having count(*) > 1