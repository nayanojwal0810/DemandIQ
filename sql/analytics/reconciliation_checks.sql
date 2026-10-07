-- Analytical Query 9: Source-to-Relational Reconciliation Checks
-- Formulates SQL verification aggregates to match against canonical dataset

-- 1. Grand total quantity and promotion count
SELECT
    'portfolio_totals' AS check_type,
    COUNT(*) AS total_rows,
    SUM(quantity) AS total_quantity,
    SUM(promotion) AS total_promotions
FROM fact_daily_demand;

-- 2. SKU-level quantity and promotion aggregates
SELECT
    brand_id,
    sku_id,
    COUNT(*) AS row_count,
    SUM(quantity) AS total_quantity,
    SUM(promotion) AS total_promotions
FROM fact_daily_demand
GROUP BY brand_id, sku_id
ORDER BY brand_id, sku_id;

-- 3. Brand-date aggregates
SELECT
    date,
    brand_id,
    COUNT(*) AS sku_count,
    SUM(quantity) AS brand_daily_quantity,
    SUM(promotion) AS brand_daily_promotions
FROM fact_daily_demand
GROUP BY date, brand_id
ORDER BY date, brand_id;
