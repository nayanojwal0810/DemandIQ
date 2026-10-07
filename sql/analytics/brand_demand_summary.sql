-- Analytical Query 2: Brand Demand Summary (Development Period <= 2017-12-31)
-- Aggregates demand by brand and computes portfolio share and average zero-demand rate

WITH dev_facts AS (
    SELECT brand_id, sku_id, quantity, date
    FROM fact_daily_demand
    WHERE date <= DATE '2017-12-31'
),
portfolio_total AS (
    SELECT SUM(quantity) AS portfolio_demand FROM dev_facts
),
observed_dates AS (
    SELECT COUNT(DISTINCT date) AS total_days FROM dev_facts
)
SELECT
    f.brand_id,
    COUNT(DISTINCT f.sku_id) AS sku_count,
    SUM(f.quantity) AS total_demand,
    ROUND((SUM(f.quantity) / (SELECT total_days FROM observed_dates))::NUMERIC, 4) AS mean_daily_demand,
    ROUND((SUM(f.quantity) / (SELECT portfolio_demand FROM portfolio_total))::NUMERIC, 6) AS portfolio_demand_share,
    ROUND((COUNT(CASE WHEN f.quantity = 0 THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC), 4) AS average_zero_rate
FROM dev_facts f
GROUP BY f.brand_id
ORDER BY f.brand_id;
