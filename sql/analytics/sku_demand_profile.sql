-- Analytical Query 3: SKU Demand Profile (Development Period <= 2017-12-31)
-- Per-SKU demand velocity, volatility, and intermittency metrics

SELECT
    brand_id,
    sku_id,
    SUM(quantity) AS total_demand,
    ROUND(AVG(quantity)::NUMERIC, 4) AS mean_daily_demand,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quantity) AS median_daily_demand,
    ROUND(STDDEV_SAMP(quantity)::NUMERIC, 4) AS std_daily_demand,
    ROUND(MIN(quantity)::NUMERIC, 4) AS min_daily_demand,
    ROUND(MAX(quantity)::NUMERIC, 4) AS max_daily_demand,
    ROUND((COUNT(CASE WHEN quantity = 0 THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC), 4) AS zero_demand_rate,
    ROUND((COUNT(CASE WHEN quantity > 0 THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC), 4) AS nonzero_demand_rate
FROM fact_daily_demand
WHERE date <= DATE '2017-12-31'
GROUP BY brand_id, sku_id
ORDER BY brand_id, sku_id;
