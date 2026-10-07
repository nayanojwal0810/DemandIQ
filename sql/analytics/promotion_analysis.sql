-- Analytical Query 4: Promotion Demand Analysis (Development Period <= 2017-12-31)
-- Compares observed demand statistics between promotion and non-promotion trading days

SELECT
    brand_id,
    sku_id,
    COUNT(*) AS total_observations,
    COUNT(CASE WHEN promotion = 1 THEN 1 END) AS promotion_observations,
    ROUND((COUNT(CASE WHEN promotion = 1 THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC), 4) AS promotion_rate,
    ROUND(AVG(CASE WHEN promotion = 1 THEN quantity END)::NUMERIC, 4) AS mean_promo_demand,
    ROUND(AVG(CASE WHEN promotion = 0 THEN quantity END)::NUMERIC, 4) AS mean_non_promo_demand
FROM fact_daily_demand
WHERE date <= DATE '2017-12-31'
GROUP BY brand_id, sku_id
ORDER BY brand_id, sku_id;
