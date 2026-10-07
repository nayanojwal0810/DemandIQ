-- Analytical Query 1: Portfolio Demand Summary (Development Period <= 2017-12-31)
-- Aggregates daily demand across all 118 SKUs and computes portfolio statistics

WITH daily_portfolio AS (
    SELECT
        date,
        SUM(quantity) AS total_demand
    FROM fact_daily_demand
    WHERE date <= DATE '2017-12-31'
    GROUP BY date
)
SELECT
    COUNT(*) AS day_count,
    SUM(total_demand) AS total_demand,
    ROUND(AVG(total_demand)::NUMERIC, 4) AS mean_daily_demand,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_demand) AS median_daily_demand,
    ROUND(STDDEV_SAMP(total_demand)::NUMERIC, 4) AS std_daily_demand,
    MIN(total_demand) AS min_daily_demand,
    MAX(total_demand) AS max_daily_demand
FROM daily_portfolio;
