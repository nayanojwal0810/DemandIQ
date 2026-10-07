-- Analytical Queries 5, 6, 7 & 8: Window Functions and Temporal Analytics
-- Demonstrates previous-observed lag, exact calendar lookup, trailing calendar window, and weekly seasonality

-- Part A: Window Function vs Exact Calendar Lag vs Trailing Rolling Mean
WITH base_lags AS (
    SELECT
        curr.date,
        curr.brand_id,
        curr.sku_id,
        curr.quantity,
        -- Query 5: Immediately preceding observed demand date (row-based window)
        LAG(curr.quantity) OVER (
            PARTITION BY curr.sku_id
            ORDER BY curr.date
        ) AS previous_observed_quantity,
        -- Query 6: Exact calendar date lookup (date-aware self-join, returns NULL for unobserved dates)
        prior_7.quantity AS quantity_t_minus_7
    FROM fact_daily_demand curr
    LEFT JOIN fact_daily_demand prior_7
        ON prior_7.sku_id = curr.sku_id
       AND prior_7.date = (curr.date - INTERVAL '7 days')::DATE
    WHERE curr.date <= DATE '2017-12-31'
),
rolling_28 AS (
    -- Query 7: Trailing calendar window [t - 28 days, t - 1 day] excluding target date
    SELECT
        curr.date,
        curr.sku_id,
        COUNT(prior.quantity) AS observed_days_28,
        ROUND(AVG(prior.quantity)::NUMERIC, 4) AS rolling_mean_28
    FROM fact_daily_demand curr
    LEFT JOIN fact_daily_demand prior
        ON prior.sku_id = curr.sku_id
       AND prior.date >= (curr.date - INTERVAL '28 days')::DATE
       AND prior.date <= (curr.date - INTERVAL '1 day')::DATE
    WHERE curr.date <= DATE '2017-12-31'
    GROUP BY curr.date, curr.sku_id
)
SELECT
    b.date,
    b.brand_id,
    b.sku_id,
    b.quantity,
    b.previous_observed_quantity,
    b.quantity_t_minus_7,
    r.observed_days_28,
    r.rolling_mean_28
FROM base_lags b
JOIN rolling_28 r
    ON b.date = r.date
   AND b.sku_id = r.sku_id
ORDER BY b.date, b.brand_id, b.sku_id;
