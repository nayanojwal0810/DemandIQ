-- Population of dimensions and fact table from staging
-- Ensures idempotency and referential integrity

-- 1. Populate dim_brand
INSERT INTO dim_brand (brand_id)
SELECT DISTINCT brand_id
FROM staging_daily_demand
ORDER BY brand_id
ON CONFLICT (brand_id) DO NOTHING;

-- 2. Populate dim_sku
INSERT INTO dim_sku (sku_id, brand_id)
SELECT DISTINCT sku_id, brand_id
FROM staging_daily_demand
ORDER BY sku_id
ON CONFLICT (sku_id) DO UPDATE SET brand_id = EXCLUDED.brand_id;

-- 3. Populate dim_date (2014-01-02 through 2018-12-31 continuous calendar)
INSERT INTO dim_date (date, year, quarter, month, week_of_year, day_of_month, day_of_year, day_of_week, is_weekend)
SELECT
    d::DATE AS date,
    EXTRACT(YEAR FROM d)::INTEGER AS year,
    EXTRACT(QUARTER FROM d)::INTEGER AS quarter,
    EXTRACT(MONTH FROM d)::INTEGER AS month,
    EXTRACT(WEEK FROM d)::INTEGER AS week_of_year,
    EXTRACT(DAY FROM d)::INTEGER AS day_of_month,
    EXTRACT(DOY FROM d)::INTEGER AS day_of_year,
    (EXTRACT(ISODOW FROM d)::INTEGER - 1) AS day_of_week,
    CASE WHEN EXTRACT(ISODOW FROM d) IN (6, 7) THEN 1 ELSE 0 END AS is_weekend
FROM generate_series('2014-01-02'::DATE, '2018-12-31'::DATE, '1 day'::INTERVAL) AS d
ON CONFLICT (date) DO NOTHING;

-- 4. Populate fact_daily_demand
INSERT INTO fact_daily_demand (date, sku_id, brand_id, quantity, promotion)
SELECT
    date,
    sku_id,
    brand_id,
    quantity,
    promotion
FROM staging_daily_demand
ORDER BY date, brand_id, sku_id
ON CONFLICT (date, sku_id) DO UPDATE
SET
    brand_id = EXCLUDED.brand_id,
    quantity = EXCLUDED.quantity,
    promotion = EXCLUDED.promotion;
