-- Staging table for canonical daily demand
-- Mirrors the canonical CSV schema exactly

CREATE TABLE IF NOT EXISTS staging_daily_demand (
    date DATE NOT NULL,
    brand_id VARCHAR(10) NOT NULL,
    sku_id VARCHAR(20) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    promotion INTEGER NOT NULL
);
