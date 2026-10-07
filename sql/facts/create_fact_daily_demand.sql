-- Fact Table: Daily Demand
-- Granularity: One row per observed trading date and SKU

CREATE TABLE IF NOT EXISTS fact_daily_demand (
    date DATE NOT NULL REFERENCES dim_date(date),
    sku_id VARCHAR(20) NOT NULL REFERENCES dim_sku(sku_id),
    brand_id VARCHAR(10) NOT NULL REFERENCES dim_brand(brand_id),
    quantity DOUBLE PRECISION NOT NULL CHECK (quantity >= 0),
    promotion INTEGER NOT NULL CHECK (promotion IN (0, 1)),
    PRIMARY KEY (date, sku_id)
);

-- Purpose-built indexes supporting analytical queries
CREATE INDEX IF NOT EXISTS idx_fact_demand_date ON fact_daily_demand(date);
CREATE INDEX IF NOT EXISTS idx_fact_demand_sku_date ON fact_daily_demand(sku_id, date);
CREATE INDEX IF NOT EXISTS idx_fact_demand_brand_date ON fact_daily_demand(brand_id, date);
