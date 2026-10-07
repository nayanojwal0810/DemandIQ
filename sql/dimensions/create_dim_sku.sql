-- Dimension: SKU
-- Minimal reproducible SKU dimension referencing brand

CREATE TABLE IF NOT EXISTS dim_sku (
    sku_id VARCHAR(20) PRIMARY KEY,
    brand_id VARCHAR(10) NOT NULL REFERENCES dim_brand(brand_id)
);
