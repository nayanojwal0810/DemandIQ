-- Dimension: Date
-- Calendar dimension covering the complete project span (2014-01-02 through 2018-12-31)
-- Contains all calendar dates regardless of retail trading closures

CREATE TABLE IF NOT EXISTS dim_date (
    date DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    week_of_year INTEGER NOT NULL,
    day_of_month INTEGER NOT NULL,
    day_of_year INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL, -- Monday=0 ... Sunday=6
    is_weekend INTEGER NOT NULL
);
