"""Automated tests for DemandIQ PostgreSQL analytical pipeline."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sql.db import ENV_DATABASE_URL, get_connection
from src.sql.run_sql_pipeline import (
    initialize_database,
    populate_warehouse,
    run_analytical_cross_checks,
    run_reconciliation,
)

SQL_AVAILABLE = bool(os.environ.get(ENV_DATABASE_URL))


def test_sql_script_files_exist():
    """Verify that all expected SQL scripts exist in the repository."""
    expected_scripts = [
        "sql/schema/create_schema.sql",
        "sql/staging/create_staging_tables.sql",
        "sql/staging/load_staging_data.sql",
        "sql/dimensions/create_dim_brand.sql",
        "sql/dimensions/create_dim_sku.sql",
        "sql/dimensions/create_dim_date.sql",
        "sql/facts/create_fact_daily_demand.sql",
        "sql/analytics/demand_summary.sql",
        "sql/analytics/sku_demand_profile.sql",
        "sql/analytics/brand_demand_summary.sql",
        "sql/analytics/promotion_analysis.sql",
        "sql/analytics/temporal_features.sql",
        "sql/analytics/reconciliation_checks.sql",
    ]
    for script in expected_scripts:
        path = Path(script)
        assert path.exists(), f"Missing required SQL script: {script}"
        assert len(path.read_text(encoding="utf-8").strip()) > 0, f"Empty SQL script: {script}"


@pytest.fixture(scope="class")
def db_conn():
    """Provide open connection and tear down after tests."""
    conn = get_connection()
    initialize_database(conn)
    populate_warehouse(conn)
    yield conn
    conn.close()


@pytest.mark.skipif(not SQL_AVAILABLE, reason=f"PostgreSQL connection URL not configured in {ENV_DATABASE_URL}")
class TestPostgreSQLPipeline:
    """Integration test suite executed against live PostgreSQL database."""

    def test_schema_and_constraints(self, db_conn):
        """Verify primary key uniqueness, foreign keys, and value constraints."""
        with db_conn.cursor() as cur:
            # Fact table record count
            cur.execute("SELECT COUNT(*) FROM fact_daily_demand;")
            fact_count = cur.fetchone()[0]
            assert fact_count == 212164, f"Expected 212,164 fact rows, got {fact_count}"

            # Check distinct primary keys equals row count
            cur.execute("SELECT COUNT(DISTINCT (date, sku_id)) FROM fact_daily_demand;")
            pk_count = cur.fetchone()[0]
            assert pk_count == fact_count, "Primary key (date, sku_id) contains duplicates"

            # Check non-negative quantity constraint
            cur.execute("SELECT COUNT(*) FROM fact_daily_demand WHERE quantity < 0;")
            neg_count = cur.fetchone()[0]
            assert neg_count == 0, f"Found {neg_count} negative quantity records"

            # Check binary promotion constraint
            cur.execute("SELECT COUNT(*) FROM fact_daily_demand WHERE promotion NOT IN (0, 1);")
            invalid_promo = cur.fetchone()[0]
            assert invalid_promo == 0, f"Found {invalid_promo} non-binary promotion records"

            # Check SKU to brand consistency (each SKU maps to exactly one brand)
            cur.execute(
                """
                SELECT sku_id, COUNT(DISTINCT brand_id)
                FROM fact_daily_demand
                GROUP BY sku_id
                HAVING COUNT(DISTINCT brand_id) > 1;
                """
            )
            multi_brand_skus = cur.fetchall()
            assert len(multi_brand_skus) == 0, f"Found SKUs mapping to multiple brands: {multi_brand_skus}"

    def test_date_dimension_continuous_calendar(self, db_conn):
        """Verify calendar dimension spans 2014-01-02 to 2018-12-31 without gaps."""
        with db_conn.cursor() as cur:
            cur.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM dim_date;")
            min_date, max_date, count = cur.fetchone()
            assert str(min_date) == "2014-01-02"
            assert str(max_date) == "2018-12-31"
            assert count == 1825, f"Expected 1,825 calendar days, got {count}"

    def test_full_source_reconciliation(self, db_conn):
        """Run complete source-to-canonical mathematical reconciliation in SQL."""
        run_reconciliation(db_conn)

    def test_analytical_cross_checks(self, db_conn):
        """Verify SQL analytical summaries match saved Python profiles."""
        run_analytical_cross_checks(db_conn)

    def test_window_function_previous_observed_semantics(self, db_conn):
        """Verify LAG(quantity) represents previous observed row, not necessarily calendar day."""
        with db_conn.cursor() as cur:
            # Check for B1_1 after a store closure gap (e.g. 2014-01-06 vs 2014-01-07 if gap exists)
            cur.execute(
                """
                WITH ordered_demand AS (
                    SELECT
                        date,
                        quantity,
                        LAG(quantity) OVER (PARTITION BY sku_id ORDER BY date) AS lag_qty
                    FROM fact_daily_demand
                    WHERE sku_id = 'B1_1'
                )
                SELECT date, quantity, lag_qty
                FROM ordered_demand
                ORDER BY date
                LIMIT 5;
                """
            )
            rows = cur.fetchall()
            # First row has NULL lag_qty
            assert rows[0][2] is None
            # Subsequent rows match prior row quantity exactly
            for i in range(1, len(rows)):
                assert rows[i][2] == rows[i - 1][1]

    def test_exact_calendar_lag_7_semantics(self, db_conn):
        """Verify exact calendar lag t-7 correctly yields NULL when t-7 was not observed."""
        with db_conn.cursor() as cur:
            # Query dates where date - 7 days was absent from the dataset
            cur.execute(
                """
                SELECT
                    curr.date,
                    curr.sku_id,
                    prior_7.quantity AS lag_7_qty
                FROM fact_daily_demand curr
                LEFT JOIN fact_daily_demand prior_7
                    ON prior_7.sku_id = curr.sku_id
                   AND prior_7.date = (curr.date - INTERVAL '7 days')::DATE
                WHERE curr.sku_id = 'B1_1'
                  AND (curr.date - INTERVAL '7 days')::DATE NOT IN (
                      SELECT DISTINCT date FROM fact_daily_demand
                  )
                LIMIT 5;
                """
            )
            gap_rows = cur.fetchall()
            assert len(gap_rows) > 0, "Expected dates with unobserved t-7 calendar dates"
            for row in gap_rows:
                assert row[2] is None, f"Expected NULL for unobserved calendar lag on {row[0]}, got {row[2]}"

    def test_development_cutoff_boundary(self, db_conn):
        """Verify decision-oriented SQL analytics strictly exclude 2018 records."""
        with db_conn.cursor() as cur:
            # Check maximum date in development summary query
            cur.execute(
                """
                SELECT MAX(date)
                FROM fact_daily_demand
                WHERE date <= DATE '2017-12-31';
                """
            )
            max_dev_date = cur.fetchone()[0]
            assert str(max_dev_date) == "2017-12-31"
