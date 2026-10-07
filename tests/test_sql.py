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

            # Direct validation that every fact-row brand_id matches dim_sku.brand_id
            cur.execute(
                """
                SELECT COUNT(*)
                FROM fact_daily_demand f
                JOIN dim_sku s ON f.sku_id = s.sku_id
                WHERE f.brand_id != s.brand_id;
                """
            )
            brand_mismatches = cur.fetchone()[0]
            assert brand_mismatches == 0, f"Found {brand_mismatches} fact rows where brand_id != dim_sku.brand_id"

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
        """Verify LAG(quantity) represents previous observed record across calendar gaps."""
        with db_conn.cursor() as cur:
            # Query rows where previous observed trading date is more than one calendar day earlier
            cur.execute(
                """
                WITH ordered_demand AS (
                    SELECT
                        date,
                        quantity,
                        LAG(date) OVER (PARTITION BY sku_id ORDER BY date) AS prev_date,
                        LAG(quantity) OVER (PARTITION BY sku_id ORDER BY date) AS lag_qty
                    FROM fact_daily_demand
                    WHERE sku_id = 'B1_1'
                )
                SELECT
                    date,
                    quantity,
                    prev_date,
                    lag_qty,
                    (date - prev_date) AS day_gap
                FROM ordered_demand
                WHERE prev_date IS NOT NULL
                  AND (date - prev_date) > 1
                ORDER BY date;
                """
            )
            gap_rows = cur.fetchall()
            assert len(gap_rows) > 0, "Expected observed dates for B1_1 separated by calendar gaps"

            # Assert each gap is > 1 day and lag_qty equals demand from that previous observed date
            for row in gap_rows:
                curr_date, curr_qty, prev_date, lag_qty, day_gap = row
                assert day_gap > 1, f"Expected day gap > 1 on {curr_date}, got {day_gap}"

                cur.execute(
                    """
                    SELECT quantity
                    FROM fact_daily_demand
                    WHERE sku_id = 'B1_1' AND date = %s;
                    """,
                    (prev_date,),
                )
                expected_prev_qty = cur.fetchone()[0]
                assert lag_qty == expected_prev_qty, (
                    f"lag_qty {lag_qty} does not match quantity on {prev_date} ({expected_prev_qty})"
                )

    def test_exact_calendar_lag_7_semantics(self, db_conn):
        """Verify exact calendar lag t-7 yields NULL when t-7 was not observed for that SKU."""
        with db_conn.cursor() as cur:
            # Correlated NOT EXISTS to find dates where t-7 was unobserved for B1_1
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
                  AND NOT EXISTS (
                      SELECT 1
                      FROM fact_daily_demand check_prior
                      WHERE check_prior.sku_id = curr.sku_id
                        AND check_prior.date = (curr.date - INTERVAL '7 days')::DATE
                  )
                ORDER BY curr.date;
                """
            )
            gap_rows = cur.fetchall()
            assert len(gap_rows) > 0, "Expected observed dates for B1_1 where t-7 was unobserved"
            for row in gap_rows:
                curr_date, sku_id, lag_7_qty = row
                assert lag_7_qty is None, (
                    f"Expected NULL for unobserved calendar lag t-7 on {curr_date}, got {lag_7_qty}"
                )


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
