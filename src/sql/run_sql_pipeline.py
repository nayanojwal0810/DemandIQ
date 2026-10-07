"""End-to-end execution runner for the DemandIQ PostgreSQL analytical pipeline."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.sql.db import execute_sql_file, get_connection, load_staging_copy

logger = logging.getLogger(__name__)


def initialize_database(conn) -> None:
    """Create schema, staging, dimension, and fact tables."""
    sql_base = Path("sql")
    ddl_scripts = [
        sql_base / "schema" / "create_schema.sql",
        sql_base / "staging" / "create_staging_tables.sql",
        sql_base / "dimensions" / "create_dim_brand.sql",
        sql_base / "dimensions" / "create_dim_sku.sql",
        sql_base / "dimensions" / "create_dim_date.sql",
        sql_base / "facts" / "create_fact_daily_demand.sql",
    ]
    for script in ddl_scripts:
        execute_sql_file(conn, script)
    logger.info("Database schema and tables initialized.")


def populate_warehouse(
    conn,
    csv_path: Path = Path("data/processed/sku_demand_daily.csv"),
) -> None:
    """Load canonical CSV into staging and populate dimensions and facts."""
    load_staging_copy(conn, csv_path)
    execute_sql_file(conn, Path("sql/staging/load_staging_data.sql"))
    logger.info("Dimensions and fact table populated successfully.")


def run_reconciliation(
    conn,
    canonical_csv_path: Path = Path("data/processed/sku_demand_daily.csv"),
) -> None:
    """Verify that fact table mathematically reconciles with canonical CSV.

    Checks:
        1. Grand total quantity and promotions.
        2. SKU-level sums across all 118 SKUs.
        3. Brand-date aggregations across all 7,192 combinations.
    """
    df_canonical = pd.read_csv(canonical_csv_path)

    with conn.cursor() as cur:
        # 1. Total row count, quantity, and promotions
        cur.execute("SELECT COUNT(*), SUM(quantity), SUM(promotion) FROM fact_daily_demand;")
        fact_count, fact_qty, fact_promo = cur.fetchone()

        canon_count = len(df_canonical)
        canon_qty = float(df_canonical["quantity"].sum())
        canon_promo = int(df_canonical["promotion"].sum())

        if fact_count != canon_count:
            raise ValueError(f"Row count mismatch: SQL fact={fact_count}, Canonical={canon_count}")
        if not np.isclose(fact_qty, canon_qty, atol=1e-5):
            raise ValueError(f"Quantity sum mismatch: SQL fact={fact_qty}, Canonical={canon_qty}")
        if fact_promo != canon_promo:
            raise ValueError(f"Promotion count mismatch: SQL fact={fact_promo}, Canonical={canon_promo}")

        logger.info(
            "Portfolio reconciliation passed: %d rows, %.2f units, %d promo instances.",
            fact_count,
            fact_qty,
            fact_promo,
        )

        # 2. SKU-level quantity reconciliation
        cur.execute(
            """
            SELECT sku_id, SUM(quantity) AS sql_qty, SUM(promotion) AS sql_promo
            FROM fact_daily_demand
            GROUP BY sku_id
            ORDER BY sku_id;
            """
        )
        sql_sku_rows = cur.fetchall()
        df_sql_sku = pd.DataFrame(sql_sku_rows, columns=["sku_id", "sql_qty", "sql_promo"])

        canon_sku_agg = (
            df_canonical.groupby("sku_id")
            .agg(canon_qty=("quantity", "sum"), canon_promo=("promotion", "sum"))
            .reset_index()
        )

        merged_sku = pd.merge(df_sql_sku, canon_sku_agg, on="sku_id")
        if len(merged_sku) != 118:
            raise ValueError(f"Expected 118 reconciled SKUs, got {len(merged_sku)}")

        qty_diff = np.abs(merged_sku["sql_qty"] - merged_sku["canon_qty"]).max()
        promo_diff = np.abs(merged_sku["sql_promo"] - merged_sku["canon_promo"]).max()

        if qty_diff > 1e-5:
            raise ValueError(f"SKU quantity reconciliation failed with max diff: {qty_diff}")
        if promo_diff > 0:
            raise ValueError(f"SKU promotion reconciliation failed with max diff: {promo_diff}")

        logger.info("SKU-level reconciliation passed for all 118 SKUs (max diff = %.6f).", qty_diff)

        # 3. Brand-date aggregation reconciliation
        cur.execute(
            """
            SELECT date::TEXT, brand_id, SUM(quantity) AS sql_qty
            FROM fact_daily_demand
            GROUP BY date, brand_id
            ORDER BY date, brand_id;
            """
        )
        sql_bd_rows = cur.fetchall()
        df_sql_bd = pd.DataFrame(sql_bd_rows, columns=["date", "brand_id", "sql_qty"])

        canon_bd = (
            df_canonical.groupby(["date", "brand_id"])["quantity"]
            .sum()
            .reset_index(name="canon_qty")
        )

        merged_bd = pd.merge(df_sql_bd, canon_bd, on=["date", "brand_id"])
        if len(merged_bd) != len(canon_bd):
            raise ValueError(f"Brand-date count mismatch: SQL={len(merged_bd)}, Canonical={len(canon_bd)}")

        bd_diff = np.abs(merged_bd["sql_qty"] - merged_bd["canon_qty"]).max()
        if bd_diff > 1e-5:
            raise ValueError(f"Brand-date quantity reconciliation failed with max diff: {bd_diff}")

        logger.info("Brand-date reconciliation passed for all %d cells (max diff = %.6f).", len(merged_bd), bd_diff)


def run_analytical_cross_checks(conn) -> None:
    """Compare SQL development analytics (<= 2017-12-31) against saved Python summaries."""
    logger.info("Running SQL vs Python analytical cross-checks (development period <= 2017-12-31)...")

    # 1. Brand Summary Cross-Check
    df_py_brand = pd.read_csv("data/processed/brand_summary.csv")
    with conn.cursor() as cur:
        brand_sql = Path("sql/analytics/brand_demand_summary.sql").read_text(encoding="utf-8")
        cur.execute(brand_sql)
        sql_brand_rows = cur.fetchall()

    df_sql_brand = pd.DataFrame(
        sql_brand_rows,
        columns=[
            "brand_id",
            "sku_count",
            "total_demand",
            "mean_daily_demand",
            "portfolio_demand_share",
            "average_zero_rate",
        ],
    )
    for c in ["sku_count", "total_demand", "mean_daily_demand", "portfolio_demand_share", "average_zero_rate"]:
        df_sql_brand[c] = df_sql_brand[c].astype(float)
    for c in ["sku_count", "total_demand", "mean_daily_demand", "portfolio_demand_share", "avg_zero_demand_rate"]:
        df_py_brand[c] = df_py_brand[c].astype(float)

    # Explicitly prefix comparison columns before merging
    sql_b = df_sql_brand.rename(
        columns={c: f"sql_{c}" for c in df_sql_brand.columns if c != "brand_id"}
    )
    py_b = df_py_brand.rename(
        columns={c: f"py_{c}" for c in df_py_brand.columns if c != "brand_id"}
    )
    merged_brand = pd.merge(sql_b, py_b, on="brand_id")

    if len(merged_brand) != 4:
        raise ValueError(f"Expected 4 brands in cross-check, got {len(merged_brand)}")

    for _, row in merged_brand.iterrows():
        b = row["brand_id"]
        if int(row["sql_sku_count"]) != int(row["py_sku_count"]):
            raise ValueError(f"Brand {b} SKU count mismatch: SQL={row['sql_sku_count']}, Py={row['py_sku_count']}")
        if not np.isclose(float(row["sql_total_demand"]), float(row["py_total_demand"]), atol=1e-4):
            raise ValueError(f"Brand {b} total demand mismatch: SQL={row['sql_total_demand']}, Py={row['py_total_demand']}")
        if not np.isclose(float(row["sql_mean_daily_demand"]), float(row["py_mean_daily_demand"]), atol=1e-3):
            raise ValueError(f"Brand {b} mean daily demand mismatch: SQL={row['sql_mean_daily_demand']}, Py={row['py_mean_daily_demand']}")
        if not np.isclose(float(row["sql_portfolio_demand_share"]), float(row["py_portfolio_demand_share"]), atol=1e-4):
            raise ValueError(f"Brand {b} portfolio share mismatch: SQL={row['sql_portfolio_demand_share']}, Py={row['py_portfolio_demand_share']}")
        if not np.isclose(float(row["sql_average_zero_rate"]), float(row["py_avg_zero_demand_rate"]), atol=1e-3):
            raise ValueError(f"Brand {b} zero rate mismatch: SQL={row['sql_average_zero_rate']}, Py={row['py_avg_zero_demand_rate']}")
    logger.info("SQL Brand Demand Summary matches Python output across all brands and metrics.")

    # 2. SKU Demand Profile Cross-Check
    df_py_sku = pd.read_csv("data/processed/demand_profile.csv")
    with conn.cursor() as cur:
        sku_sql = Path("sql/analytics/sku_demand_profile.sql").read_text(encoding="utf-8")
        cur.execute(sku_sql)
        sql_sku_rows = cur.fetchall()

    df_sql_sku = pd.DataFrame(
        sql_sku_rows,
        columns=[
            "brand_id",
            "sku_id",
            "total_demand",
            "mean_daily_demand",
            "median_daily_demand",
            "std_daily_demand",
            "min_daily_demand",
            "max_daily_demand",
            "zero_demand_rate",
            "nonzero_demand_rate",
        ],
    )
    for c in [
        "total_demand", "mean_daily_demand", "median_daily_demand", "std_daily_demand",
        "min_daily_demand", "max_daily_demand", "zero_demand_rate", "nonzero_demand_rate"
    ]:
        df_sql_sku[c] = df_sql_sku[c].astype(float)
        df_py_sku[c] = df_py_sku[c].astype(float)

    sql_s = df_sql_sku.rename(
        columns={c: f"sql_{c}" for c in df_sql_sku.columns if c not in ["brand_id", "sku_id"]}
    )
    py_s = df_py_sku.rename(
        columns={c: f"py_{c}" for c in df_py_sku.columns if c not in ["brand_id", "sku_id"]}
    )
    merged_sku = pd.merge(sql_s, py_s, on=["brand_id", "sku_id"])

    if len(merged_sku) != 118:
        raise ValueError(f"Expected 118 SKUs in demand profile cross-check, got {len(merged_sku)}")

    demand_diff = np.abs(merged_sku["sql_total_demand"] - merged_sku["py_total_demand"]).max()
    mean_diff = np.abs(merged_sku["sql_mean_daily_demand"] - merged_sku["py_mean_daily_demand"]).max()
    median_diff = np.abs(merged_sku["sql_median_daily_demand"] - merged_sku["py_median_daily_demand"]).max()
    std_diff = np.abs(merged_sku["sql_std_daily_demand"] - merged_sku["py_std_daily_demand"]).max()
    min_diff = np.abs(merged_sku["sql_min_daily_demand"] - merged_sku["py_min_daily_demand"]).max()
    max_diff = np.abs(merged_sku["sql_max_daily_demand"] - merged_sku["py_max_daily_demand"]).max()
    zero_diff = np.abs(merged_sku["sql_zero_demand_rate"] - merged_sku["py_zero_demand_rate"]).max()

    if demand_diff > 1e-4:
        raise ValueError(f"SKU demand profile total demand mismatch: max diff = {demand_diff}")
    if mean_diff > 1e-3:
        raise ValueError(f"SKU demand profile mean demand mismatch: max diff = {mean_diff}")
    if median_diff > 1e-4:
        raise ValueError(f"SKU demand profile median demand mismatch: max diff = {median_diff}")
    if std_diff > 1e-3:
        raise ValueError(f"SKU demand profile std demand mismatch: max diff = {std_diff}")
    if min_diff > 1e-4:
        raise ValueError(f"SKU demand profile min demand mismatch: max diff = {min_diff}")
    if max_diff > 1e-4:
        raise ValueError(f"SKU demand profile max demand mismatch: max diff = {max_diff}")
    if zero_diff > 1e-3:
        raise ValueError(f"SKU demand profile zero rate mismatch: max diff = {zero_diff}")

    logger.info("SQL SKU Demand Profile matches Python profiles across all 118 SKUs (all metrics).")

    # 3. Promotion Analysis Cross-Check
    df_py_promo = pd.read_csv("data/processed/promotion_summary.csv")
    with conn.cursor() as cur:
        promo_sql = Path("sql/analytics/promotion_analysis.sql").read_text(encoding="utf-8")
        cur.execute(promo_sql)
        sql_promo_rows = cur.fetchall()

    df_sql_promo = pd.DataFrame(
        sql_promo_rows,
        columns=[
            "brand_id",
            "sku_id",
            "total_obs",
            "promo_obs",
            "promo_rate",
            "mean_promo_demand",
            "mean_non_promo_demand",
        ],
    )
    for c in ["total_obs", "promo_obs", "promo_rate", "mean_promo_demand", "mean_non_promo_demand"]:
        df_sql_promo[c] = df_sql_promo[c].astype(float)
        df_py_promo[c] = df_py_promo[c].astype(float)

    sql_p = df_sql_promo.rename(
        columns={c: f"sql_{c}" for c in df_sql_promo.columns if c not in ["brand_id", "sku_id"]}
    )
    py_p = df_py_promo.rename(
        columns={c: f"py_{c}" for c in df_py_promo.columns if c not in ["brand_id", "sku_id"]}
    )
    merged_promo = pd.merge(sql_p, py_p, on=["brand_id", "sku_id"])

    if len(merged_promo) != 118:
        raise ValueError(f"Expected 118 SKUs in promo cross-check, got {len(merged_promo)}")

    obs_diff = np.abs(merged_promo["sql_total_obs"] - merged_promo["py_total_obs"]).max()
    promo_obs_diff = np.abs(merged_promo["sql_promo_obs"] - merged_promo["py_promo_obs"]).max()
    rate_diff = np.abs(merged_promo["sql_promo_rate"] - merged_promo["py_promo_rate"]).max()
    promo_mean_diff = np.abs(merged_promo["sql_mean_promo_demand"] - merged_promo["py_mean_promo_demand"]).max()
    non_promo_mean_diff = np.abs(merged_promo["sql_mean_non_promo_demand"] - merged_promo["py_mean_non_promo_demand"]).max()

    if obs_diff > 0:
        raise ValueError(f"Total observations mismatch: max diff = {obs_diff}")
    if promo_obs_diff > 0:
        raise ValueError(f"Promotion observations mismatch: max diff = {promo_obs_diff}")
    if rate_diff > 1e-3:
        raise ValueError(f"Promotion rate mismatch: max diff = {rate_diff}")
    if promo_mean_diff > 1e-3:
        raise ValueError(f"Mean promo demand mismatch: max diff = {promo_mean_diff}")
    if non_promo_mean_diff > 1e-3:
        raise ValueError(f"Mean non-promo demand mismatch: max diff = {non_promo_mean_diff}")

    logger.info("SQL Promotion Analysis matches Python promotion summaries across all 118 SKUs.")



def run_pipeline() -> None:
    """Execute complete DemandIQ SQL data warehousing and analytical pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger.info("Starting DemandIQ PostgreSQL analytical pipeline...")

    conn = get_connection()
    try:
        initialize_database(conn)
        populate_warehouse(conn)
        run_reconciliation(conn)
        run_analytical_cross_checks(conn)
        logger.info("All PostgreSQL stages and reconciliation checks completed successfully.")
    finally:
        conn.close()
        logger.info("PostgreSQL connection closed.")


if __name__ == "__main__":
    run_pipeline()
