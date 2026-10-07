"""Data contract and deterministic integrity validation for DemandIQ.

Implements structural schema checks, key uniqueness constraints, value sanity
rules, temporal continuity checks, and source-to-canonical mathematical reconciliation.
"""

from typing import Dict, Any, List
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ["date", "brand_id", "sku_id", "quantity", "promotion"]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate canonical schema column names, count, and datatypes.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical SKU demand DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing, extra columns exist, or types are invalid.
    """
    actual_columns = list(df.columns)
    if actual_columns != CANONICAL_COLUMNS:
        raise ValueError(
            f"Schema mismatch. Expected columns {CANONICAL_COLUMNS}, got {actual_columns}"
        )

    # Check non-null constraints
    null_counts = df.isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values detected in canonical columns: {cols_with_nulls}")

    # Check types
    if not np.issubdtype(df["quantity"].dtype, np.integer):
        raise ValueError(f"Quantity column must be integer type, got {df['quantity'].dtype}")

    if not np.issubdtype(df["promotion"].dtype, np.integer):
        raise ValueError(f"Promotion column must be integer type, got {df['promotion'].dtype}")

    logger.info("Schema check passed: exact column structure and data types verified.")


def validate_keys(df: pd.DataFrame) -> None:
    """Validate primary composite key uniqueness.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical SKU demand DataFrame.

    Raises
    ------
    ValueError
        If duplicate (date, sku_id) observations exist.
    """
    dup_keys = df.duplicated(subset=["date", "sku_id"], keep=False)
    dup_count = int(dup_keys.sum())
    if dup_count > 0:
        sample_dups = df[dup_keys].head(6)[["date", "sku_id", "brand_id", "quantity"]]
        raise ValueError(
            f"Key uniqueness violation: found {dup_count} duplicate (date, sku_id) records. "
            f"Sample:\n{sample_dups.to_string()}"
        )

    dup_brand_keys = df.duplicated(subset=["date", "brand_id", "sku_id"], keep=False)
    if dup_brand_keys.any():
        raise ValueError("Key uniqueness violation: found duplicate (date, brand_id, sku_id) records.")

    logger.info("Key uniqueness check passed: (date, sku_id) is strictly unique.")


def validate_values(df: pd.DataFrame) -> None:
    """Validate value bounds and sanity constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical SKU demand DataFrame.

    Raises
    ------
    ValueError
        If negative quantities, non-binary promotions, or invalid dates exist.
    """
    # 1. Date parseability
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates_count = int(parsed_dates.isnull().sum())
    if invalid_dates_count > 0:
        raise ValueError(f"Date validation failed: {invalid_dates_count} unparseable dates found.")

    # 2. Quantity constraints
    min_qty = df["quantity"].min()
    if min_qty < 0:
        neg_count = int((df["quantity"] < 0).sum())
        raise ValueError(f"Quantity constraint failed: {neg_count} negative demand values found (min={min_qty}).")

    if np.isinf(df["quantity"].values).any():
        raise ValueError("Quantity constraint failed: infinite demand values found.")

    # 3. Promotion binary constraints
    unique_promos = set(df["promotion"].unique())
    if not unique_promos.issubset({0, 1}):
        raise ValueError(f"Promotion constraint failed: expected binary values in {{0, 1}}, got {unique_promos}")

    logger.info("Value check passed: non-negative demand, binary promotion, and valid dates.")


def check_temporal_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect and report temporal structure and calendar continuity.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical SKU demand DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary with temporal summary statistics.
    """
    unique_dates = pd.to_datetime(df["date"].unique()).sort_values()
    min_date = unique_dates.min()
    max_date = unique_dates.max()
    obs_date_count = len(unique_dates)

    expected_span = (max_date - min_date).days + 1
    full_calendar = pd.date_range(start=min_date, end=max_date, freq="D")
    missing_dates = sorted(list(set(full_calendar) - set(unique_dates)))

    dup_dates_count = int(pd.Series(df["date"].unique()).duplicated().sum())

    summary = {
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()),
        "observed_date_count": obs_date_count,
        "expected_calendar_span_days": expected_span,
        "missing_calendar_dates_count": len(missing_dates),
        "missing_calendar_dates": [str(d.date()) for d in missing_dates],
        "duplicate_date_count": dup_dates_count,
    }
    logger.info(
        "Temporal structure: %s to %s, %d observed dates, %d calendar gaps.",
        summary["min_date"],
        summary["max_date"],
        summary["observed_date_count"],
        summary["missing_calendar_dates_count"],
    )
    return summary


def check_sku_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect SKU coverage and brand allocation.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical SKU demand DataFrame.

    Returns
    -------
    Dict[str, Any]
        SKU and brand coverage metrics.

    Raises
    ------
    ValueError
        If an SKU maps to multiple brands.
    """
    # Verify 1-to-1 SKU -> Brand mapping
    sku_to_brands = df.groupby("sku_id")["brand_id"].nunique()
    multi_brand_skus = sku_to_brands[sku_to_brands > 1]
    if len(multi_brand_skus) > 0:
        raise ValueError(f"SKU mapping violation: SKUs mapped to multiple brands: {multi_brand_skus.to_dict()}")

    unique_skus = df["sku_id"].unique()
    unique_dates_count = df["date"].nunique()

    # Verify every SKU has full date coverage
    sku_obs_counts = df.groupby("sku_id")["date"].count()
    incomplete_skus = sku_obs_counts[sku_obs_counts != unique_dates_count]
    if len(incomplete_skus) > 0:
        raise ValueError(
            f"SKU coverage violation: {len(incomplete_skus)} SKUs do not have {unique_dates_count} observations. "
            f"Details: {incomplete_skus.to_dict()}"
        )

    brand_sku_counts = df.groupby("brand_id")["sku_id"].nunique().to_dict()

    summary = {
        "total_skus": len(unique_skus),
        "brand_sku_counts": brand_sku_counts,
        "observations_per_sku": unique_dates_count,
        "total_sku_day_records": len(df),
    }
    logger.info(
        "SKU coverage check passed: %d SKUs across %d brands, perfectly uniform coverage (%d days each).",
        len(unique_skus),
        len(brand_sku_counts),
        unique_dates_count,
    )
    return summary


def reconcile_source_to_canonical(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Mathematically reconcile long canonical data back to wide source data.

    Parameters
    ----------
    source_df : pd.DataFrame
        Raw wide source dataset.
    canonical_df : pd.DataFrame
        Long canonical dataset.

    Returns
    -------
    Dict[str, Any]
        Summary of reconciliation results.

    Raises
    ------
    ValueError
        If any quantity or promotion sums or records fail to reconcile exactly.
    """
    # 1. Total portfolio quantity reconciliation
    qty_cols = [c for c in source_df.columns if c.startswith("QTY_")]
    source_total_qty = int(source_df[qty_cols].values.sum())
    canonical_total_qty = int(canonical_df["quantity"].sum())

    if source_total_qty != canonical_total_qty:
        raise ValueError(
            f"Portfolio total quantity reconciliation failed: "
            f"Source total = {source_total_qty}, Canonical total = {canonical_total_qty}"
        )

    # 2. Per-SKU quantity sum reconciliation
    mismatched_skus: List[str] = []
    for c in qty_cols:
        sku = c.replace("QTY_", "")
        src_sum = int(source_df[c].sum())
        can_sum = int(canonical_df.loc[canonical_df["sku_id"] == sku, "quantity"].sum())
        if src_sum != can_sum:
            mismatched_skus.append(f"{sku} (source={src_sum}, canonical={can_sum})")

    if mismatched_skus:
        raise ValueError(f"SKU quantity sum reconciliation failed for {len(mismatched_skus)} SKUs: {mismatched_skus[:5]}")

    # 3. Date + Brand quantity sum reconciliation
    brands = sorted(canonical_df["brand_id"].unique())
    source_dates = source_df["DATE"].astype(str).values

    for b in brands:
        b_qty_cols = [c for c in qty_cols if c.startswith(f"QTY_{b}_")]
        src_b_daily = source_df.set_index("DATE")[b_qty_cols].sum(axis=1)

        can_b_daily = (
            canonical_df[canonical_df["brand_id"] == b]
            .groupby("date")["quantity"]
            .sum()
        )

        diff = (src_b_daily.loc[can_b_daily.index] - can_b_daily).abs().max()
        if diff != 0:
            raise ValueError(f"Brand {b} daily reconciliation failed with max absolute diff {diff}")

    # 4. Total daily portfolio sum reconciliation
    src_daily_total = source_df.set_index("DATE")[qty_cols].sum(axis=1)
    can_daily_total = canonical_df.groupby("date")["quantity"].sum()
    diff_daily = (src_daily_total.loc[can_daily_total.index] - can_daily_total).abs().max()
    if diff_daily != 0:
        raise ValueError(f"Daily portfolio total reconciliation failed with max absolute diff {diff_daily}")

    # 5. Promotion alignment reconciliation
    promo_cols = [c for c in source_df.columns if c.startswith("PROMO_")]
    source_total_promo = int(source_df[promo_cols].values.sum())
    canonical_total_promo = int(canonical_df["promotion"].sum())

    if source_total_promo != canonical_total_promo:
        raise ValueError(
            f"Portfolio total promotion reconciliation failed: "
            f"Source total = {source_total_promo}, Canonical total = {canonical_total_promo}"
        )

    for c in promo_cols:
        sku = c.replace("PROMO_", "")
        src_promo_sum = int(source_df[c].sum())
        can_promo_sum = int(canonical_df.loc[canonical_df["sku_id"] == sku, "promotion"].sum())
        if src_promo_sum != can_promo_sum:
            raise ValueError(f"Promotion sum reconciliation failed for SKU {sku}")

    reconciliation_summary = {
        "portfolio_total_quantity": source_total_qty,
        "portfolio_total_promotion_days": source_total_promo,
        "sku_quantity_mismatches": 0,
        "sku_promotion_mismatches": 0,
        "brand_daily_max_diff": 0,
        "portfolio_daily_max_diff": 0,
        "status": "EXACT_RECONCILIATION_CONFIRMED",
    }
    logger.info("Source-to-canonical reconciliation passed: 100%% exact mathematical match.")
    return reconciliation_summary


def validate_canonical_dataset(
    canonical_df: pd.DataFrame,
    source_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Run all data contract checks on the canonical dataset.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical SKU demand DataFrame.
    source_df : pd.DataFrame, optional
        Source wide DataFrame for mathematical reconciliation.

    Returns
    -------
    Dict[str, Any]
        Dictionary with all validation and reconciliation outcomes.
    """
    validate_schema(canonical_df)
    validate_keys(canonical_df)
    validate_values(canonical_df)
    temporal_summary = check_temporal_structure(canonical_df)
    coverage_summary = check_sku_coverage(canonical_df)

    reconciliation_summary = None
    if source_df is not None:
        reconciliation_summary = reconcile_source_to_canonical(source_df, canonical_df)

    return {
        "schema_valid": True,
        "keys_valid": True,
        "values_valid": True,
        "temporal": temporal_summary,
        "coverage": coverage_summary,
        "reconciliation": reconciliation_summary,
    }
