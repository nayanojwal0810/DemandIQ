"""Unit and integration tests for data contract and reconciliation validation."""

import pandas as pd
import pytest

from src.validation.data_contract import (
    validate_schema,
    validate_keys,
    validate_values,
    check_temporal_structure,
    check_sku_coverage,
    reconcile_source_to_canonical,
    validate_canonical_dataset,
)


@pytest.fixture
def valid_canonical_df():
    """Create a valid canonical DataFrame fixture."""
    return pd.DataFrame({
        "date": ["2017-01-02", "2017-01-02", "2017-01-03", "2017-01-03"],
        "brand_id": ["B1", "B2", "B1", "B2"],
        "sku_id": ["B1_1", "B2_1", "B1_1", "B2_1"],
        "quantity": [10, 20, 5, 0],
        "promotion": [1, 0, 0, 1],
    })


@pytest.fixture
def valid_source_df():
    """Create a matching source DataFrame fixture."""
    return pd.DataFrame({
        "DATE": ["2017-01-02", "2017-01-03"],
        "QTY_B1_1": [10, 5],
        "QTY_B2_1": [20, 0],
        "PROMO_B1_1": [1, 0],
        "PROMO_B2_1": [0, 1],
    })


def test_validate_schema_success(valid_canonical_df):
    """Test valid schema passes without error."""
    validate_schema(valid_canonical_df)


def test_validate_schema_missing_column(valid_canonical_df):
    """Test schema failure when a required column is missing."""
    df_missing = valid_canonical_df.drop(columns=["promotion"])
    with pytest.raises(ValueError, match="Schema mismatch"):
        validate_schema(df_missing)


def test_validate_schema_null_values(valid_canonical_df):
    """Test schema failure when null values exist."""
    df_null = valid_canonical_df.copy()
    df_null.loc[0, "quantity"] = None
    with pytest.raises(ValueError, match="Null values detected"):
        validate_schema(df_null)


def test_validate_keys_duplicate_detection(valid_canonical_df):
    """Test error when duplicate (date, sku_id) keys exist."""
    df_dup = pd.concat([valid_canonical_df, valid_canonical_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Key uniqueness violation"):
        validate_keys(df_dup)


def test_validate_values_negative_quantity(valid_canonical_df):
    """Test error on negative demand value."""
    df_neg = valid_canonical_df.copy()
    df_neg.loc[0, "quantity"] = -1
    with pytest.raises(ValueError, match="Quantity constraint failed.*negative"):
        validate_values(df_neg)


def test_validate_values_non_binary_promotion(valid_canonical_df):
    """Test error on non-binary promotion value."""
    df_promo = valid_canonical_df.copy()
    df_promo.loc[0, "promotion"] = 2
    with pytest.raises(ValueError, match="Promotion constraint failed"):
        validate_values(df_promo)


def test_reconcile_source_to_canonical_success(valid_source_df, valid_canonical_df):
    """Test exact mathematical reconciliation between source and canonical."""
    res = reconcile_source_to_canonical(valid_source_df, valid_canonical_df)
    assert res["status"] == "EXACT_RECONCILIATION_CONFIRMED"
    assert res["portfolio_total_quantity"] == 35
    assert res["portfolio_total_promotion_days"] == 2


def test_reconcile_source_to_canonical_quantity_mismatch(valid_source_df, valid_canonical_df):
    """Test reconciliation error on quantity tampering."""
    tampered_df = valid_canonical_df.copy()
    tampered_df.loc[0, "quantity"] = 999
    with pytest.raises(ValueError, match="reconciliation failed"):
        reconcile_source_to_canonical(valid_source_df, tampered_df)


def test_check_temporal_structure(valid_canonical_df):
    """Test temporal structure calculation."""
    res = check_temporal_structure(valid_canonical_df)
    assert res["min_date"] == "2017-01-02"
    assert res["max_date"] == "2017-01-03"
    assert res["observed_date_count"] == 2
    assert res["expected_calendar_span_days"] == 2
    assert res["missing_calendar_dates_count"] == 0


def test_check_sku_coverage_multi_brand_violation(valid_canonical_df):
    """Test error when SKU maps to multiple brands."""
    violation_df = valid_canonical_df.copy()
    violation_df.loc[0, "brand_id"] = "B99"  # B1_1 now maps to both B1 and B99
    with pytest.raises(ValueError, match="SKU mapping violation"):
        check_sku_coverage(violation_df)
