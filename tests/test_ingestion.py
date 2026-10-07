"""Unit and integration tests for data ingestion pipeline."""

from pathlib import Path
import pandas as pd
import pytest

from src.ingestion.ingest_sales_data import (
    identify_source_columns,
    parse_sku_identity,
    transform_wide_to_canonical,
    extract_sku_dimension,
    ingest_sales_data,
)


@pytest.fixture
def synthetic_wide_df():
    """Create a minimal synthetic wide dataset."""
    return pd.DataFrame({
        "DATE": ["2017-01-02", "2017-01-03"],
        "QTY_B1_1": [10, 15],
        "QTY_B1_2": [5, 0],
        "QTY_B2_1": [20, 25],
        "PROMO_B1_1": [1, 0],
        "PROMO_B1_2": [0, 0],
        "PROMO_B2_1": [1, 1],
    })


def test_identify_source_columns_success(synthetic_wide_df):
    """Test successful identification of date, quantity, and promo columns."""
    date_col, qty_cols, promo_cols = identify_source_columns(list(synthetic_wide_df.columns))
    assert date_col == "DATE"
    assert qty_cols == ["QTY_B1_1", "QTY_B1_2", "QTY_B2_1"]
    assert promo_cols == ["PROMO_B1_1", "PROMO_B1_2", "PROMO_B2_1"]


def test_identify_source_columns_missing_date():
    """Test error when date column is absent."""
    cols = ["QTY_B1_1", "PROMO_B1_1"]
    with pytest.raises(ValueError, match="DATE"):
        identify_source_columns(cols)


def test_identify_source_columns_mismatched_promo():
    """Test error when promo columns do not match quantity columns."""
    cols = ["DATE", "QTY_B1_1", "QTY_B1_2", "PROMO_B1_1"]
    with pytest.raises(ValueError, match="Mismatch between QTY and PROMO"):
        identify_source_columns(cols)


def test_parse_sku_identity():
    """Test parsing brand and SKU identifiers."""
    assert parse_sku_identity("QTY_B1_1") == ("B1", "B1_1")
    assert parse_sku_identity("PROMO_B2_45") == ("B2", "B2_45")
    assert parse_sku_identity("QTY_B4_10") == ("B4", "B4_10")


def test_wide_to_long_transformation(synthetic_wide_df):
    """Test wide-to-long transformation logic and alignment."""
    date_col, qty_cols, promo_cols = identify_source_columns(list(synthetic_wide_df.columns))
    canonical_df = transform_wide_to_canonical(synthetic_wide_df, date_col, qty_cols, promo_cols)

    # 2 dates * 3 SKUs = 6 rows
    assert len(canonical_df) == 6
    assert list(canonical_df.columns) == ["date", "brand_id", "sku_id", "quantity", "promotion"]

    # Verify specific row values
    row_b1_1 = canonical_df[(canonical_df["date"] == "2017-01-02") & (canonical_df["sku_id"] == "B1_1")].iloc[0]
    assert row_b1_1["brand_id"] == "B1"
    assert row_b1_1["quantity"] == 10
    assert row_b1_1["promotion"] == 1

    row_b1_2 = canonical_df[(canonical_df["date"] == "2017-01-03") & (canonical_df["sku_id"] == "B1_2")].iloc[0]
    assert row_b1_2["quantity"] == 0
    assert row_b1_2["promotion"] == 0


def test_deterministic_sorting(synthetic_wide_df):
    """Test that canonical DataFrame is sorted by date, brand_id, sku_id."""
    date_col, qty_cols, promo_cols = identify_source_columns(list(synthetic_wide_df.columns))
    canonical_df = transform_wide_to_canonical(synthetic_wide_df, date_col, qty_cols, promo_cols)

    expected_keys = [
        ("2017-01-02", "B1", "B1_1"),
        ("2017-01-02", "B1", "B1_2"),
        ("2017-01-02", "B2", "B2_1"),
        ("2017-01-03", "B1", "B1_1"),
        ("2017-01-03", "B1", "B1_2"),
        ("2017-01-03", "B2", "B2_1"),
    ]
    actual_keys = list(zip(canonical_df["date"], canonical_df["brand_id"], canonical_df["sku_id"]))
    assert actual_keys == expected_keys


def test_extract_sku_dimension(synthetic_wide_df):
    """Test SKU dimension extraction."""
    date_col, qty_cols, promo_cols = identify_source_columns(list(synthetic_wide_df.columns))
    canonical_df = transform_wide_to_canonical(synthetic_wide_df, date_col, qty_cols, promo_cols)
    dim_df = extract_sku_dimension(canonical_df)

    assert len(dim_df) == 3
    assert list(dim_df.columns) == ["brand_id", "sku_id"]
    assert list(dim_df["sku_id"]) == ["B1_1", "B1_2", "B2_1"]


def test_real_data_ingestion_and_determinism(tmp_path):
    """Test real-data ingestion and verify determinism across multiple runs."""
    real_raw_path = Path("data/raw/hierarchical_sales_data.csv")
    if not real_raw_path.exists():
        pytest.skip("Real raw dataset not present for integration test.")

    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    df1, dim1 = ingest_sales_data(raw_csv_path=real_raw_path, processed_dir=out_dir_1)
    df2, dim2 = ingest_sales_data(raw_csv_path=real_raw_path, processed_dir=out_dir_2)

    # 1,798 dates * 118 SKUs = 212,164 rows
    assert len(df1) == 212164
    assert len(dim1) == 118

    # Verify identical output between run 1 and run 2
    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(dim1, dim2)
