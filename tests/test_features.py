"""Unit and leakage tests for the forecasting feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_forecasting_features import (
    build_forecasting_feature_table,
    load_forecasting_features,
)
from src.features.feature_config import (
    CANONICAL_FEATURE_COLUMNS,
    DEVELOPMENT_CUTOFF_DATE,
    FEATURE_COLUMNS_NO_TARGET_PROMO,
    FEATURE_COLUMNS_WITH_TARGET_PROMO,
)


@pytest.fixture
def synthetic_sku_data() -> pd.DataFrame:
    """Create a synthetic daily demand table with known gaps and known values."""
    # Dates: 2014-01-02 to 2014-01-20, skipping 2014-01-05 and 2014-01-12
    dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range("2014-01-02", "2014-01-20")
        if d.strftime("%Y-%m-%d") not in {"2014-01-05", "2014-01-12"}
    ]
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "brand_id": ["B1"] * n,
            "sku_id": ["B1_1"] * n,
            "quantity": [float(i + 1) for i in range(n)],
            "promotion": [1 if i % 3 == 0 else 0 for i in range(n)],
        }
    )


def test_leakage_1_lag_1_previous_observed_date():
    """Verify lag_1 is the immediately preceding observed date demand, not assuming contiguous calendar."""
    # Dates with a gap: Jan 2, Jan 4 (Jan 3 missing)
    df = pd.DataFrame(
        {
            "date": ["2014-01-02", "2014-01-04", "2014-01-07"],
            "brand_id": ["B1", "B1", "B1"],
            "sku_id": ["B1_1", "B1_1", "B1_1"],
            "quantity": [10.0, 25.0, 40.0],
            "promotion": [0, 1, 0],
        }
    )
    features = build_forecasting_feature_table(df)

    assert pd.isna(features.loc[features["date"] == "2014-01-02", "lag_1"].values[0])
    assert features.loc[features["date"] == "2014-01-04", "lag_1"].values[0] == 10.0
    assert features.loc[features["date"] == "2014-01-07", "lag_1"].values[0] == 25.0
    assert (
        features.loc[features["date"] == "2014-01-04", "days_since_previous_observation"].values[0]
        == 2.0
    )


def test_leakage_2_calendar_lag_exact_lookup_mandatory():
    """Verify lag_7 performs exact calendar date t-7 lookup and does NOT row-shift."""
    # Jan 02, 03, 09, 10, 11 (Jan 04 was not observed, so Jan 11 - 7 = Jan 04 is missing)
    df = pd.DataFrame(
        {
            "date": ["2014-01-02", "2014-01-03", "2014-01-09", "2014-01-10", "2014-01-11"],
            "brand_id": ["B1"] * 5,
            "sku_id": ["B1_1"] * 5,
            "quantity": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promotion": [0, 0, 0, 0, 0],
        }
    )
    features = build_forecasting_feature_table(df)

    # 2014-01-09 - 7 days = 2014-01-02 (exists, quantity 10.0)
    assert features.loc[features["date"] == "2014-01-09", "lag_7"].values[0] == 10.0

    # 2014-01-10 - 7 days = 2014-01-03 (exists, quantity 20.0)
    assert features.loc[features["date"] == "2014-01-10", "lag_7"].values[0] == 20.0

    # 2014-01-11 - 7 days = 2014-01-04 (does NOT exist in source, must be NaN)
    # A simple row shift would have incorrectly returned 10.0 or 20.0
    val_11 = features.loc[features["date"] == "2014-01-11", "lag_7"].values[0]
    assert pd.isna(val_11), f"Expected NaN for lag_7 on 2014-01-11, got {val_11}"


def test_leakage_3_rolling_window_excludes_target():
    """Verify that changing target day's quantity leaves rolling statistics unchanged."""
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2014-01-02", "2014-01-10")]
    n = len(dates)

    df1 = pd.DataFrame(
        {
            "date": dates,
            "brand_id": ["B1"] * n,
            "sku_id": ["B1_1"] * n,
            "quantity": [10.0] * (n - 1) + [10.0],
            "promotion": [0] * n,
        }
    )

    df2 = pd.DataFrame(
        {
            "date": dates,
            "brand_id": ["B1"] * n,
            "sku_id": ["B1_1"] * n,
            "quantity": [10.0] * (n - 1) + [99999.0],  # Extreme target change on last day
            "promotion": [0] * n,
        }
    )

    feat1 = build_forecasting_feature_table(df1)
    feat2 = build_forecasting_feature_table(df2)

    # Rolling stats for the target day (last day) must be identical because target is excluded
    last_date = dates[-1]
    for col in ["rolling_mean_7", "rolling_std_7", "observed_days_7"]:
        v1 = feat1.loc[feat1["date"] == last_date, col].values[0]
        v2 = feat2.loc[feat2["date"] == last_date, col].values[0]
        assert v1 == v2 or (pd.isna(v1) and pd.isna(v2)), (
            f"Feature {col} leaked target value: {v1} != {v2}"
        )


def test_leakage_4_future_sentinel():
    """Verify that adding a future observation does not alter earlier target feature values."""
    dates_past = [d.strftime("%Y-%m-%d") for d in pd.date_range("2014-01-02", "2014-01-15")]
    n = len(dates_past)

    df_base = pd.DataFrame(
        {
            "date": dates_past,
            "brand_id": ["B1"] * n,
            "sku_id": ["B1_1"] * n,
            "quantity": [5.0] * n,
            "promotion": [0] * n,
        }
    )

    df_future = pd.concat(
        [
            df_base,
            pd.DataFrame(
                {
                    "date": ["2014-01-16"],
                    "brand_id": ["B1"],
                    "sku_id": ["B1_1"],
                    "quantity": [1_000_000.0],  # Future sentinel
                    "promotion": [1],
                }
            ),
        ],
        ignore_index=True,
    )

    feat_base = build_forecasting_feature_table(df_base)
    feat_future = build_forecasting_feature_table(df_future)

    # All rows through 2014-01-15 must have exactly identical features
    cols_to_check = [c for c in CANONICAL_FEATURE_COLUMNS if c != "target_quantity"]
    sub_base = feat_base[cols_to_check]
    sub_future = feat_future[feat_future["date"] <= "2014-01-15"][cols_to_check]

    pd.testing.assert_frame_equal(sub_base, sub_future)


def test_leakage_5_promotion_leakage():
    """Verify historical promotion features exclude target date promotion."""
    df = pd.DataFrame(
        {
            "date": ["2014-01-02", "2014-01-03", "2014-01-04"],
            "brand_id": ["B1"] * 3,
            "sku_id": ["B1_1"] * 3,
            "quantity": [10.0, 10.0, 10.0],
            "promotion": [0, 0, 1],  # Promotion active only on target date (Jan 4)
        }
    )
    features = build_forecasting_feature_table(df)

    row_jan4 = features[features["date"] == "2014-01-04"].iloc[0]
    # lag_1_promotion must be Jan 3's promotion (0)
    assert row_jan4["lag_1_promotion"] == 0.0
    # promotion_rate_28 over [Jan 2, Jan 3] must be 0.0
    assert row_jan4["promotion_rate_28"] == 0.0
    # target_promotion must be Jan 4's promotion (1)
    assert row_jan4["target_promotion"] == 1


def test_leakage_6_cutoff_drops_2018_before_features():
    """Verify that rows beyond cutoff_date are removed before constructing features."""
    df = pd.DataFrame(
        {
            "date": ["2017-12-30", "2017-12-31", "2018-01-01", "2018-01-02"],
            "brand_id": ["B1"] * 4,
            "sku_id": ["B1_1"] * 4,
            "quantity": [10.0, 20.0, 30.0, 40.0],
            "promotion": [0, 0, 1, 1],
        }
    )
    features = build_forecasting_feature_table(df, cutoff_date="2017-12-31")

    assert features["date"].max() == "2017-12-31"
    assert len(features) == 2
    assert "2018-01-01" not in features["date"].values
    assert "2018-01-02" not in features["date"].values


def test_structural_7_key_uniqueness(synthetic_sku_data):
    """Verify (date, sku_id) pairs are unique."""
    features = build_forecasting_feature_table(synthetic_sku_data)
    duplicates = features.duplicated(subset=["date", "sku_id"])
    assert not duplicates.any()


def test_structural_8_split_assignment():
    """Verify 2014-2016 -> train, 2017 -> validation, with no other values."""
    dates = ["2014-01-02", "2016-12-31", "2017-01-01", "2017-12-31"]
    df = pd.DataFrame(
        {
            "date": dates,
            "brand_id": ["B1"] * 4,
            "sku_id": ["B1_1"] * 4,
            "quantity": [1.0, 2.0, 3.0, 4.0],
            "promotion": [0, 0, 0, 0],
        }
    )
    features = build_forecasting_feature_table(df)

    assert features.loc[features["date"] == "2014-01-02", "split"].values[0] == "train"
    assert features.loc[features["date"] == "2016-12-31", "split"].values[0] == "train"
    assert features.loc[features["date"] == "2017-01-01", "split"].values[0] == "validation"
    assert features.loc[features["date"] == "2017-12-31", "split"].values[0] == "validation"
    assert set(features["split"].unique()) == {"train", "validation"}


def test_structural_9_determinism(synthetic_sku_data):
    """Verify two runs produce bit-for-bit identical feature data."""
    f1 = build_forecasting_feature_table(synthetic_sku_data)
    f2 = build_forecasting_feature_table(synthetic_sku_data)
    pd.testing.assert_frame_equal(f1, f2)


def test_real_data_feature_dataset_integration():
    """Verify feature dataset generated from actual canonical data satisfies all requirements."""
    df_features = load_forecasting_features("data/processed/forecasting_features_development.csv")

    assert df_features.shape == (170274, 31)
    assert list(df_features.columns) == CANONICAL_FEATURE_COLUMNS
    assert df_features["date"].min() == "2014-01-02"
    assert df_features["date"].max() == DEVELOPMENT_CUTOFF_DATE
    assert df_features["sku_id"].nunique() == 118
    assert df_features["brand_id"].nunique() == 4
    assert (df_features["split"] == "train").sum() == 127558
    assert (df_features["split"] == "validation").sum() == 42716

    # Verify no 2018 dates
    assert (df_features["year"] == 2018).sum() == 0

    # Verify non-negative quantities
    assert (df_features["target_quantity"] < 0).sum() == 0

    # Verify binary target promotion
    assert set(df_features["target_promotion"].unique()).issubset({0, 1})
