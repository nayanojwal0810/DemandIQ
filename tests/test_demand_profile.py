"""Unit and integration tests for demand analytics and profiling."""

import pandas as pd
import pytest

from src.business.demand_profile import (
    filter_development_period,
    calculate_portfolio_metrics,
    calculate_sku_profiles,
    calculate_brand_summary,
    calculate_weekly_analysis,
    calculate_calendar_seasonality,
    calculate_promotion_summary,
    generate_demand_analytics,
)


@pytest.fixture
def multi_year_canonical_df():
    """Create multi-year canonical DataFrame fixture spanning 2016-2018."""
    records = []
    # 2017 observations (development)
    records.append({"date": "2017-06-01", "brand_id": "B1", "sku_id": "B1_1", "quantity": 10, "promotion": 1})
    records.append({"date": "2017-06-01", "brand_id": "B1", "sku_id": "B1_2", "quantity": 0, "promotion": 0})
    records.append({"date": "2017-06-02", "brand_id": "B1", "sku_id": "B1_1", "quantity": 20, "promotion": 0})
    records.append({"date": "2017-06-02", "brand_id": "B1", "sku_id": "B1_2", "quantity": 5, "promotion": 0})

    # 2018 observations (holdout / sealed)
    records.append({"date": "2018-01-02", "brand_id": "B1", "sku_id": "B1_1", "quantity": 999, "promotion": 1})
    records.append({"date": "2018-01-02", "brand_id": "B1", "sku_id": "B1_2", "quantity": 999, "promotion": 1})

    return pd.DataFrame(records)


def test_development_cutoff_behavior(multi_year_canonical_df):
    """Test that cutoff strictly excludes 2018 holdout observations."""
    dev_df = filter_development_period(multi_year_canonical_df, cutoff_date="2017-12-31")

    assert dev_df["date"].max() == "2017-06-02"
    assert len(dev_df) == 4
    # Ensure 2018 values (999) never leak into development dataset
    assert dev_df["quantity"].max() == 20


def test_calculate_portfolio_metrics(multi_year_canonical_df):
    """Test portfolio-level demand metrics."""
    dev_df = filter_development_period(multi_year_canonical_df, cutoff_date="2017-12-31")
    metrics = calculate_portfolio_metrics(dev_df)

    assert metrics["observation_days"] == 2
    # Day 1 total = 10, Day 2 total = 25 -> Total = 35
    assert metrics["total_demand"] == 35
    assert metrics["mean_daily_demand"] == 17.5
    assert metrics["min_daily_demand"] == 10
    assert metrics["max_daily_demand"] == 25


def test_calculate_sku_profiles(multi_year_canonical_df):
    """Test SKU demand profiles calculation."""
    dev_df = filter_development_period(multi_year_canonical_df, cutoff_date="2017-12-31")
    profiles = calculate_sku_profiles(dev_df)

    assert len(profiles) == 2
    assert list(profiles["sku_id"]) == ["B1_1", "B1_2"]

    b1_1 = profiles[profiles["sku_id"] == "B1_1"].iloc[0]
    assert b1_1["total_demand"] == 30
    assert b1_1["mean_daily_demand"] == 15.0
    assert b1_1["zero_demand_rate"] == 0.0

    b1_2 = profiles[profiles["sku_id"] == "B1_2"].iloc[0]
    assert b1_2["total_demand"] == 5
    assert b1_2["mean_daily_demand"] == 2.5
    assert b1_2["zero_demand_rate"] == 0.5


def test_calculate_brand_summary(multi_year_canonical_df):
    """Test brand-level demand summary."""
    dev_df = filter_development_period(multi_year_canonical_df, cutoff_date="2017-12-31")
    profiles = calculate_sku_profiles(dev_df)
    brand_sum = calculate_brand_summary(dev_df, profiles)

    assert len(brand_sum) == 1
    assert brand_sum.iloc[0]["brand_id"] == "B1"
    assert brand_sum.iloc[0]["sku_count"] == 2
    assert brand_sum.iloc[0]["total_demand"] == 35
    assert brand_sum.iloc[0]["portfolio_demand_share"] == 1.0


def test_calculate_promotion_summary(multi_year_canonical_df):
    """Test promotion summary calculation."""
    dev_df = filter_development_period(multi_year_canonical_df, cutoff_date="2017-12-31")
    promo_sum = calculate_promotion_summary(dev_df)

    assert len(promo_sum) == 2
    b1_1_promo = promo_sum[promo_sum["sku_id"] == "B1_1"].iloc[0]
    assert b1_1_promo["promo_obs"] == 1
    assert b1_1_promo["promo_rate"] == 0.5
    assert b1_1_promo["mean_promo_demand"] == 10.0
    assert b1_1_promo["mean_non_promo_demand"] == 20.0
