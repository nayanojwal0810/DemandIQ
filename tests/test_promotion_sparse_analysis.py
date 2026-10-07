"""Unit, temporal safety, join integrity, and output contract tests for promotion and sparse-demand analysis."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.business.promotion_sparse_analysis import (
    CLASSICAL_MODELS,
    ML_MODELS,
    MODELS_ALL,
    SPARSITY_BUCKETS,
    assign_sparsity_bucket,
    compute_training_sparsity_map,
    load_and_prepare_analysis_datasets,
)
from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.evaluation.rolling_validation import ROLLING_FOLDS


# =====================================================================
# 1. TEMPORAL SAFETY & LEAKAGE TESTS
# =====================================================================

def test_sparse_classification_uses_training_window_only():
    """Verify sparsity zero-demand rate is estimated strictly before the validation window."""
    # Build synthetic features with distinct pre/post cutoff behavior
    dates = pd.date_range("2014-01-02", "2015-03-31", freq="D")
    n = len(dates)
    df_feat = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sku_id": ["SKU_TEST"] * n,
            "target_quantity": [0.0 if d <= pd.Timestamp("2014-12-31") else 50.0 for d in dates],
        }
    )

    # In Fold 1 train (2014-01-02 to 2014-12-31), demand is 100% zero -> very_high_sparsity
    fold1 = ROLLING_FOLDS[0]
    sparse_map = compute_training_sparsity_map(df_feat, fold1.train_start, fold1.train_end)
    assert sparse_map["SKU_TEST"] == "very_high_sparsity"

    # Validation actuals (all 50.0) never alter the training-only classification
    assert fold1.train_end < fold1.val_start


def test_2017_sparse_classification_ends_at_2016():
    """Verify 2017 sparsity calculation period strictly stops on 2016-12-31."""
    dates = pd.date_range("2014-01-02", "2017-12-31", freq="D")
    n = len(dates)
    df_feat = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sku_id": ["SKU_A"] * n,
            # Zero demand in 2014-2016 (rate = 1.0), positive in 2017
            "target_quantity": [0.0 if d <= pd.Timestamp("2016-12-31") else 100.0 for d in dates],
        }
    )

    sparse_map_2017 = compute_training_sparsity_map(df_feat, "2014-01-02", "2016-12-31")
    assert sparse_map_2017["SKU_A"] == "very_high_sparsity"


def test_no_2018_in_analysis_inputs():
    """Verify 2018 holdout observations are completely absent from all inputs."""
    df_rolling, df_2017, df_feat = load_and_prepare_analysis_datasets()
    for name, df in [("rolling", df_rolling), ("2017", df_2017), ("features", df_feat)]:
        assert not (df["date"] >= "2018-01-01").any(), f"Found 2018 in {name}"


# =====================================================================
# 2. PROMOTION JOIN & INTEGRITY TESTS
# =====================================================================

def test_promotion_join_is_strictly_one_to_one():
    """Verify target_promotion join does not expand or drop prediction rows."""
    df_rolling, df_2017, _ = load_and_prepare_analysis_datasets()

    # Original rolling rows: 85,078; 2017 rows: 42,716
    assert len(df_rolling) == 85078
    assert len(df_2017) == 42716

    # Keys remain unique
    assert not df_rolling.duplicated(subset=["fold", "date", "sku_id"]).any()
    assert not df_2017.duplicated(subset=["date", "sku_id"]).any()


def test_target_promotion_values_are_binary():
    """Verify promotion status contains strictly binary 0 and 1 values."""
    df_rolling, df_2017, _ = load_and_prepare_analysis_datasets()
    assert set(df_rolling["target_promotion"].unique()) == {0, 1}
    assert set(df_2017["target_promotion"].unique()) == {0, 1}


# =====================================================================
# 3. SPARSITY BUCKET INTEGRITY TESTS
# =====================================================================

def test_deterministic_sparsity_bucket_thresholds():
    """Verify exact bucket classification at and across boundary values."""
    # Low sparsity: < 0.10
    assert assign_sparsity_bucket(0.0) == "low_sparsity"
    assert assign_sparsity_bucket(0.05) == "low_sparsity"
    assert assign_sparsity_bucket(0.0999) == "low_sparsity"

    # Moderate sparsity: >= 0.10 and < 0.25
    assert assign_sparsity_bucket(0.10) == "moderate_sparsity"
    assert assign_sparsity_bucket(0.18) == "moderate_sparsity"
    assert assign_sparsity_bucket(0.2499) == "moderate_sparsity"

    # High sparsity: >= 0.25 and < 0.50
    assert assign_sparsity_bucket(0.25) == "high_sparsity"
    assert assign_sparsity_bucket(0.35) == "high_sparsity"
    assert assign_sparsity_bucket(0.4999) == "high_sparsity"

    # Very high sparsity: >= 0.50
    assert assign_sparsity_bucket(0.50) == "very_high_sparsity"
    assert assign_sparsity_bucket(0.75) == "very_high_sparsity"
    assert assign_sparsity_bucket(1.0) == "very_high_sparsity"


# =====================================================================
# 4. METRIC INTEGRITY & FULL-PRECISION WINNERS
# =====================================================================

def test_wape_mae_rmse_agreement():
    """Verify metrics calculation agrees with repository standard."""
    y_true = np.array([5.0, 10.0, 15.0])
    y_pred = np.array([6.0, 8.0, 15.0])

    assert calculate_wape(y_true, y_pred) == pytest.approx(3.0 / 30.0)
    assert calculate_mae(y_true, y_pred) == pytest.approx(3.0 / 3.0)
    assert calculate_rmse(y_true, y_pred) == pytest.approx(np.sqrt((1 + 4 + 0) / 3.0))


def test_full_precision_sku_comparison():
    """Verify SKU win decision uses full precision before rounding."""
    y_true = np.array([10.0])
    # with promo error = 1.0000001, no promo error = 1.0000002
    y_with = np.array([11.0000001])
    y_no = np.array([11.0000002])

    w_with = calculate_wape(y_true, y_with)
    w_no = calculate_wape(y_true, y_no)

    assert w_with < w_no  # with promo wins at full precision


# =====================================================================
# 5. OUTPUT DELIVERABLES & INTEGRITY TESTS
# =====================================================================

def test_promotion_condition_metrics_integrity():
    """Verify promotion_condition_metrics.csv exists and matches expected schema."""
    path = Path("data/processed/promotion_condition_metrics.csv")
    if not path.exists():
        pytest.skip("promotion_condition_metrics.csv not yet generated.")

    df = pd.read_csv(path)
    expected_cols = [
        "evaluation_period",
        "target_promotion",
        "promotion_condition",
        "model",
        "target_count",
        "target_share",
        "actual_sum",
        "demand_share",
        "wape",
        "mae",
        "rmse",
    ]
    assert list(df.columns) == expected_cols
    assert set(df["target_promotion"]) == {0, 1}
    assert set(df["promotion_condition"]) == {"non_promotion", "promotion"}
    assert set(df["model"]) == set(MODELS_ALL)
    assert (df["wape"] > 0.0).all()


def test_promotion_model_comparison_integrity():
    """Verify promotion_model_comparison.csv reconciles SKU totals across periods."""
    path = Path("data/processed/promotion_model_comparison.csv")
    if not path.exists():
        pytest.skip("promotion_model_comparison.csv not yet generated.")

    df = pd.read_csv(path)
    expected_cols = [
        "evaluation_period",
        "model_family",
        "wape_with_promo",
        "wape_no_promo",
        "abs_wape_diff",
        "rel_wape_diff",
        "skus_improved",
        "skus_worsened",
        "skus_unchanged",
        "total_skus_evaluated",
    ]
    assert list(df.columns) == expected_cols
    assert set(df["model_family"]) == {"xgboost", "lightgbm"}

    for _, r in df.iterrows():
        assert r["skus_improved"] + r["skus_worsened"] + r["skus_unchanged"] == r["total_skus_evaluated"]
        assert r["total_skus_evaluated"] == 118


def test_sparse_demand_metrics_integrity():
    """Verify sparse_demand_metrics.csv covers all four buckets across all models."""
    path = Path("data/processed/sparse_demand_metrics.csv")
    if not path.exists():
        pytest.skip("sparse_demand_metrics.csv not yet generated.")

    df = pd.read_csv(path)
    assert set(df["sparsity_bucket"]) == set(SPARSITY_BUCKETS)
    assert set(df["model"]) == set(MODELS_ALL)
    assert (df["wape"] > 0.0).all()
    assert (df["sku_count"] > 0).all()


def test_sparse_demand_summary_integrity():
    """Verify sparse_demand_summary.csv contains Croston SBA and tree model benchmarks."""
    path = Path("data/processed/sparse_demand_summary.csv")
    if not path.exists():
        pytest.skip("sparse_demand_summary.csv not yet generated.")

    df = pd.read_csv(path)
    assert "croston_wape" in df.columns
    assert "xgboost_with_promo_wape" in df.columns
    assert "lightgbm_with_promo_wape" in df.columns
    assert "best_model" in df.columns
    assert (df["croston_wape"] > 0.0).all()


def test_promotion_sparse_metrics_integrity():
    """Verify promotion_sparse_metrics.csv contains cross-tabulation without missing cells."""
    path = Path("data/processed/promotion_sparse_metrics.csv")
    if not path.exists():
        pytest.skip("promotion_sparse_metrics.csv not yet generated.")

    df = pd.read_csv(path)
    assert set(df["target_promotion"]) == {0, 1}
    assert set(df["sparsity_bucket"]) == set(SPARSITY_BUCKETS)
    assert len(df) == 2 * 2 * 4  # 2 periods x 2 promo conditions x 4 sparsity buckets = 16
    assert (df["xgboost_with_promo_wape"] > 0.0).all()
