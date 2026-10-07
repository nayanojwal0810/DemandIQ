"""Unit, temporal safety, leakage prevention, and output contract tests for Stage 9.

Tests cover:
1. Fold definitions & temporal hierarchy (strict chronological ordering, no 2017/2018).
2. Frozen Stage 8 model configurations (exact selected tree counts, no early stopping).
3. Feature exclusions and categorical handling.
4. Classical baseline fold semantics.
5. Metric correctness & common-valid ranking determinism.
6. Target and future-fold leakage safety.
7. Deliverable file schemas, coverage, and absence of 2018.
8. 2017 quarterly stability metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.evaluation.robustness_analysis import (
    CLASSICAL_MODELS,
    ML_MODELS,
    MODELS_ALL,
    assign_robustness_flag,
    calculate_2017_quarterly_metrics,
    calculate_brand_metrics,
    calculate_fold_metrics,
    calculate_model_robustness_summary,
    calculate_rolling_summary,
    calculate_sku_metrics,
)
from src.evaluation.rolling_validation import (
    CLASSICAL_BASELINES,
    DEFAULT_FROZEN_TREE_COUNTS,
    ML_CONFIG_SPECS,
    ROLLING_FOLDS,
    ROLLING_PREDICTION_COLUMNS,
    RollingFold,
    load_development_feature_table,
    load_frozen_tree_counts,
    train_and_predict_fold_ml_models,
)
from src.features.feature_config import (
    FEATURE_COLUMNS_NO_TARGET_PROMO,
    FEATURE_COLUMNS_WITH_TARGET_PROMO,
)
from src.models.tree_models import (
    ModelConfig,
    build_lightgbm_estimator,
    build_xgboost_estimator,
    get_model_feature_columns,
)


# =====================================================================
# 1. FOLD DEFINITIONS & TEMPORAL INTEGRITY
# =====================================================================

def test_eight_folds_exact_boundaries():
    """Verify all 8 rolling folds conform exactly to approved quarterly date boundaries."""
    assert len(ROLLING_FOLDS) == 8

    expected_boundaries = [
        (1, "2014-01-02", "2014-12-31", "2015-01-01", "2015-03-31"),
        (2, "2014-01-02", "2015-03-31", "2015-04-01", "2015-06-30"),
        (3, "2014-01-02", "2015-06-30", "2015-07-01", "2015-09-30"),
        (4, "2014-01-02", "2015-09-30", "2015-10-01", "2015-12-31"),
        (5, "2014-01-02", "2015-12-31", "2016-01-01", "2016-03-31"),
        (6, "2014-01-02", "2016-03-31", "2016-04-01", "2016-06-30"),
        (7, "2014-01-02", "2016-06-30", "2016-07-01", "2016-09-30"),
        (8, "2014-01-02", "2016-09-30", "2016-10-01", "2016-12-31"),
    ]

    for fold, (f_id, tr_s, tr_e, val_s, val_e) in zip(ROLLING_FOLDS, expected_boundaries):
        assert fold.fold_id == f_id
        assert fold.train_start == tr_s
        assert fold.train_end == tr_e
        assert fold.val_start == val_s
        assert fold.val_end == val_e


def test_fold_chronological_ordering_and_no_overlap():
    """Verify each fold's validation strictly succeeds training and validation windows do not overlap."""
    prev_val_end = None
    for fold in ROLLING_FOLDS:
        # Strict temporal ordering
        assert fold.train_start <= fold.train_end
        assert fold.train_end < fold.val_start
        assert fold.val_start <= fold.val_end

        # No 2017 or 2018 in rolling folds
        assert fold.val_end < "2017-01-01"
        assert fold.train_end < "2017-01-01"

        # Sequential non-overlapping validation quarters
        if prev_val_end is not None:
            assert fold.val_start > prev_val_end
        prev_val_end = fold.val_end


# =====================================================================
# 2. FROZEN STAGE 8 MODEL CONFIGURATION
# =====================================================================

def test_frozen_tree_counts_match_stage_8():
    """Verify tree counts match Stage 8 inner early-stopping refit selected_n_estimators."""
    counts = load_frozen_tree_counts()
    assert counts["xgboost_no_target_promo"] == 167
    assert counts["xgboost_with_target_promo"] == 231
    assert counts["lightgbm_no_target_promo"] == 124
    assert counts["lightgbm_with_target_promo"] == 162


def test_no_early_stopping_applied_in_rolling_fold_models():
    """Verify built estimators do not have early_stopping_rounds configured during rolling robustness."""
    counts = load_frozen_tree_counts()
    for cfg_name, (family, variant) in ML_CONFIG_SPECS.items():
        cfg = ModelConfig(model_family=family, feature_variant=variant)
        trees = counts[cfg_name]
        if family == "xgboost":
            model = build_xgboost_estimator(cfg, n_estimators=trees)
            assert model.early_stopping_rounds is None
            assert model.n_estimators == trees
        elif family == "lightgbm":
            model = build_lightgbm_estimator(cfg, n_estimators=trees)
            assert model.n_estimators == trees


def test_all_four_ml_configurations_present():
    """Verify exactly the four approved ML candidates are defined."""
    expected = {
        "xgboost_no_target_promo",
        "xgboost_with_target_promo",
        "lightgbm_no_target_promo",
        "lightgbm_with_target_promo",
    }
    assert set(ML_CONFIG_SPECS.keys()) == expected


# =====================================================================
# 3. FEATURE SPECIFICATIONS & SAFETY
# =====================================================================

def test_forbidden_columns_excluded():
    """Verify target_quantity, date, and split are excluded from feature matrices."""
    for variant in ["no_target_promo", "with_target_promo"]:
        cols = get_model_feature_columns(variant)
        assert "target_quantity" not in cols
        assert "date" not in cols
        assert "split" not in cols


def test_promotion_feature_isolation():
    """Verify target_promotion appears exclusively in with_target_promo variant."""
    no_promo = get_model_feature_columns("no_target_promo")
    with_promo = get_model_feature_columns("with_target_promo")

    assert "target_promotion" not in no_promo
    assert "target_promotion" in with_promo
    assert set(with_promo) - set(no_promo) == {"target_promotion"}


def test_categorical_identities_retained():
    """Verify brand_id and sku_id are present in both variants."""
    for variant in ["no_target_promo", "with_target_promo"]:
        cols = get_model_feature_columns(variant)
        assert "brand_id" in cols
        assert "sku_id" in cols


# =====================================================================
# 4. METRIC DEFINITIONS & COMMON-VALID RANKING
# =====================================================================

def test_wape_mae_rmse_calculation():
    """Verify core error metrics match exact definitions."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    # abs errors: 2, 2, 3 -> sum = 7, actual sum = 60
    assert calculate_wape(y_true, y_pred) == pytest.approx(7.0 / 60.0)
    assert calculate_mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)
    assert calculate_rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3.0))


def test_common_valid_coverage_and_ranking():
    """Verify common_valid scope enforces 100% intersection across all 9 models."""
    synthetic_preds = pd.DataFrame(
        {
            "fold": [1, 1, 1, 1],
            "date": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
            "brand_id": ["B1", "B1", "B1", "B1"],
            "sku_id": ["SKU_1", "SKU_1", "SKU_1", "SKU_1"],
            "actual": [10.0, 20.0, 30.0, 40.0],
            "naive": [10.0, 20.0, 30.0, 40.0],
            "seasonal_naive_7": [np.nan, 20.0, 30.0, 40.0],  # 1 missing
            "moving_average_7d": [10.0, 20.0, 30.0, 40.0],
            "ets_ses": [10.0, 20.0, 30.0, 40.0],
            "croston_sba": [10.0, 20.0, 30.0, 40.0],
            "xgboost_no_target_promo": [10.0, 20.0, 30.0, 40.0],
            "xgboost_with_target_promo": [10.0, 20.0, 30.0, 40.0],
            "lightgbm_no_target_promo": [10.0, 20.0, 30.0, 40.0],
            "lightgbm_with_target_promo": [10.0, 20.0, 30.0, 40.0],
        }
    )

    metrics_df = calculate_fold_metrics(synthetic_preds)
    common_metrics = metrics_df[metrics_df["evaluation_scope"] == "common_valid"]

    # All 9 models have exactly 3 valid targets in common_valid (since row 0 has NaN seasonal)
    for model in synthetic_preds.columns[5:]:
        m_row = common_metrics[common_metrics["model"] == model].iloc[0]
        assert m_row["valid_prediction_count"] == 3
        assert m_row["total_validation_targets"] == 4
        assert m_row["coverage"] == 0.75


def test_deterministic_robustness_flags():
    """Verify robustness flags map cleanly according to predefined rules."""
    # frequent_fold_win
    assert assign_robustness_flag(fold_wins=5, mean_rank=2.0, std_fold_wape=0.03, median_std_wape=0.03) == "frequent_fold_win"
    # high_variability
    assert assign_robustness_flag(fold_wins=1, mean_rank=3.0, std_fold_wape=0.08, median_std_wape=0.04) == "high_variability"
    # frequent_fold_loss
    assert assign_robustness_flag(fold_wins=0, mean_rank=7.5, std_fold_wape=0.02, median_std_wape=0.03) == "frequent_fold_loss"
    # stable_wape
    assert assign_robustness_flag(fold_wins=2, mean_rank=3.5, std_fold_wape=0.03, median_std_wape=0.03) == "stable_wape"


# =====================================================================
# 5. LEAKAGE PREVENTION TESTS
# =====================================================================

def test_validation_actual_does_not_affect_own_ml_prediction():
    """Verify mutating a validation target actual does not alter that observation's forecast."""
    df_feat = load_development_feature_table()
    # Select small contiguous slice of real features
    df_tr = df_feat.iloc[:120].copy()
    df_val = df_feat.iloc[120:130].copy()

    counts = {"xgboost_no_target_promo": 5, "xgboost_with_target_promo": 5, "lightgbm_no_target_promo": 5, "lightgbm_with_target_promo": 5}
    p1 = train_and_predict_fold_ml_models(df_tr, df_val, counts)

    # Mutate validation actual
    df_val_mutated = df_val.copy()
    df_val_mutated["target_quantity"] = 999999.0
    p2 = train_and_predict_fold_ml_models(df_tr, df_val_mutated, counts)

    for k in p1:
        np.testing.assert_array_almost_equal(p1[k], p2[k], decimal=6)


def test_training_fold_does_not_contain_validation_targets():
    """Verify no training slice contains timestamps belonging to the validation period."""
    for fold in ROLLING_FOLDS:
        assert fold.train_end < fold.val_start


# =====================================================================
# 6. OUTPUT DELIVERABLES & INTEGRITY TESTS
# =====================================================================

def test_rolling_validation_predictions_file_integrity():
    """Verify rolling_validation_predictions.csv adheres strictly to deliverable specifications."""
    path = Path("data/processed/rolling_validation_predictions.csv")
    if not path.exists():
        pytest.skip("rolling_validation_predictions.csv not yet generated.")

    df = pd.read_csv(path)

    # Schema check
    assert list(df.columns) == ROLLING_PREDICTION_COLUMNS
    assert len(df) == 85078
    assert df["fold"].nunique() == 8
    assert df["sku_id"].nunique() == 118

    # Strictly pre-2017, no 2018
    assert (df["date"] <= "2016-12-31").all()
    assert not (df["date"] >= "2017-01-01").any()
    assert not (df["date"] >= "2018-01-01").any()

    # No duplicate (fold, date, sku_id)
    assert df.duplicated(subset=["fold", "date", "sku_id"]).sum() == 0

    # ML models have 0 NaNs, 0 Infs, and no negative forecasts
    for m in ML_MODELS:
        assert not df[m].isna().any(), f"{m} contains NaNs"
        assert not np.isinf(df[m]).any(), f"{m} contains Infs"
        assert (df[m] >= 0.0).all(), f"{m} contains negative predictions"


def test_rolling_validation_metrics_file_integrity():
    """Verify rolling_validation_metrics.csv covers all folds, models, and scopes."""
    path = Path("data/processed/rolling_validation_metrics.csv")
    if not path.exists():
        pytest.skip("rolling_validation_metrics.csv not yet generated.")

    df = pd.read_csv(path)
    assert set(df["evaluation_scope"]) == {"model_valid", "common_valid"}
    assert df["fold"].nunique() == 8
    assert df["model"].nunique() == 9
    assert len(df) == 8 * 9 * 2
    assert (df["wape"] > 0.0).all()


def test_rolling_validation_summary_file_integrity():
    """Verify rolling_validation_summary.csv reconciles fold wins to 8 folds."""
    path = Path("data/processed/rolling_validation_summary.csv")
    if not path.exists():
        pytest.skip("rolling_validation_summary.csv not yet generated.")

    df = pd.read_csv(path)
    assert len(df) == 9
    assert df["fold_wins"].sum() == 8
    assert df["fold_win_share"].sum() == pytest.approx(1.0)
    assert (df["pooled_wape"] > 0.0).all()


def test_rolling_validation_brand_metrics_integrity():
    """Verify rolling_validation_brand_metrics.csv covers all 4 brands."""
    path = Path("data/processed/rolling_validation_brand_metrics.csv")
    if not path.exists():
        pytest.skip("rolling_validation_brand_metrics.csv not yet generated.")

    df = pd.read_csv(path)
    assert set(df["brand_id"]) == {"B1", "B2", "B3", "B4"}
    assert len(df) == 4 * 9
    assert (df["wape"] > 0.0).all()


def test_rolling_validation_sku_metrics_integrity():
    """Verify rolling_validation_sku_metrics.csv covers all 118 SKUs across 9 models."""
    path = Path("data/processed/rolling_validation_sku_metrics.csv")
    if not path.exists():
        pytest.skip("rolling_validation_sku_metrics.csv not yet generated.")

    df = pd.read_csv(path)
    assert df["sku_id"].nunique() == 118
    assert len(df) == 118 * 9
    assert (df["eligible_fold_count"] > 0).all()


def test_model_robustness_summary_integrity():
    """Verify model_robustness_summary.csv contains valid robustness flags."""
    path = Path("data/processed/model_robustness_summary.csv")
    if not path.exists():
        pytest.skip("model_robustness_summary.csv not yet generated.")

    df = pd.read_csv(path)
    assert len(df) == 9
    allowed_flags = {"frequent_fold_win", "high_variability", "frequent_fold_loss", "stable_wape"}
    assert set(df["robustness_flag"]).issubset(allowed_flags)


def test_ml_2017_quarterly_metrics_integrity():
    """Verify ml_2017_quarterly_metrics.csv has all four quarters without retraining."""
    path = Path("data/processed/ml_2017_quarterly_metrics.csv")
    if not path.exists():
        pytest.skip("ml_2017_quarterly_metrics.csv not yet generated.")

    df = pd.read_csv(path)
    assert set(df["quarter"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert (df["wape"] > 0.0).all()
    assert (df["valid_prediction_count"] > 0).all()
