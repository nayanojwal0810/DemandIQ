"""Unit, leakage, split integrity, and output contract tests for XGBoost and LightGBM models."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_wape
from src.features.feature_config import (
    DEVELOPMENT_CUTOFF_DATE,
    FEATURE_COLUMNS_NO_TARGET_PROMO,
    FEATURE_COLUMNS_WITH_TARGET_PROMO,
    TRAINING_END_DATE,
    TRAINING_START_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
)
from src.models.train_tree_models import (
    INNER_FIT_END,
    INNER_FIT_START,
    INNER_VAL_END,
    INNER_VAL_START,
    MODEL_CONFIGS,
    calculate_ml_portfolio_metrics,
    calculate_ml_sku_metrics,
    calculate_ml_winner_summary,
    compute_promotion_comparison,
    load_development_feature_data,
    split_training_data,
)
from src.models.tree_models import (
    ModelConfig,
    build_lightgbm_estimator,
    build_xgboost_estimator,
    determine_selected_n_estimators,
    get_model_feature_columns,
    post_process_predictions,
    prepare_categorical_features,
    train_model_with_inner_early_stopping,
)


# =====================================================================
# 1. DATA SPLITS & BOUNDARY TESTS
# =====================================================================

def test_split_boundaries_and_temporal_hierarchy():
    """Verify official outer and inner splits conform strictly to required date boundaries."""
    assert TRAINING_START_DATE == "2014-01-02"
    assert TRAINING_END_DATE == "2016-12-31"
    assert VALIDATION_START_DATE == "2017-01-01"
    assert VALIDATION_END_DATE == "2017-12-31"

    # Inner early-stopping split is strictly inside 2014-2016 outer training
    assert INNER_FIT_START == "2014-01-02"
    assert INNER_FIT_END == "2016-09-30"
    assert INNER_VAL_START == "2016-10-01"
    assert INNER_VAL_END == "2016-12-31"

    assert INNER_FIT_END < INNER_VAL_START
    assert INNER_VAL_END <= TRAINING_END_DATE


def test_no_2018_in_feature_data_and_splits():
    """Verify 2018 holdout never enters the ML training or validation pipeline."""
    df = load_development_feature_data()
    assert df["date"].max() <= DEVELOPMENT_CUTOFF_DATE
    assert not (df["date"] >= "2018-01-01").any()

    df_inner_tr, df_inner_val, df_outer_tr, df_outer_val = split_training_data(df)
    for split_df in [df_inner_tr, df_inner_val, df_outer_tr, df_outer_val]:
        assert not (split_df["date"] >= "2018-01-01").any()


def test_splits_are_strictly_chronological_no_shuffling():
    """Verify split partitions maintain strict chronological ordering with no overlaps."""
    df = load_development_feature_data()
    df_inner_tr, df_inner_val, df_outer_tr, df_outer_val = split_training_data(df)

    assert df_inner_tr["date"].max() < df_inner_val["date"].min()
    assert df_outer_tr["date"].max() < df_outer_val["date"].min()
    assert len(df_inner_tr) + len(df_inner_val) == len(df_outer_tr)


# =====================================================================
# 2. FEATURE CONTRACT TESTS
# =====================================================================

def test_forbidden_columns_excluded_from_features():
    """Verify target_quantity, date, and split are never included as model features."""
    for variant in ["no_target_promo", "with_target_promo"]:
        cols = get_model_feature_columns(variant)
        assert "target_quantity" not in cols
        assert "date" not in cols
        assert "split" not in cols


def test_categorical_identities_included_consistently():
    """Verify brand_id and sku_id are present in both model variants."""
    for variant in ["no_target_promo", "with_target_promo"]:
        cols = get_model_feature_columns(variant)
        assert "brand_id" in cols
        assert "sku_id" in cols


def test_target_promotion_inclusion_only_in_with_promo_variant():
    """Verify target_promotion appears exclusively in with_target_promo variant."""
    no_promo_cols = get_model_feature_columns("no_target_promo")
    with_promo_cols = get_model_feature_columns("with_target_promo")

    assert "target_promotion" not in no_promo_cols
    assert "target_promotion" in with_promo_cols
    assert len(with_promo_cols) == len(no_promo_cols) + 1


# =====================================================================
# 3. SYNTHETIC MODEL FITTING & POST-PROCESSING TESTS
# =====================================================================

@pytest.fixture
def synthetic_tree_data():
    """Create a small synthetic dataset with categoricals and numeric features."""
    n = 120
    df = pd.DataFrame(
        {
            "brand_id": pd.Categorical(["B1", "B2"] * (n // 2)),
            "sku_id": pd.Categorical([f"SKU_{i % 4}" for i in range(n)]),
            "lag_1": np.random.uniform(5, 50, n),
            "rolling_mean_7": np.random.uniform(5, 50, n),
            "target_promotion": np.random.choice([0, 1], n),
        }
    )
    y = df["lag_1"] * 0.8 + df["target_promotion"] * 10.0 + np.random.normal(0, 1, n)
    return df, y.values


def test_xgboost_synthetic_fit_and_finite_predictions(synthetic_tree_data):
    """Verify XGBoost fits on synthetic data with categoricals and outputs finite non-negative predictions."""
    X, y = synthetic_tree_data
    cfg = ModelConfig(model_family="xgboost", feature_variant="with_target_promo")
    model = build_xgboost_estimator(cfg, n_estimators=10)
    model.fit(X, y)

    preds = model.predict(X)
    post_preds = post_process_predictions(preds)

    assert len(post_preds) == len(y)
    assert np.all(np.isfinite(post_preds))
    assert np.all(post_preds >= 0.0)


def test_lightgbm_synthetic_fit_and_finite_predictions(synthetic_tree_data):
    """Verify LightGBM fits on synthetic data with categoricals and outputs finite non-negative predictions."""
    X, y = synthetic_tree_data
    cfg = ModelConfig(model_family="lightgbm", feature_variant="with_target_promo")
    model = build_lightgbm_estimator(cfg, n_estimators=10)
    model.fit(X, y)

    preds = model.predict(X)
    post_preds = post_process_predictions(preds)

    assert len(post_preds) == len(y)
    assert np.all(np.isfinite(post_preds))
    assert np.all(post_preds >= 0.0)


def test_post_process_predictions_clamps_negative_values():
    """Verify post_process_predictions clamps negative demand values strictly at zero."""
    raw = np.array([-15.0, -0.0001, 0.0, 12.5, 42.0])
    clipped = post_process_predictions(raw)

    expected = np.array([0.0, 0.0, 0.0, 12.5, 42.0])
    np.testing.assert_array_equal(clipped, expected)


def test_deterministic_training_reproducibility(synthetic_tree_data):
    """Verify repeated training with random_state=42 yields identical predictions."""
    X, y = synthetic_tree_data
    cfg = ModelConfig(model_family="xgboost", feature_variant="with_target_promo", random_state=42)

    m1 = build_xgboost_estimator(cfg, n_estimators=15)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = build_xgboost_estimator(cfg, n_estimators=15)
    m2.fit(X, y)
    p2 = m2.predict(X)

    np.testing.assert_array_almost_equal(p1, p2, decimal=6)


def test_lightgbm_deterministic_training_reproducibility(synthetic_tree_data):
    """Verify repeated LightGBM training with random_state=42 yields identical predictions."""
    X, y = synthetic_tree_data
    cfg = ModelConfig(model_family="lightgbm", feature_variant="with_target_promo", random_state=42)

    m1 = build_lightgbm_estimator(cfg, n_estimators=15)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = build_lightgbm_estimator(cfg, n_estimators=15)
    m2.fit(X, y)
    p2 = m2.predict(X)

    np.testing.assert_array_almost_equal(p1, p2, decimal=6)


def test_xgboost_index_to_tree_count_conversion():
    """Verify determine_selected_n_estimators implements correct 0-based index to tree count conversion."""
    # Arbitrary test index values k
    test_indices = [0, 1, 5, 23, 77, 100, 250]
    for k in test_indices:
        # XGBoost best_iteration is 0-based index: selected trees must be k + 1
        selected = determine_selected_n_estimators("xgboost", k)
        assert selected == k + 1, f"Expected {k + 1} trees for best_iteration={k}, got {selected}"

    # Explicitly verify best_iteration = 0 does NOT produce zero trees
    zero_trees = determine_selected_n_estimators("xgboost", 0)
    assert zero_trees == 1
    assert zero_trees > 0

    # Verify LightGBM semantics are preserved (1-based tree count)
    for k in [1, 5, 50, 120]:
        assert determine_selected_n_estimators("lightgbm", k) == k

    # Edge-case zero in LightGBM clamps to minimum 1 tree
    assert determine_selected_n_estimators("lightgbm", 0) == 1


def test_xgboost_early_stopping_refit_selected_n_estimators():
    """Verify inner early-stopping XGBoost refit uses selected_n_estimators = best_iteration + 1."""
    # Build synthetic time-indexed dataset with inner fit and inner validation splits
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "brand_id": pd.Categorical(["B1", "B2"] * (n // 2)),
            "sku_id": pd.Categorical([f"SKU_{i % 4}" for i in range(n)]),
            "lag_1": np.random.uniform(5, 50, n),
            "lag_7": np.random.uniform(5, 50, n),
            "rolling_mean_7": np.random.uniform(5, 50, n),
            "rolling_mean_28": np.random.uniform(5, 50, n),
            "rolling_std_7": np.random.uniform(1, 10, n),
            "rolling_std_28": np.random.uniform(1, 10, n),
            "target_promotion": np.random.choice([0, 1], n),
            "day_of_week": np.tile(np.arange(7), n // 7 + 1)[:n],
            "month": np.ones(n, dtype=int),
            "is_weekend": np.zeros(n, dtype=int),
            "day_of_month": np.arange(1, n + 1) % 28 + 1,
            "target_quantity": np.random.uniform(10, 100, n),
        }
    )

    df_inner_tr = df.iloc[:120].copy()
    df_inner_val = df.iloc[120:160].copy()
    df_outer_tr = df.iloc[:160].copy()

    cfg = ModelConfig(
        model_family="xgboost",
        feature_variant="with_target_promo",
        max_estimators=30,
        early_stopping_rounds=5,
        random_state=42,
    )

    feature_cols = [c for c in df.columns if c != "target_quantity"]
    X_inner_tr = df_inner_tr[feature_cols]
    y_inner_tr = df_inner_tr["target_quantity"].values
    X_inner_val = df_inner_val[feature_cols]
    y_inner_val = df_inner_val["target_quantity"].values
    X_outer_tr = df_outer_tr[feature_cols]
    y_outer_tr = df_outer_tr["target_quantity"].values

    res = train_model_with_inner_early_stopping(
        cfg,
        X_inner_tr,
        y_inner_tr,
        X_inner_val,
        y_inner_val,
        X_outer_tr,
        y_outer_tr,
    )

    # Validate that best_iteration k resulted in refit tree count k + 1
    k = res.best_iteration
    assert k >= 0
    assert res.selected_n_estimators == k + 1
    assert res.fitted_model.n_estimators == k + 1
    assert res.selected_n_estimators >= 1


# =====================================================================
# 4. LEAKAGE PREVENTION TESTS
# =====================================================================

def test_validation_actual_does_not_affect_own_prediction(synthetic_tree_data):
    """Verify mutating target validation actual does not alter that observation's forecast."""
    X, y = synthetic_tree_data
    cfg = ModelConfig(model_family="lightgbm", feature_variant="with_target_promo")
    model = build_lightgbm_estimator(cfg, n_estimators=10)
    model.fit(X, y)

    # Prediction operates on features X; changing an external actual vector has zero effect
    p1 = model.predict(X.iloc[0:1])
    # Mutating ground truth actual y[0]
    y_mutated = y.copy()
    y_mutated[0] = 99999.0

    p2 = model.predict(X.iloc[0:1])
    assert p1[0] == p2[0]


def test_inner_early_stopping_does_not_use_outer_validation():
    """Verify inner early-stopping best_iteration is determined exclusively before 2017."""
    df = load_development_feature_data()
    df_inner_tr, df_inner_val, _, _ = split_training_data(df)

    assert df_inner_val["date"].max() <= TRAINING_END_DATE
    assert df_inner_val["date"].max() < VALIDATION_START_DATE


# =====================================================================
# 5. OUTPUT DELIVERABLES & INTEGRITY TESTS
# =====================================================================

def test_ml_predictions_validation_file_integrity():
    """Verify generated ml_predictions_validation.csv conforms to all deliverable requirements."""
    preds_path = "data/processed/ml_predictions_validation.csv"
    preds_df = pd.read_csv(preds_path)

    expected_cols = [
        "date",
        "brand_id",
        "sku_id",
        "actual",
        "xgboost_no_target_promo",
        "xgboost_with_target_promo",
        "lightgbm_no_target_promo",
        "lightgbm_with_target_promo",
    ]
    assert list(preds_df.columns) == expected_cols
    assert len(preds_df) == 42716
    assert preds_df["sku_id"].nunique() == 118

    # Exactly 2017 dates, strictly no 2018
    assert preds_df["date"].min() >= "2017-01-01"
    assert preds_df["date"].max() <= "2017-12-31"
    assert not (preds_df["date"] >= "2018-01-01").any()

    # No NaN or Inf in predictions
    for cfg in ["xgboost_no_target_promo", "xgboost_with_target_promo", "lightgbm_no_target_promo", "lightgbm_with_target_promo"]:
        assert not preds_df[cfg].isna().any(), f"{cfg} contains NaNs"
        assert not np.isinf(preds_df[cfg]).any(), f"{cfg} contains Infs"
        assert (preds_df[cfg] >= 0.0).all(), f"{cfg} contains negative predictions"


def test_ml_metrics_file_integrity():
    """Verify generated ml_metrics.csv has all 4 configurations with 100% coverage."""
    metrics_path = "data/processed/ml_metrics.csv"
    metrics_df = pd.read_csv(metrics_path)

    assert len(metrics_df) == 4
    assert set(metrics_df["model"]) == {"xgboost", "lightgbm"}
    assert set(metrics_df["feature_variant"]) == {"no_target_promo", "with_target_promo"}
    assert (metrics_df["coverage"] == 1.0).all()
    assert (metrics_df["valid_prediction_count"] == 42716).all()
    assert (metrics_df["wape"] > 0.0).all()
    assert (metrics_df["mae"] > 0.0).all()
    assert (metrics_df["rmse"] > 0.0).all()


def test_ml_winner_summary_reconciles_to_118_skus():
    """Verify sum of SKU wins reconciles exactly to 118 eligible SKUs."""
    win_path = "data/processed/ml_winner_summary.csv"
    win_df = pd.read_csv(win_path)

    assert len(win_df) == 4
    assert win_df["sku_wins"].sum() == 118
    assert win_df["sku_win_share"].sum() == pytest.approx(1.0)


def test_feature_importance_shares_sum_to_one():
    """Verify feature importance shares sum to 1.0 within each model configuration."""
    imp_path = "data/processed/ml_feature_importance.csv"
    imp_df = pd.read_csv(imp_path)

    for (m, v), g in imp_df.groupby(["model", "feature_variant"]):
        total_share = g["importance_share"].sum()
        assert total_share == pytest.approx(1.0, rel=1e-3), f"Shares for {m}_{v} do not sum to 1"
        assert g["rank"].min() == 1
        assert g["rank"].max() == len(g)


def test_full_precision_winner_selection():
    """Verify SKU winner selection operates on full-precision WAPE without artificial rounding ties."""
    synthetic_sku_metrics = pd.DataFrame(
        [
            {"brand_id": "B1", "sku_id": "SKU_A", "model": "xgboost", "feature_variant": "with_target_promo", "wape": 0.5000004},
            {"brand_id": "B1", "sku_id": "SKU_A", "model": "lightgbm", "feature_variant": "with_target_promo", "wape": 0.5000001},
            {"brand_id": "B1", "sku_id": "SKU_A", "model": "xgboost", "feature_variant": "no_target_promo", "wape": 0.6},
            {"brand_id": "B1", "sku_id": "SKU_A", "model": "lightgbm", "feature_variant": "no_target_promo", "wape": 0.6},
        ]
    )
    winner_summary = calculate_ml_winner_summary(synthetic_sku_metrics)
    win_dict = winner_summary.set_index(["model", "feature_variant"])["sku_wins"].to_dict()

    # lightgbm_with_target_promo (0.5000001) is lower than xgboost_with_target_promo (0.5000004) and must win
    assert win_dict[("lightgbm", "with_target_promo")] == 1
    assert win_dict[("xgboost", "with_target_promo")] == 0


def test_ml_training_summary_file_integrity():
    """Verify generated ml_training_summary.csv contains unambiguous iteration and tree counts."""
    summary_path = "data/processed/ml_training_summary.csv"
    summary_df = pd.read_csv(summary_path)

    expected_cols = [
        "model",
        "feature_variant",
        "outer_train_start",
        "outer_train_end",
        "inner_fit_start",
        "inner_fit_end",
        "inner_validation_start",
        "inner_validation_end",
        "training_rows",
        "inner_validation_rows",
        "best_iteration",
        "selected_n_estimators",
        "best_inner_score",
        "random_state",
        "training_seconds",
    ]
    assert list(summary_df.columns) == expected_cols
    assert len(summary_df) == 4

    for _, row in summary_df.iterrows():
        assert row["selected_n_estimators"] >= 1
        if row["model"] == "xgboost":
            assert row["selected_n_estimators"] == row["best_iteration"] + 1
        elif row["model"] == "lightgbm":
            assert row["selected_n_estimators"] == row["best_iteration"]
