"""Unit, synthetic, leakage, and output integrity tests for forecasting baselines."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.features.feature_config import (
    DEVELOPMENT_CUTOFF_DATE,
    TRAINING_END_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
)
from src.forecasting.baselines import (
    CrostonState,
    fit_croston_sba_training_model,
    fit_ses_training_model,
    generate_croston_sba_validation_forecast,
    generate_moving_average_forecast,
    generate_naive_forecast,
    generate_seasonal_naive_forecast,
    generate_ses_validation_forecast,
)
from src.forecasting.evaluate_baselines import (
    BASELINES,
    PREDICTION_COLUMNS,
    calculate_baseline_metrics,
    calculate_baseline_winners,
    calculate_sku_metrics,
    generate_validation_predictions,
    load_baseline_data,
)


# =====================================================================
# 1. NAIVE TESTS
# =====================================================================

def test_naive_previous_observed_value_and_calendar_gaps():
    """Verify naive baseline takes previous observed row, ignoring calendar gaps."""
    # Sequence with gap: Jan 2 (10), Jan 5 (25), Jan 8 (40)
    qty = np.array([10.0, 25.0, 40.0])
    forecasts = generate_naive_forecast(qty)

    assert np.isnan(forecasts[0])
    assert forecasts[1] == 10.0  # Jan 5 gets Jan 2's demand
    assert forecasts[2] == 25.0  # Jan 8 gets Jan 5's demand


def test_naive_target_value_does_not_affect_own_forecast():
    """Verify target actual value at step t does not leak into forecast at step t."""
    qty1 = np.array([10.0, 20.0, 30.0])
    qty2 = np.array([10.0, 20.0, 999.0])  # change target value at t=2

    fc1 = generate_naive_forecast(qty1)
    fc2 = generate_naive_forecast(qty2)

    assert fc1[2] == fc2[2] == 20.0


# =====================================================================
# 2. SEASONAL NAIVE TESTS
# =====================================================================

def test_seasonal_naive_exact_t_minus_7_lookup():
    """Verify seasonal naive returns exact t - 7 calendar days demand."""
    dates = pd.to_datetime(["2017-01-08", "2017-01-09", "2017-01-15"])
    sku_demand_map = {
        ("SKU_1", pd.Timestamp("2017-01-01")): 12.0,
        ("SKU_1", pd.Timestamp("2017-01-08")): 18.0,
    }

    fc = generate_seasonal_naive_forecast(dates, sku_demand_map, "SKU_1")

    assert fc[0] == 12.0  # 2017-01-08 looks up 2017-01-01 -> 12.0
    assert np.isnan(fc[1])  # 2017-01-09 looks up 2017-01-02 -> missing -> NaN
    assert fc[2] == 18.0  # 2017-01-15 looks up 2017-01-08 -> 18.0


def test_seasonal_naive_missing_t_minus_7_produces_nan_no_fallback():
    """Verify missing t - 7 strictly produces NaN without falling back to naive or moving average."""
    dates = pd.to_datetime(["2017-01-10"])
    sku_demand_map = {
        ("SKU_1", pd.Timestamp("2017-01-09")): 50.0,  # t - 1 exists
        ("SKU_1", pd.Timestamp("2017-01-04")): 30.0,  # t - 6 exists
    }

    fc = generate_seasonal_naive_forecast(dates, sku_demand_map, "SKU_1")
    assert np.isnan(fc[0])


# =====================================================================
# 3. 7-DAY CALENDAR MOVING AVERAGE TESTS
# =====================================================================

def test_moving_average_7d_exact_window_and_target_exclusion():
    """Verify moving average averages [t - 7D, t - 1D] excluding target date t."""
    dates = pd.to_datetime(["2017-01-01", "2017-01-03", "2017-01-05", "2017-01-08"])
    qty = pd.Series([10.0, 20.0, 30.0, 100.0])

    fc = generate_moving_average_forecast(dates, qty, window="7D", min_periods=1)

    # For 2017-01-08: window is [2017-01-01, 2017-01-08).
    # Observed prior dates in window: Jan 1 (10), Jan 3 (20), Jan 5 (30).
    # Jan 8 actual (100.0) must be excluded.
    # Mean = (10 + 20 + 30) / 3 = 20.0
    assert np.isnan(fc[0])
    assert fc[3] == pytest.approx(20.0)


def test_moving_average_7d_calendar_gaps_not_converted_to_zeros():
    """Verify unobserved calendar dates in the window are not treated as zeros."""
    # Jan 1 (10) and Jan 5 (20). Window for Jan 7 is [Jan 1, Jan 7).
    # If unobserved were zeros, sum would be 30 / 6 = 5.
    # Correct semantics: only observed values are averaged -> (10 + 20) / 2 = 15.0
    dates = pd.to_datetime(["2017-01-01", "2017-01-05", "2017-01-07"])
    qty = pd.Series([10.0, 20.0, 99.0])

    fc = generate_moving_average_forecast(dates, qty, window="7D", min_periods=1)
    assert fc[2] == pytest.approx(15.0)


def test_moving_average_7d_min_periods():
    """Verify min_periods threshold enforcement."""
    dates = pd.to_datetime(["2017-01-01", "2017-01-05"])
    qty = pd.Series([10.0, 20.0])

    fc_min1 = generate_moving_average_forecast(dates, qty, window="7D", min_periods=1)
    fc_min2 = generate_moving_average_forecast(dates, qty, window="7D", min_periods=2)

    assert fc_min1[1] == 10.0
    assert np.isnan(fc_min2[1])  # Only 1 observation in window, min_periods=2 requires 2


# =====================================================================
# 4. SIMPLE EXPONENTIAL SMOOTHING (SES) TESTS
# =====================================================================

def test_ses_alpha_estimated_from_training_only():
    """Verify alpha is estimated from training data and remains fixed throughout validation."""
    train_y = np.array([10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.0])
    alpha, init_level = fit_ses_training_model(train_y)

    assert 0.0 < alpha <= 1.0
    assert init_level > 0.0


def test_ses_recurrence_and_no_target_leakage():
    """Verify SES 1-step forecast uses prior level and does not incorporate target actual."""
    alpha = 0.2
    init_level = 10.0
    val_y = np.array([20.0, 30.0, 40.0])

    fc = generate_ses_validation_forecast(val_y, alpha=alpha, initial_level=init_level)

    # Step 0: forecast = l_0 = 10.0
    assert fc[0] == 10.0
    # After step 0, l_1 = 0.2 * 20 + 0.8 * 10 = 4 + 8 = 12.0
    assert fc[1] == pytest.approx(12.0)
    # After step 1, l_2 = 0.2 * 30 + 0.8 * 12 = 6 + 9.6 = 15.6
    assert fc[2] == pytest.approx(15.6)


def test_ses_changing_target_does_not_change_target_forecast():
    """Verify mutating target actual does not alter that target's forecast."""
    alpha = 0.3
    init_level = 15.0
    val_y1 = np.array([10.0, 25.0, 30.0])
    val_y2 = np.array([10.0, 999.0, 30.0])  # change step 1 actual

    fc1 = generate_ses_validation_forecast(val_y1, alpha, init_level)
    fc2 = generate_ses_validation_forecast(val_y2, alpha, init_level)

    # Step 1 forecast was produced BEFORE observing step 1 actual
    assert fc1[1] == fc2[1]
    # But step 2 forecast SHOULD differ because step 1 actual updated the state
    assert fc1[2] != fc2[2]


def test_ses_non_negative_forecasts():
    """Verify SES forecasts remain non-negative for non-negative demand sequences."""
    train_y = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    alpha, init_level = fit_ses_training_model(train_y)
    val_y = np.array([0.0, 0.0, 1.0, 0.0])

    fc = generate_ses_validation_forecast(val_y, alpha, init_level)
    assert np.all(fc >= 0.0)


# =====================================================================
# 5. CROSTON SBA TESTS
# =====================================================================

def test_croston_sba_state_updates_and_observed_positions():
    """Verify Croston state updates only on positive demand and uses observed-period positions."""
    alpha = 0.1
    # Observed positions:
    # idx 0: 0
    # idx 1: 10 (first pos) -> z = 10.0, last_pos = 1
    # idx 2: 0
    # idx 3: 0
    # idx 4: 20 (second pos) -> interval q = 4 - 1 = 3.
    # p = 3.0, z = 0.1 * 20 + 0.9 * 10 = 11.0, last_pos = 4
    train_y = np.array([0.0, 10.0, 0.0, 0.0, 20.0])
    state = fit_croston_sba_training_model(train_y, alpha=alpha)

    assert not state.is_all_zero
    assert state.last_pos_idx == 4
    assert state.smoothed_demand_size == pytest.approx(11.0)
    assert state.smoothed_interval == pytest.approx(3.0)


def test_croston_sba_correction_factor_applied():
    """Verify SBA correction factor (1 - alpha / 2) is applied."""
    alpha = 0.1
    state = CrostonState(
        smoothed_demand_size=10.0,
        smoothed_interval=2.0,
        last_pos_idx=0,
        is_all_zero=False,
    )
    val_y = np.array([0.0])
    fc = generate_croston_sba_validation_forecast(val_y, state, start_pos=1, alpha=alpha)

    expected_fc = (1.0 - alpha / 2.0) * (10.0 / 2.0)  # 0.95 * 5.0 = 4.75
    assert fc[0] == pytest.approx(expected_fc)


def test_croston_sba_zero_observations_do_not_update_state():
    """Verify zero-demand observations leave smoothed size and interval unchanged."""
    alpha = 0.1
    state = CrostonState(
        smoothed_demand_size=10.0,
        smoothed_interval=2.0,
        last_pos_idx=0,
        is_all_zero=False,
    )
    val_y = np.array([0.0, 0.0, 0.0])
    fc = generate_croston_sba_validation_forecast(val_y, state, start_pos=1, alpha=alpha)

    assert np.all(fc == pytest.approx(4.75))


def test_croston_sba_all_zero_training_history():
    """Verify all-zero training history produces 0 forecast without errors."""
    train_y = np.zeros(10)
    state = fit_croston_sba_training_model(train_y, alpha=0.1)

    assert state.is_all_zero
    val_y = np.array([0.0, 0.0, 5.0, 0.0])
    fc = generate_croston_sba_validation_forecast(val_y, state, start_pos=10, alpha=0.1)

    assert fc[0] == 0.0
    assert fc[1] == 0.0
    assert fc[2] == 0.0  # Forecast BEFORE seeing 5.0 is still 0.0
    assert fc[3] > 0.0  # After observing 5.0, future forecast becomes positive


def test_croston_sba_exactly_one_positive_event_history():
    """Verify single positive event uses deterministic fallback interval."""
    train_y = np.array([0.0, 0.0, 8.0, 0.0])
    state = fit_croston_sba_training_model(train_y, alpha=0.1)

    assert not state.is_all_zero
    assert state.smoothed_demand_size == 8.0
    assert state.smoothed_interval == 1.0  # Fallback interval


def test_croston_sba_no_target_leakage():
    """Verify current validation target is not used for its own forecast."""
    alpha = 0.1
    state = CrostonState(
        smoothed_demand_size=10.0,
        smoothed_interval=2.0,
        last_pos_idx=0,
        is_all_zero=False,
    )
    val_y1 = np.array([5.0])
    val_y2 = np.array([100.0])

    fc1 = generate_croston_sba_validation_forecast(val_y1, state, start_pos=1, alpha=alpha)
    fc2 = generate_croston_sba_validation_forecast(val_y2, state, start_pos=1, alpha=alpha)

    assert fc1[0] == fc2[0]


# =====================================================================
# 6. METRICS TESTS
# =====================================================================

def test_metrics_wape_mae_rmse_calculation():
    """Verify standard metric calculations on known synthetic values."""
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])

    # abs errors: [2, 2, 3] -> sum = 7
    # actual sum = 60
    # WAPE = 7 / 60
    # MAE = 7 / 3
    # sq errors: [4, 4, 9] -> mean = 17 / 3 -> RMSE = sqrt(17 / 3)
    assert calculate_wape(actual, pred) == pytest.approx(7.0 / 60.0)
    assert calculate_mae(actual, pred) == pytest.approx(7.0 / 3.0)
    assert calculate_rmse(actual, pred) == pytest.approx(np.sqrt(17.0 / 3.0))


def test_metrics_zero_actual_handling():
    """Verify zero actual sum handling prevents zero division crashes."""
    actual_zero = np.array([0.0, 0.0])
    pred_zero = np.array([0.0, 0.0])
    pred_nonzero = np.array([1.0, 2.0])

    assert calculate_wape(actual_zero, pred_zero) == 0.0
    assert np.isnan(calculate_wape(actual_zero, pred_nonzero))


# =====================================================================
# 7. END-TO-END PIPELINE AND OUTPUT INTEGRITY TESTS
# =====================================================================

def test_development_cutoff_and_no_2018_in_predictions():
    """Verify predictions cover validation period (2017) and strictly contain no 2018 rows."""
    df = load_baseline_data()
    val_preds = generate_validation_predictions(df)

    # 1. Dates in predictions must be in 2017
    assert val_preds["date"].min() >= VALIDATION_START_DATE
    assert val_preds["date"].max() <= VALIDATION_END_DATE

    # 2. Absolutely no 2018 dates
    has_2018 = (val_preds["date"] >= "2018-01-01").any()
    assert not has_2018


def test_predictions_schema_and_deterministic_sort():
    """Verify predictions output schema, row uniqueness, and sorting."""
    df = load_baseline_data()
    val_preds = generate_validation_predictions(df)

    assert list(val_preds.columns) == PREDICTION_COLUMNS
    assert len(val_preds) == 42716
    assert val_preds["sku_id"].nunique() == 118

    # Uniqueness: exactly one row per date x sku_id
    dups = val_preds.duplicated(subset=["date", "sku_id"]).sum()
    assert dups == 0

    # Deterministic sorting
    sorted_df = val_preds.sort_values(["date", "brand_id", "sku_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(val_preds, sorted_df)


def test_common_valid_requires_all_five_baselines():
    """Verify common_valid subset strictly requires valid predictions across all five baselines."""
    df = load_baseline_data()
    val_preds = generate_validation_predictions(df)
    portfolio_metrics = calculate_baseline_metrics(val_preds)

    cv = portfolio_metrics[portfolio_metrics["evaluation_scope"] == "common_valid"]
    assert len(cv) == 5

    # All five baselines in common_valid must evaluate the exact same row count
    common_counts = cv["valid_prediction_count"].unique()
    assert len(common_counts) == 1
    assert common_counts[0] == 42480  # 42,480 common rows (236 seasonal naive missing t-7)


def test_sku_winners_reconcile_to_total_skus():
    """Verify sum of SKU wins reconciles to eligible SKU count (118)."""
    df = load_baseline_data()
    val_preds = generate_validation_predictions(df)
    sku_metrics = calculate_sku_metrics(val_preds)
    winner_summary = calculate_baseline_winners(sku_metrics)

    assert winner_summary["sku_wins"].sum() == 118
    assert winner_summary["sku_win_share"].sum() == pytest.approx(1.0)
    assert set(winner_summary["baseline"]) == set(BASELINES)


def test_no_inf_or_unexpected_nan_in_predictions():
    """Verify forecasts contain no infinite values and no unexplained NaNs."""
    df = load_baseline_data()
    val_preds = generate_validation_predictions(df)

    for b in ["naive", "moving_average_7d", "ets_ses", "croston_sba"]:
        assert not np.isinf(val_preds[b]).any(), f"{b} contains inf"
        assert not val_preds[b].isna().any(), f"{b} contains unexpected NaN"

    # seasonal_naive_7 has legitimate NaNs only where t - 7 was not observed
    assert not np.isinf(val_preds["seasonal_naive_7"]).any()
    assert val_preds["seasonal_naive_7"].isna().sum() == 236


def test_ses_constant_series_follows_normal_fitting_path():
    """Verify constant training series follows standard statsmodels fitting path rather than hardcoded 0.05."""
    train_y = np.full(50, 5.0)
    alpha, init_level = fit_ses_training_model(train_y)

    assert init_level == pytest.approx(5.0)
    assert alpha != 0.05  # Proves it does not use a hardcoded 0.05 special case
    assert 0.0 < alpha <= 1.0


def test_sku_winner_selection_uses_full_precision_wape():
    """Verify winner selection uses full precision rather than rounded ties.

    If two baselines differ by less than 1e-6 WAPE (e.g. 0.5000001 vs 0.5000004),
    the lower full-precision value must win even if a tie-break priority would favor
    the other under 6-decimal rounding.
    """
    # ets_ses has higher tie-break priority than croston_sba (priority 0 vs 1).
    # Give ets_ses WAPE = 0.5000004, and croston_sba WAPE = 0.5000001.
    # If rounded to 6 decimals, both would be 0.500000 and ets_ses would win on tie-break.
    # Under full precision, croston_sba (0.5000001 < 0.5000004) MUST win.
    synthetic_sku_metrics = pd.DataFrame(
        [
            {"evaluation_scope": "common_valid", "brand_id": "B1", "sku_id": "SKU_X", "baseline": "naive", "wape": 0.8},
            {"evaluation_scope": "common_valid", "brand_id": "B1", "sku_id": "SKU_X", "baseline": "seasonal_naive_7", "wape": 0.9},
            {"evaluation_scope": "common_valid", "brand_id": "B1", "sku_id": "SKU_X", "baseline": "moving_average_7d", "wape": 0.7},
            {"evaluation_scope": "common_valid", "brand_id": "B1", "sku_id": "SKU_X", "baseline": "ets_ses", "wape": 0.5000004},
            {"evaluation_scope": "common_valid", "brand_id": "B1", "sku_id": "SKU_X", "baseline": "croston_sba", "wape": 0.5000001},
        ]
    )
    winner_summary = calculate_baseline_winners(synthetic_sku_metrics)
    winner_dict = winner_summary.set_index("baseline")["sku_wins"].to_dict()

    assert winner_dict["croston_sba"] == 1
    assert winner_dict["ets_ses"] == 0
