"""Classical forecasting baseline implementations for DemandIQ.

This module provides five deterministic, leakage-safe forecasting baselines:
1. Naive (immediately preceding observed demand)
2. Seasonal Naive 7-Day (exact t - 7 calendar days demand)
3. 7-Day Calendar Moving Average (trailing [t - 7D, t - 1D] mean)
4. Simple Exponential Smoothing / ETS(A,N,N) (level-only, training-fitted alpha)
5. Croston SBA (Syntetos-Boylan Approximation for intermittent demand)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


@dataclass
class CrostonState:
    """State tracking for Croston Syntetos-Boylan Approximation (SBA).

    Attributes
    ----------
    smoothed_demand_size : float
        Current smoothed non-zero demand size (z_t).
    smoothed_interval : float
        Current smoothed inter-demand interval in observed-period units (p_t).
    last_pos_idx : Optional[int]
        Index position of the most recent positive demand event in the observed sequence.
    is_all_zero : bool
        Flag indicating whether no positive demand has been observed yet.
    """

    smoothed_demand_size: float
    smoothed_interval: float
    last_pos_idx: Optional[int]
    is_all_zero: bool


def generate_naive_forecast(quantities: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Generate 1-step ahead naive forecasts from preceding observed records.

    For SKU s at validation target t, forecast(s, t) equals demand at the
    immediately preceding observed record for SKU s. This follows observed-date
    sequence order and does not insert unobserved calendar dates as zeros.

    Parameters
    ----------
    quantities : array-like
        Ordered sequence of observed demand quantities for a single SKU.

    Returns
    -------
    np.ndarray
        Array of forecasts where forecast[i] = quantities[i-1], and forecast[0] = np.nan.
    """
    s = pd.Series(quantities, dtype=float)
    return s.shift(1).values


def generate_seasonal_naive_forecast(
    dates: pd.Series,
    sku_demand_map: Dict[Tuple[str, pd.Timestamp], float],
    sku_id: str,
) -> np.ndarray:
    """Generate seasonal naive forecasts using exact 7 calendar days lookup.

    For SKU s at validation target date t, forecast(s, t) = demand(s, t - 7 calendar days).
    If exact t - 7 was not an observed trading date, returns np.nan.
    No forward-fill, backward-fill, or fallback baselines are applied.

    Parameters
    ----------
    dates : pd.Series
        Series of dates for the SKU.
    sku_demand_map : Dict[Tuple[str, pd.Timestamp], float]
        Dictionary mapping (sku_id, date_dt) -> quantity across the full history.
    sku_id : str
        The SKU identifier.

    Returns
    -------
    np.ndarray
        Array of seasonal naive forecasts with NaN for unobserved t - 7 dates.
    """
    dates_dt = pd.to_datetime(dates)
    forecasts = np.full(len(dates_dt), np.nan, dtype=float)

    for i, d in enumerate(dates_dt):
        target_prior = d - pd.Timedelta(days=7)
        key = (sku_id, target_prior)
        if key in sku_demand_map:
            forecasts[i] = sku_demand_map[key]

    return forecasts


def generate_moving_average_forecast(
    dates: pd.Series,
    quantities: pd.Series,
    window: str = "7D",
    min_periods: int = 1,
) -> np.ndarray:
    """Calculate moving average demand over exact trailing calendar window.

    For target date t, computes mean of observed demand in [t - 7D, t - 1D].
    The target date is strictly excluded (closed='left'). Unobserved calendar
    dates are not converted to zeros; only observed values are averaged.

    Parameters
    ----------
    dates : pd.Series
        Chronological observation dates for a single SKU.
    quantities : pd.Series
        Corresponding observed demand quantities.
    window : str, default "7D"
        Rolling calendar time-window specification.
    min_periods : int, default 1
        Minimum number of observed records required in the window.

    Returns
    -------
    np.ndarray
        Array of rolling mean forecasts.
    """
    dates_dt = pd.to_datetime(dates)
    s = pd.Series(quantities.values, index=dates_dt, dtype=float)
    roll = s.rolling(window, closed="left", min_periods=min_periods).mean()
    return roll.values


def fit_ses_training_model(y_train: np.ndarray) -> Tuple[float, float]:
    """Fit level-only Simple Exponential Smoothing (ETS(A,N,N)) on training data.

    Smoothing parameter alpha is estimated strictly on training data using
    statsmodels SimpleExpSmoothing with optimized=True.

    Parameters
    ----------
    y_train : np.ndarray
        Chronological training demand array for a single SKU.

    Returns
    -------
    Tuple[float, float]
        (alpha, initial_level) where initial_level is the fitted level at the
        end of the training sequence (l_T), ready for 1-step validation forecasting.
    """
    y = np.asarray(y_train, dtype=float)

    # Edge case: empty training history
    if len(y) == 0:
        return 0.1, 0.0

    try:
        model = SimpleExpSmoothing(y, initialization_method="estimated")
        res = model.fit(optimized=True)
        alpha = float(res.params.get("smoothing_level", 0.1))
        # Ensure alpha is well-behaved within [0, 1]
        alpha = min(max(alpha, 1e-4), 1.0)
        initial_level = float(res.level[-1])
        initial_level = max(0.0, initial_level)
    except Exception:
        # Robust fallback only for genuine fitting / optimization exceptions.
        # Fallback is deterministic, training-only, and non-leaky.
        alpha = 0.1
        initial_level = max(0.0, float(np.mean(y)))

    return alpha, initial_level


def generate_ses_validation_forecast(
    y_val: np.ndarray,
    alpha: float,
    initial_level: float,
) -> np.ndarray:
    """Generate 1-step ahead SES forecasts sequentially through validation period.

    Follows standard recurrence:
        forecast_{t} = l_{t-1}
        l_t = alpha * y_t + (1 - alpha) * l_{t-1}

    The target actual y_t is recorded only after forecast_{t} is produced,
    preventing any target leakage. Alpha is fixed from training and not refit.

    Parameters
    ----------
    y_val : np.ndarray
        Chronological validation demand observations for a single SKU.
    alpha : float
        Training-estimated smoothing parameter.
    initial_level : float
        Level at the conclusion of the training period (l_T).

    Returns
    -------
    np.ndarray
        Array of 1-step ahead forecasts for each validation observation.
    """
    forecasts = np.zeros(len(y_val), dtype=float)
    level = float(initial_level)

    for t in range(len(y_val)):
        # 1. Forecast before observing actual y_val[t]
        forecasts[t] = max(0.0, level)

        # 2. Update level with observed y_val[t]
        y_actual = float(y_val[t])
        level = alpha * y_actual + (1.0 - alpha) * level

    return forecasts


def fit_croston_sba_training_model(
    train_y: np.ndarray,
    alpha: float = 0.1,
) -> CrostonState:
    """Initialize and fit Croston SBA state across training observations.

    Uses Syntetos-Boylan Approximation (SBA) with fixed alpha=0.1.
    Time semantics follow the sequence of observed SKU records (unobserved
    calendar dates are not inserted as zeros). Inter-demand interval is
    measured in observed-period positions.

    Parameters
    ----------
    train_y : np.ndarray
        Training demand sequence for a single SKU.
    alpha : float, default 0.1
        Fixed smoothing parameter for demand size and interval.

    Returns
    -------
    CrostonState
        Initialized Croston state at the end of training.
    """
    y = np.asarray(train_y, dtype=float)
    pos_indices = np.where(y > 0)[0]

    # Edge case 1: all-zero training history
    if len(pos_indices) == 0:
        return CrostonState(
            smoothed_demand_size=0.0,
            smoothed_interval=1.0,
            last_pos_idx=None,
            is_all_zero=True,
        )

    # Edge case 2: exactly one positive event in training
    if len(pos_indices) == 1:
        first_idx = int(pos_indices[0])
        return CrostonState(
            smoothed_demand_size=float(y[first_idx]),
            smoothed_interval=1.0,  # Documented deterministic fallback
            last_pos_idx=first_idx,
            is_all_zero=False,
        )

    # Standard case: >= 2 positive events
    first_idx = int(pos_indices[0])
    second_idx = int(pos_indices[1])

    # Initialize demand size and first interval
    z = float(y[first_idx])
    p = float(second_idx - first_idx)

    # Update state at second positive event
    z = alpha * float(y[second_idx]) + (1.0 - alpha) * z
    last_pos = second_idx

    # Process remaining training positive events sequentially
    for k in range(2, len(pos_indices)):
        curr_idx = int(pos_indices[k])
        q = float(curr_idx - last_pos)
        z = alpha * float(y[curr_idx]) + (1.0 - alpha) * z
        p = alpha * q + (1.0 - alpha) * p
        last_pos = curr_idx

    return CrostonState(
        smoothed_demand_size=max(0.0, z),
        smoothed_interval=max(1.0, p),
        last_pos_idx=last_pos,
        is_all_zero=False,
    )


def generate_croston_sba_validation_forecast(
    val_y: np.ndarray,
    state: CrostonState,
    start_pos: int,
    alpha: float = 0.1,
) -> np.ndarray:
    """Generate 1-step ahead Croston SBA forecasts sequentially across validation.

    Syntetos-Boylan Approximation applies:
        forecast = (1 - alpha / 2) * (z_t / p_t)

    State updates only occur when non-zero demand is observed. Zero-demand
    observations do not alter z_t or p_t. Current actual is observed strictly
    after generating the forecast.

    Parameters
    ----------
    val_y : np.ndarray
        Validation demand observations for a single SKU.
    state : CrostonState
        Fitted Croston state from the conclusion of training.
    start_pos : int
        Global observed index position corresponding to val_y[0].
    alpha : float, default 0.1
        Fixed smoothing parameter.

    Returns
    -------
    np.ndarray
        Array of Croston SBA forecasts for each validation observation.
    """
    forecasts = np.zeros(len(val_y), dtype=float)
    sba_factor = 1.0 - (alpha / 2.0)

    z = float(state.smoothed_demand_size)
    p = float(state.smoothed_interval)
    last_pos = state.last_pos_idx
    is_all_zero = state.is_all_zero

    for step, y_actual in enumerate(val_y):
        curr_pos = start_pos + step

        # 1. Forecast before seeing y_actual
        if is_all_zero:
            forecasts[step] = 0.0
        else:
            denom = max(p, 1e-6)
            fc = sba_factor * (z / denom)
            forecasts[step] = max(0.0, float(fc))

        # 2. Update state after observing y_actual
        val_qty = float(y_actual)
        if val_qty > 0:
            if is_all_zero:
                z = val_qty
                p = 1.0
                last_pos = curr_pos
                is_all_zero = False
            else:
                q = float(curr_pos - last_pos) if last_pos is not None else 1.0
                z = alpha * val_qty + (1.0 - alpha) * z
                p = alpha * q + (1.0 - alpha) * p
                last_pos = curr_pos

    return forecasts
