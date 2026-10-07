"""Metric definitions for DemandIQ forecast evaluation.

Primary metric is WAPE (Weighted Absolute Percentage Error), accompanied by MAE and RMSE.
"""

from typing import Union
import numpy as np
import pandas as pd


def calculate_wape(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    Definition:
        WAPE = sum(|actual - forecast|) / sum(|actual|)

    Parameters
    ----------
    actual : array-like
        Ground truth demand values.
    forecast : array-like
        Predicted demand values.

    Returns
    -------
    float
        WAPE value. Returns 0.0 if both total actual and total absolute error are 0.
        Returns np.nan if total actual is 0 but error is non-zero.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(forecast, dtype=float)

    if len(y_true) == 0:
        return np.nan

    abs_errors = np.abs(y_true - y_pred)
    sum_errors = float(np.sum(abs_errors))
    sum_actual = float(np.sum(np.abs(y_true)))

    if sum_actual == 0.0:
        if sum_errors == 0.0:
            return 0.0
        return np.nan

    return sum_errors / sum_actual


def calculate_mae(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate Mean Absolute Error (MAE).

    Definition:
        MAE = mean(|actual - forecast|)

    Parameters
    ----------
    actual : array-like
        Ground truth demand values.
    forecast : array-like
        Predicted demand values.

    Returns
    -------
    float
        MAE value, or np.nan if input arrays are empty.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(forecast, dtype=float)

    if len(y_true) == 0:
        return np.nan

    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Definition:
        RMSE = sqrt(mean((actual - forecast)^2))

    Parameters
    ----------
    actual : array-like
        Ground truth demand values.
    forecast : array-like
        Predicted demand values.

    Returns
    -------
    float
        RMSE value, or np.nan if input arrays are empty.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(forecast, dtype=float)

    if len(y_true) == 0:
        return np.nan

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))
