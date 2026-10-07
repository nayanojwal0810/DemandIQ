"""Leakage-safe forecasting feature pipeline for DemandIQ."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.features.feature_config import (
    CANONICAL_FEATURE_COLUMNS,
    DEVELOPMENT_CUTOFF_DATE,
    PROJECT_START_DATE,
    TRAINING_END_DATE,
    VALIDATION_START_DATE,
)

logger = logging.getLogger(__name__)


def build_forecasting_feature_table(
    df_canonical: pd.DataFrame,
    cutoff_date: str = DEVELOPMENT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Construct leakage-safe forecasting features from canonical SKU-day demand.

    The input table is strictly filtered to date <= cutoff_date prior to feature
    construction to prevent any forward temporal leakage.

    Args:
        df_canonical: Canonical daily demand DataFrame with columns
            ['date', 'brand_id', 'sku_id', 'quantity', 'promotion'].
        cutoff_date: Maximum target date permitted in the development feature set.

    Returns:
        DataFrame containing identity columns, target_quantity, historical lags,
        rolling statistics, calendar features, promotion signals, and split label.
    """
    required_cols = {"date", "brand_id", "sku_id", "quantity", "promotion"}
    missing_cols = required_cols - set(df_canonical.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing_cols)}")

    # 1. Enforce development cutoff boundary BEFORE any feature calculations
    df = df_canonical[df_canonical["date"] <= cutoff_date].copy()
    if df.empty:
        raise ValueError(f"No records found on or before cutoff date: {cutoff_date}")

    logger.info("Development cutoff applied: %s (retained %d rows).", cutoff_date, len(df))

    # Ensure correct data types
    df["date_dt"] = pd.to_datetime(df["date"])
    df["quantity"] = df["quantity"].astype(float)
    df["promotion"] = df["promotion"].astype(int)

    # 2. Build SKU-level time-series features
    sku_features = []
    project_start_dt = pd.Timestamp(PROJECT_START_DATE)

    for sku_id, g in df.groupby("sku_id", sort=False):
        g = g.sort_values("date_dt").copy()
        date_dt = g["date_dt"]
        qty = g["quantity"]
        promo = g["promotion"]

        # A. Historical Demand Lags
        # lag_1: immediately preceding observed demand date for the same SKU
        lag_1 = qty.shift(1)

        # lag_7, lag_14, lag_28: exact calendar date lookups (NaN if date was not observed)
        date_to_qty = pd.Series(qty.values, index=date_dt)
        lag_7 = (date_dt - pd.Timedelta(days=7)).map(date_to_qty)
        lag_14 = (date_dt - pd.Timedelta(days=14)).map(date_to_qty)
        lag_28 = (date_dt - pd.Timedelta(days=28)).map(date_to_qty)

        # B. Rolling Demand Features over trailing [t - k days, t - 1 day]
        s_qty = pd.Series(qty.values, index=date_dt)
        s_promo = pd.Series(promo.values, index=date_dt)

        r7 = s_qty.rolling("7D", closed="left")
        r14 = s_qty.rolling("14D", closed="left")
        r28 = s_qty.rolling("28D", closed="left")

        rolling_mean_7 = r7.mean()
        rolling_mean_14 = r14.mean()
        rolling_mean_28 = r28.mean()

        rolling_std_7 = s_qty.rolling("7D", closed="left", min_periods=2).std()
        rolling_std_28 = s_qty.rolling("28D", closed="left", min_periods=2).std()

        observed_days_7 = s_qty.rolling("7D", closed="left", min_periods=0).count().astype(int)
        observed_days_28 = s_qty.rolling("28D", closed="left", min_periods=0).count().astype(int)

        zero_rate_28 = (s_qty == 0).astype(float).rolling("28D", closed="left", min_periods=1).mean()

        # C. Timing feature
        days_since_prev = (date_dt - date_dt.shift(1)).dt.days

        # E. Historical Promotion Features
        lag_1_promotion = promo.shift(1)
        promotion_rate_28 = s_promo.astype(float).rolling("28D", closed="left", min_periods=1).mean()
        promotion_observations_28 = s_promo.rolling("28D", closed="left", min_periods=0).count().astype(int)

        sku_df = pd.DataFrame(
            {
                "date": g["date"].values,
                "brand_id": g["brand_id"].values,
                "sku_id": g["sku_id"].values,
                "target_quantity": qty.values,
                "lag_1": lag_1.values,
                "lag_7": lag_7.values,
                "lag_14": lag_14.values,
                "lag_28": lag_28.values,
                "rolling_mean_7": rolling_mean_7.values,
                "rolling_mean_14": rolling_mean_14.values,
                "rolling_mean_28": rolling_mean_28.values,
                "rolling_std_7": rolling_std_7.values,
                "rolling_std_28": rolling_std_28.values,
                "zero_rate_28": zero_rate_28.values,
                "observed_days_7": observed_days_7.values,
                "observed_days_28": observed_days_28.values,
                "days_since_previous_observation": days_since_prev.values,
                "lag_1_promotion": lag_1_promotion.values,
                "promotion_rate_28": promotion_rate_28.values,
                "promotion_observations_28": promotion_observations_28.values,
                "target_promotion": promo.values,
                "date_dt": date_dt.values,
            }
        )
        sku_features.append(sku_df)

    out_df = pd.concat(sku_features, ignore_index=True)

    # D. Calendar Features (computed deterministically from target date)
    date_dt = pd.to_datetime(out_df["date_dt"])
    out_df["day_of_week"] = date_dt.dt.dayofweek.astype(int)
    out_df["week_of_year"] = date_dt.dt.isocalendar().week.astype(int)
    out_df["month"] = date_dt.dt.month.astype(int)
    out_df["quarter"] = date_dt.dt.quarter.astype(int)
    out_df["day_of_month"] = date_dt.dt.day.astype(int)
    out_df["day_of_year"] = date_dt.dt.dayofyear.astype(int)
    out_df["year"] = date_dt.dt.year.astype(int)
    out_df["is_weekend"] = (date_dt.dt.dayofweek >= 5).astype(int)
    out_df["time_index"] = (date_dt - project_start_dt).dt.days.astype(int)

    # Split assignment: 2014-01-02..2016-12-31 -> train, 2017-01-01..2017-12-31 -> validation
    out_df["split"] = np.where(out_df["date"] <= TRAINING_END_DATE, "train", "validation")

    # Drop temporary datetime column
    out_df = out_df.drop(columns=["date_dt"])

    # Deterministic sorting by date, brand_id, sku_id
    out_df = out_df.sort_values(by=["date", "brand_id", "sku_id"]).reset_index(drop=True)

    # Reorder columns to match canonical column specification exactly
    out_df = out_df[CANONICAL_FEATURE_COLUMNS]

    logger.info("Forecasting feature table constructed.")
    logger.info("Target rows created: %d.", len(out_df))
    logger.info(
        "Split counts: train=%d, validation=%d.",
        (out_df["split"] == "train").sum(),
        (out_df["split"] == "validation").sum(),
    )

    return out_df


def save_forecasting_features(
    df_features: pd.DataFrame,
    output_path: Union[str, Path] = Path("data/processed/forecasting_features_development.csv"),
) -> None:
    """Save forecasting features DataFrame to CSV deterministically."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(path, index=False)
    logger.info("Forecasting feature dataset written: %s", path)


def load_forecasting_features(
    input_path: Union[str, Path] = Path("data/processed/forecasting_features_development.csv"),
) -> pd.DataFrame:
    """Reload forecasting feature dataset with deterministic data types."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    dtype_spec = {
        "date": str,
        "brand_id": str,
        "sku_id": str,
        "target_quantity": float,
        "lag_1": float,
        "lag_7": float,
        "lag_14": float,
        "lag_28": float,
        "rolling_mean_7": float,
        "rolling_mean_14": float,
        "rolling_mean_28": float,
        "rolling_std_7": float,
        "rolling_std_28": float,
        "zero_rate_28": float,
        "observed_days_7": int,
        "observed_days_28": int,
        "days_since_previous_observation": float,
        "day_of_week": int,
        "week_of_year": int,
        "month": int,
        "quarter": int,
        "day_of_month": int,
        "day_of_year": int,
        "year": int,
        "is_weekend": int,
        "time_index": int,
        "lag_1_promotion": float,
        "promotion_rate_28": float,
        "promotion_observations_28": int,
        "target_promotion": int,
        "split": str,
    }
    return pd.read_csv(path, dtype=dtype_spec)


def main() -> None:
    """Run feature extraction pipeline on canonical SKU-day data."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    canonical_path = Path("data/processed/sku_demand_daily.csv")
    output_path = Path("data/processed/forecasting_features_development.csv")

    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical dataset not found at {canonical_path}")

    logger.info("Loading canonical dataset from %s...", canonical_path)
    df_canonical = pd.read_csv(canonical_path)

    df_features = build_forecasting_feature_table(df_canonical)
    save_forecasting_features(df_features, output_path)
    logger.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()
