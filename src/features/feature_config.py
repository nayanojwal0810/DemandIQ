"""Feature configuration and provenance definitions for DemandIQ."""

from typing import Dict, List

# Date boundaries
DEVELOPMENT_CUTOFF_DATE: str = "2017-12-31"
PROJECT_START_DATE: str = "2014-01-02"
TRAINING_START_DATE: str = "2014-01-02"
TRAINING_END_DATE: str = "2016-12-31"
VALIDATION_START_DATE: str = "2017-01-01"
VALIDATION_END_DATE: str = "2017-12-31"

# Window parameters
DEMAND_LAG_DAYS: List[int] = [7, 14, 28]
ROLLING_WINDOW_DAYS: List[int] = [7, 14, 28]
ROLLING_STD_WINDOWS: List[int] = [7, 28]

# Feature column definitions
ID_COLUMNS: List[str] = ["date", "brand_id", "sku_id"]
TARGET_COLUMN: str = "target_quantity"
SPLIT_COLUMN: str = "split"

HISTORICAL_DEMAND_LAGS: List[str] = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
]

ROLLING_DEMAND_FEATURES: List[str] = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "zero_rate_28",
    "observed_days_7",
    "observed_days_28",
]

TIMING_FEATURES: List[str] = [
    "days_since_previous_observation",
]

CALENDAR_FEATURES: List[str] = [
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "day_of_month",
    "day_of_year",
    "year",
    "is_weekend",
    "time_index",
]

HISTORICAL_PROMOTION_FEATURES: List[str] = [
    "lag_1_promotion",
    "promotion_rate_28",
    "promotion_observations_28",
]

TARGET_PROMOTION_FEATURES: List[str] = [
    "target_promotion",
]

# Canonical ordered column list for output table
CANONICAL_FEATURE_COLUMNS: List[str] = [
    "date",
    "brand_id",
    "sku_id",
    "target_quantity",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "zero_rate_28",
    "observed_days_7",
    "observed_days_28",
    "days_since_previous_observation",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "day_of_month",
    "day_of_year",
    "year",
    "is_weekend",
    "time_index",
    "lag_1_promotion",
    "promotion_rate_28",
    "promotion_observations_28",
    "target_promotion",
    "split",
]

# Model feature subsets (excluding IDs, target, and split)
FEATURE_COLUMNS_NO_TARGET_PROMO: List[str] = (
    HISTORICAL_DEMAND_LAGS
    + ROLLING_DEMAND_FEATURES
    + TIMING_FEATURES
    + CALENDAR_FEATURES
    + HISTORICAL_PROMOTION_FEATURES
)

FEATURE_COLUMNS_WITH_TARGET_PROMO: List[str] = (
    FEATURE_COLUMNS_NO_TARGET_PROMO + TARGET_PROMOTION_FEATURES
)

# Feature provenance mapping explaining source window and derivation
FEATURE_PROVENANCE: Dict[str, str] = {
    "lag_1": "quantity at immediately preceding observed date for the same SKU",
    "lag_7": "quantity at exact calendar date t-7 for the same SKU (NaN if unobserved)",
    "lag_14": "quantity at exact calendar date t-14 for the same SKU (NaN if unobserved)",
    "lag_28": "quantity at exact calendar date t-28 for the same SKU (NaN if unobserved)",
    "rolling_mean_7": "mean observed demand in trailing calendar window [t-7, t-1]",
    "rolling_mean_14": "mean observed demand in trailing calendar window [t-14, t-1]",
    "rolling_mean_28": "mean observed demand in trailing calendar window [t-28, t-1]",
    "rolling_std_7": "sample standard deviation in [t-7, t-1] (NaN if fewer than 2 observations)",
    "rolling_std_28": "sample standard deviation in [t-28, t-1] (NaN if fewer than 2 observations)",
    "zero_rate_28": "proportion of zero-demand records in [t-28, t-1]",
    "observed_days_7": "count of actual observed trading days in [t-7, t-1]",
    "observed_days_28": "count of actual observed trading days in [t-28, t-1]",
    "days_since_previous_observation": "target_date - previous_observed_date in calendar days",
    "day_of_week": "day of week (Monday=0 ... Sunday=6)",
    "week_of_year": "ISO week number (1-53)",
    "month": "calendar month (1-12)",
    "quarter": "calendar quarter (1-4)",
    "day_of_month": "day of month (1-31)",
    "day_of_year": "day of year (1-366)",
    "year": "calendar year",
    "is_weekend": "weekend indicator (1 if Saturday/Sunday else 0)",
    "time_index": "elapsed calendar days from 2014-01-02",
    "lag_1_promotion": "promotion status on immediately preceding observed date for the same SKU",
    "promotion_rate_28": "proportion of observed promotion=1 records in [t-28, t-1]",
    "promotion_observations_28": "count of observed promotion records in [t-28, t-1]",
    "target_promotion": "promotion indicator at target date t",
}
