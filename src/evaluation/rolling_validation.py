"""Expanding-window rolling-origin validation pipeline for DemandIQ Stage 9.

This module evaluates the temporal robustness of frozen Stage 8 ML configurations
alongside Stage 7 classical baselines across eight chronological quarterly folds
spanning 2015-01-01 through 2016-12-31.

Critical Methodological Rules:
1. Frozen Stage 8 Configurations:
   Tree counts, hyperparameters, and feature definitions are strictly frozen from Stage 8.
   No hyperparameter tuning, feature selection, or inner early-stopping is performed.
2. Temporal Isolation:
   Each fold maintains strict training < validation ordering.
   No shuffling, no random splits, no lookahead leakage.
3. Holdout Sealing:
   2018 remains completely sealed and untouched.
   2017 is excluded from rolling folds (used solely for post-hoc quarterly evaluation).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.features.feature_config import (
    DEVELOPMENT_CUTOFF_DATE,
    FEATURE_COLUMNS_NO_TARGET_PROMO,
    FEATURE_COLUMNS_WITH_TARGET_PROMO,
)
from src.forecasting.baselines import (
    fit_croston_sba_training_model,
    fit_ses_training_model,
    generate_croston_sba_validation_forecast,
    generate_moving_average_forecast,
    generate_naive_forecast,
    generate_seasonal_naive_forecast,
    generate_ses_validation_forecast,
)
from src.models.tree_models import (
    ModelConfig,
    build_lightgbm_estimator,
    build_xgboost_estimator,
    get_model_feature_columns,
    post_process_predictions,
    prepare_categorical_features,
)

logger = logging.getLogger("rolling_validation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

UnionPath = Union[Path, str]


@dataclass(frozen=True)
class RollingFold:
    """Specification of an expanding-window chronological fold.

    Attributes
    ----------
    fold_id : int
        1-indexed fold identifier (1 through 8).
    train_start : str
        Start date of expanding training window (inclusive).
    train_end : str
        End date of expanding training window (inclusive).
    val_start : str
        Start date of quarterly validation window (inclusive).
    val_end : str
        End date of quarterly validation window (inclusive).
    """

    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str


# Canonical 8 quarterly expanding-window folds approved for Stage 9
ROLLING_FOLDS: List[RollingFold] = [
    RollingFold(1, "2014-01-02", "2014-12-31", "2015-01-01", "2015-03-31"),
    RollingFold(2, "2014-01-02", "2015-03-31", "2015-04-01", "2015-06-30"),
    RollingFold(3, "2014-01-02", "2015-06-30", "2015-07-01", "2015-09-30"),
    RollingFold(4, "2014-01-02", "2015-09-30", "2015-10-01", "2015-12-31"),
    RollingFold(5, "2014-01-02", "2015-12-31", "2016-01-01", "2016-03-31"),
    RollingFold(6, "2014-01-02", "2016-03-31", "2016-04-01", "2016-06-30"),
    RollingFold(7, "2014-01-02", "2016-06-30", "2016-07-01", "2016-09-30"),
    RollingFold(8, "2014-01-02", "2016-09-30", "2016-10-01", "2016-12-31"),
]

# Frozen tree counts determined by Stage 8 inner early-stopping
DEFAULT_FROZEN_TREE_COUNTS: Dict[str, int] = {
    "xgboost_no_target_promo": 167,
    "xgboost_with_target_promo": 231,
    "lightgbm_no_target_promo": 124,
    "lightgbm_with_target_promo": 162,
}

# The four frozen ML configuration specifications
ML_CONFIG_SPECS: Dict[str, Tuple[str, str]] = {
    "xgboost_no_target_promo": ("xgboost", "no_target_promo"),
    "xgboost_with_target_promo": ("xgboost", "with_target_promo"),
    "lightgbm_no_target_promo": ("lightgbm", "no_target_promo"),
    "lightgbm_with_target_promo": ("lightgbm", "with_target_promo"),
}

# The five classical baseline names
CLASSICAL_BASELINES: List[str] = [
    "naive",
    "seasonal_naive_7",
    "moving_average_7d",
    "ets_ses",
    "croston_sba",
]

# Output column schema for rolling predictions
ROLLING_PREDICTION_COLUMNS: List[str] = [
    "fold",
    "date",
    "brand_id",
    "sku_id",
    "actual",
    "naive",
    "seasonal_naive_7",
    "moving_average_7d",
    "ets_ses",
    "croston_sba",
    "xgboost_no_target_promo",
    "xgboost_with_target_promo",
    "lightgbm_no_target_promo",
    "lightgbm_with_target_promo",
]


def load_frozen_tree_counts(
    summary_path: UnionPath = "data/processed/ml_training_summary.csv",
) -> Dict[str, int]:
    """Retrieve frozen tree counts from Stage 8 training summary, with fallback.

    Parameters
    ----------
    summary_path : UnionPath, default "data/processed/ml_training_summary.csv"
        Path to Stage 8 training summary CSV artifact.

    Returns
    -------
    Dict[str, int]
        Mapping from configuration name to frozen selected_n_estimators.
    """
    path = Path(summary_path)
    if path.exists():
        try:
            df_sum = pd.read_csv(path)
            counts = {}
            for _, row in df_sum.iterrows():
                key = f"{row['model']}_{row['feature_variant']}"
                if "selected_n_estimators" in row and pd.notna(row["selected_n_estimators"]):
                    counts[key] = int(row["selected_n_estimators"])
                elif "best_iteration" in row and pd.notna(row["best_iteration"]):
                    # Fallback conversion if selected_n_estimators column is missing
                    b_iter = int(row["best_iteration"])
                    counts[key] = b_iter + 1 if row["model"] == "xgboost" else b_iter

            if all(k in counts for k in DEFAULT_FROZEN_TREE_COUNTS):
                logger.info("Loaded frozen tree counts from %s: %s", path, counts)
                return counts
        except Exception as exc:
            logger.warning("Could not parse %s (%s); using default counts.", path, exc)

    logger.info("Using verified Stage 8 default frozen tree counts: %s", DEFAULT_FROZEN_TREE_COUNTS)
    return DEFAULT_FROZEN_TREE_COUNTS.copy()


def load_development_feature_table(
    feature_path: UnionPath = "data/processed/forecasting_features_development.csv",
) -> pd.DataFrame:
    """Load canonical development feature dataset and enforce temporal safeguards.

    Parameters
    ----------
    feature_path : UnionPath
        Path to canonical feature table CSV.

    Returns
    -------
    pd.DataFrame
        Validated development feature dataset.
    """
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"date", "brand_id", "sku_id", "target_quantity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: {required_cols - set(df.columns)}")

    # Enforce strictly no 2018 holdout observations
    cutoff = pd.Timestamp(DEVELOPMENT_CUTOFF_DATE)
    df["date_dt"] = pd.to_datetime(df["date"])
    if (df["date_dt"] > cutoff).any():
        raise ValueError(f"Holdout leakage detected: dates exceed {DEVELOPMENT_CUTOFF_DATE}")

    df["target_quantity"] = df["target_quantity"].astype(float)
    df = df.sort_values(["sku_id", "date_dt"]).reset_index(drop=True)
    logger.info("Loaded %d feature rows across %d SKUs.", len(df), df["sku_id"].nunique())
    return df


def generate_fold_classical_baselines(
    df_history: pd.DataFrame,
    fold: RollingFold,
    sku_demand_map: Dict[Tuple[str, pd.Timestamp], float],
) -> pd.DataFrame:
    """Generate 1-step ahead classical baselines for a single validation fold.

    Parameters
    ----------
    df_history : pd.DataFrame
        Combined train + val observations for the fold (sorted by sku_id, date_dt).
    fold : RollingFold
        Fold boundary specification.
    sku_demand_map : Dict[Tuple[str, pd.Timestamp], float]
        Dictionary mapping (sku_id, date_dt) -> quantity across available history.

    Returns
    -------
    pd.DataFrame
        Validation rows with baseline forecast columns.
    """
    train_end_dt = pd.Timestamp(fold.train_end)
    val_start_dt = pd.Timestamp(fold.val_start)
    val_end_dt = pd.Timestamp(fold.val_end)

    val_records: List[pd.DataFrame] = []

    for sku_id, g in df_history.groupby("sku_id", sort=True):
        g = g.sort_values("date_dt").reset_index(drop=True)
        g_train = g[g["date_dt"] <= train_end_dt]
        val_mask = (g["date_dt"] >= val_start_dt) & (g["date_dt"] <= val_end_dt)
        g_val = g[val_mask].copy()

        if len(g_val) == 0:
            continue

        val_indices = g_val.index

        # 1. Naive: shift(1) across observed sequence up to target
        g["naive"] = generate_naive_forecast(g["target_quantity"])

        # 2. Seasonal Naive 7: exact calendar date t - 7 lookup
        g["seasonal_naive_7"] = generate_seasonal_naive_forecast(
            g["date_dt"], sku_demand_map, sku_id
        )

        # 3. Moving Average 7D: exact trailing window [t - 7D, t - 1D]
        g["moving_average_7d"] = generate_moving_average_forecast(
            g["date_dt"], g["target_quantity"], window="7D", min_periods=1
        )

        # 4. Simple Exponential Smoothing: alpha fit strictly on fold train
        y_train = g_train["target_quantity"].values.astype(float)
        alpha_ses, init_level_ses = fit_ses_training_model(y_train)
        y_val = g_val["target_quantity"].values.astype(float)
        ses_preds = generate_ses_validation_forecast(y_val, alpha_ses, init_level_ses)

        # 5. Croston SBA: fit state strictly on fold train with fixed alpha=0.1
        croston_state = fit_croston_sba_training_model(y_train, alpha=0.1)
        croston_preds = generate_croston_sba_validation_forecast(
            y_val, croston_state, start_pos=len(y_train), alpha=0.1
        )

        sku_val = g.loc[val_indices].copy()
        sku_val["ets_ses"] = ses_preds
        sku_val["croston_sba"] = croston_preds

        val_records.append(sku_val)

    if not val_records:
        raise ValueError(f"No validation records generated for fold {fold.fold_id}.")

    return pd.concat(val_records, ignore_index=True)


def train_and_predict_fold_ml_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    frozen_tree_counts: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Train fresh models for the 4 frozen ML configurations on fold train and predict on fold val.

    No early stopping is used; each model refits with the frozen selected_n_estimators.
    Predictions are clamped at zero via post_process_predictions.

    Parameters
    ----------
    df_train : pd.DataFrame
        Fold training feature records.
    df_val : pd.DataFrame
        Fold validation feature records.
    frozen_tree_counts : Dict[str, int]
        Frozen tree counts per configuration.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from configuration name to post-processed forecast vector.
    """
    X_train_cat = prepare_categorical_features(df_train)
    X_val_cat = prepare_categorical_features(df_val)
    y_train = df_train["target_quantity"].values.astype(float)

    ml_predictions: Dict[str, np.ndarray] = {}

    for config_name, (family, variant) in ML_CONFIG_SPECS.items():
        feature_cols = get_model_feature_columns(variant)
        n_trees = frozen_tree_counts[config_name]
        cfg = ModelConfig(model_family=family, feature_variant=variant, random_state=42)

        X_tr = X_train_cat[feature_cols]
        X_v = X_val_cat[feature_cols]

        if family == "xgboost":
            model = build_xgboost_estimator(cfg, n_estimators=n_trees)
        elif family == "lightgbm":
            model = build_lightgbm_estimator(cfg, n_estimators=n_trees)
        else:
            raise ValueError(f"Unsupported model family: {family}")

        model.fit(X_tr, y_train)
        raw_preds = model.predict(X_v)
        post_preds = post_process_predictions(raw_preds)
        ml_predictions[config_name] = post_preds

    return ml_predictions


def execute_single_rolling_fold(
    df: pd.DataFrame,
    fold: RollingFold,
    frozen_tree_counts: Dict[str, int],
    sku_demand_map: Dict[Tuple[str, pd.Timestamp], float],
) -> pd.DataFrame:
    """Execute evaluation for one chronological fold across all 9 forecasting methods.

    Parameters
    ----------
    df : pd.DataFrame
        Development feature dataset.
    fold : RollingFold
        Fold definition.
    frozen_tree_counts : Dict[str, int]
        Frozen tree counts per configuration.
    sku_demand_map : Dict[Tuple[str, pd.Timestamp], float]
        Precomputed SKU demand map for exact lag matching.

    Returns
    -------
    pd.DataFrame
        Fold prediction records matching ROLLING_PREDICTION_COLUMNS schema.
    """
    logger.info(
        "Executing Fold %d: Train %s..%s -> Val %s..%s",
        fold.fold_id,
        fold.train_start,
        fold.train_end,
        fold.val_start,
        fold.val_end,
    )

    df_train = df[(df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)].copy()
    df_val = df[(df["date"] >= fold.val_start) & (df["date"] <= fold.val_end)].copy()
    df_history = df[(df["date"] >= fold.train_start) & (df["date"] <= fold.val_end)].copy()

    if len(df_train) == 0 or len(df_val) == 0:
        raise ValueError(f"Fold {fold.fold_id} has empty train ({len(df_train)}) or val ({len(df_val)}) split.")

    # 1. Classical Baselines
    val_baselines = generate_fold_classical_baselines(df_history, fold, sku_demand_map)

    # 2. Machine Learning Models
    ml_preds = train_and_predict_fold_ml_models(df_train, df_val, frozen_tree_counts)

    # Align ML predictions to df_val order
    for cfg_name, preds in ml_preds.items():
        df_val[cfg_name] = preds

    # Merge baselines into df_val on (date, brand_id, sku_id)
    baseline_cols = ["date", "brand_id", "sku_id"] + CLASSICAL_BASELINES
    merged = pd.merge(
        df_val,
        val_baselines[baseline_cols],
        on=["date", "brand_id", "sku_id"],
        how="inner",
    )

    merged["fold"] = fold.fold_id
    merged["actual"] = merged["target_quantity"]

    # Select and order required columns
    out_df = merged[ROLLING_PREDICTION_COLUMNS].copy()
    out_df = out_df.sort_values(["fold", "date", "brand_id", "sku_id"]).reset_index(drop=True)

    logger.info(
        "Fold %d completed: %d validation targets, %d SKUs.",
        fold.fold_id,
        len(out_df),
        out_df["sku_id"].nunique(),
    )
    return out_df


def run_rolling_validation_pipeline(
    feature_path: UnionPath = "data/processed/forecasting_features_development.csv",
    summary_path: UnionPath = "data/processed/ml_training_summary.csv",
    output_dir: UnionPath = "data/processed",
) -> pd.DataFrame:
    """Run full expanding-window rolling validation backtest across all 8 folds.

    Parameters
    ----------
    feature_path : UnionPath
        Path to development feature table CSV.
    summary_path : UnionPath
        Path to Stage 8 training summary CSV.
    output_dir : UnionPath
        Directory to write rolling_validation_predictions.csv.

    Returns
    -------
    pd.DataFrame
        Complete rolling validation predictions table.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_development_feature_table(feature_path)
    frozen_tree_counts = load_frozen_tree_counts(summary_path)

    # Precompute SKU demand lookup map across available development history
    sku_demand_map: Dict[Tuple[str, pd.Timestamp], float] = df.set_index(
        ["sku_id", "date_dt"]
    )["target_quantity"].to_dict()

    fold_dfs: List[pd.DataFrame] = []
    for fold in ROLLING_FOLDS:
        fold_df = execute_single_rolling_fold(df, fold, frozen_tree_counts, sku_demand_map)
        fold_dfs.append(fold_df)

    all_preds = pd.concat(fold_dfs, ignore_index=True)
    all_preds = all_preds.sort_values(["fold", "date", "brand_id", "sku_id"]).reset_index(drop=True)

    # Post-validation integrity checks
    assert len(all_preds) == 85078, f"Expected 85,078 total validation rows, got {len(all_preds)}"
    assert all_preds["fold"].nunique() == 8, f"Expected 8 folds, got {all_preds['fold'].nunique()}"
    assert not (all_preds["date"] >= "2017-01-01").any(), "Found 2017 dates in rolling predictions"
    assert not (all_preds["date"] >= "2018-01-01").any(), "Found 2018 dates in rolling predictions"

    # Save deliverable
    preds_csv = out_path / "rolling_validation_predictions.csv"
    all_preds.to_csv(preds_csv, index=False)
    logger.info("Saved rolling validation predictions to %s (%d rows).", preds_csv, len(all_preds))

    return all_preds


if __name__ == "__main__":
    from src.evaluation.robustness_analysis import run_robustness_analysis_pipeline

    logger.info("=== DemandIQ Stage 9: Rolling Model Robustness Pipeline ===")
    predictions_df = run_rolling_validation_pipeline()
    logger.info("=== Running Robustness Analysis & Generating Summaries ===")
    run_robustness_analysis_pipeline(predictions_df)
    logger.info("=== Stage 9 Robustness Workstream Complete ===")
