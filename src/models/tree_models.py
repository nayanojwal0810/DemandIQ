"""Tree-based forecasting model estimators and training utilities for DemandIQ.

Implements global XGBoost and LightGBM models with:
- Native categorical support for brand_id and sku_id
- Inner chronological early stopping on training period only (2014-01-02 to 2016-09-30 fit, 2016-10-01 to 2016-12-31 stopping)
- Full outer-training refit (2014-01-02 to 2016-12-31) using selected best_iteration
- Normalized gain-based feature importance extraction
- Non-negative prediction post-processing
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.feature_config import (
    FEATURE_COLUMNS_NO_TARGET_PROMO,
    FEATURE_COLUMNS_WITH_TARGET_PROMO,
)

logger = logging.getLogger("tree_models")

CATEGORICAL_COLUMNS: List[str] = ["brand_id", "sku_id"]

DEFAULT_XGBOOST_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "importance_type": "gain",
    "enable_categorical": True,
}

DEFAULT_LIGHTGBM_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass
class ModelConfig:
    """Configuration specification for a tree model experiment.

    Attributes
    ----------
    model_family : str
        "xgboost" or "lightgbm".
    feature_variant : str
        "no_target_promo" or "with_target_promo".
    hyperparameters : Dict[str, Any]
        Base hyperparameters.
    max_estimators : int, default 1000
        Upper bound on boosting trees for early stopping search.
    early_stopping_rounds : int, default 50
        Patience rounds on the inner validation split.
    random_state : int, default 42
        Deterministic random seed.
    """

    model_family: str
    feature_variant: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    max_estimators: int = 1000
    early_stopping_rounds: int = 50
    random_state: int = 42

    @property
    def config_name(self) -> str:
        """Return canonical model configuration identifier."""
        return f"{self.model_family}_{self.feature_variant}"


@dataclass
class TrainingResult:
    """Encapsulates fitted model artifacts, inner validation metrics, and feature importances.

    Attributes
    ----------
    config : ModelConfig
        The specification used for training.
    fitted_model : Any
        Fitted model object (retrained on full outer training data).
    best_iteration : int
        Selected number of boosting trees determined by inner early-stopping.
    best_inner_score : float
        Best MAE evaluation score achieved on the inner validation set.
    training_seconds : float
        Wall-clock duration encompassing inner tuning and outer refit.
    feature_names : List[str]
        Ordered list of feature column names used for training.
    feature_importances : pd.DataFrame
        Gain-based feature importance table with normalized shares and ranks.
    """

    config: ModelConfig
    fitted_model: Any
    best_iteration: int
    best_inner_score: float
    training_seconds: float
    feature_names: List[str]
    feature_importances: pd.DataFrame


def get_model_feature_columns(feature_variant: str) -> List[str]:
    """Retrieve canonical feature columns for the requested promotion variant.

    Parameters
    ----------
    feature_variant : str
        "no_target_promo" or "with_target_promo".

    Returns
    -------
    List[str]
        Ordered feature column list including categorical brand and SKU identities.
    """
    if feature_variant == "with_target_promo":
        base_features = FEATURE_COLUMNS_WITH_TARGET_PROMO
    elif feature_variant == "no_target_promo":
        base_features = FEATURE_COLUMNS_NO_TARGET_PROMO
    else:
        raise ValueError(
            f"Unknown feature variant '{feature_variant}'. Must be 'no_target_promo' or 'with_target_promo'."
        )

    return CATEGORICAL_COLUMNS + list(base_features)


def prepare_categorical_features(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Ensure categorical identity columns have pandas category dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cat_cols : List[str], optional
        Columns to convert. Defaults to ['brand_id', 'sku_id'].

    Returns
    -------
    pd.DataFrame
        Dataframe with specified columns converted to category dtype.
    """
    if cat_cols is None:
        cat_cols = CATEGORICAL_COLUMNS

    df_out = df.copy()
    for col in cat_cols:
        if col in df_out.columns and not isinstance(df_out[col].dtype, pd.CategoricalDtype):
            df_out[col] = df_out[col].astype("category")
    return df_out


def post_process_predictions(predictions: np.ndarray) -> np.ndarray:
    """Enforce non-negative demand constraint on forecast predictions.

    Converts predictions to float and clips at zero via np.maximum(y, 0.0).

    Parameters
    ----------
    predictions : np.ndarray
        Raw model forecast array.

    Returns
    -------
    np.ndarray
        Clipped non-negative forecast array.
    """
    preds = np.asarray(predictions, dtype=float)
    return np.maximum(preds, 0.0)


def build_xgboost_estimator(
    config: ModelConfig,
    n_estimators: int,
    early_stopping_rounds: Optional[int] = None,
    eval_metric: str = "mae",
) -> xgb.XGBRegressor:
    """Construct an XGBoost regressor instance with project hyperparameter defaults.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    n_estimators : int
        Number of boosting trees.
    early_stopping_rounds : int, optional
        Early stopping patience rounds.
    eval_metric : str, default "mae"
        Evaluation metric for early stopping.

    Returns
    -------
    xgb.XGBRegressor
        Configured XGBoost estimator.
    """
    params = DEFAULT_XGBOOST_PARAMS.copy()
    params.update(config.hyperparameters)
    params["n_estimators"] = n_estimators
    params["random_state"] = config.random_state

    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
        params["eval_metric"] = eval_metric

    return xgb.XGBRegressor(**params)


def build_lightgbm_estimator(
    config: ModelConfig,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    """Construct a LightGBM regressor instance with project hyperparameter defaults.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    n_estimators : int
        Number of boosting trees.

    Returns
    -------
    lgb.LGBMRegressor
        Configured LightGBM estimator.
    """
    params = DEFAULT_LIGHTGBM_PARAMS.copy()
    params.update(config.hyperparameters)
    params["n_estimators"] = n_estimators
    params["random_state"] = config.random_state

    return lgb.LGBMRegressor(**params)


def extract_gain_feature_importance(
    model: Any,
    model_family: str,
    feature_names: List[str],
    feature_variant: str,
) -> pd.DataFrame:
    """Extract gain-based feature importances and compute normalized shares.

    Parameters
    ----------
    model : Any
        Fitted XGBoost or LightGBM model.
    model_family : str
        "xgboost" or "lightgbm".
    feature_names : List[str]
        Ordered feature column names.
    feature_variant : str
        "no_target_promo" or "with_target_promo".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [model, feature_variant, feature, importance, importance_share, rank].
    """
    if model_family == "xgboost":
        # XGBRegressor with importance_type="gain" provides gain importance in feature_importances_
        raw_importances = model.feature_importances_
    elif model_family == "lightgbm":
        raw_importances = model.booster_.feature_importance(importance_type="gain")
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    raw_importances = np.asarray(raw_importances, dtype=float)
    total_gain = float(np.sum(raw_importances))

    if total_gain > 0.0:
        shares = raw_importances / total_gain
    else:
        shares = np.zeros_like(raw_importances)

    df_imp = pd.DataFrame(
        {
            "model": model_family,
            "feature_variant": feature_variant,
            "feature": feature_names,
            "importance": raw_importances,
            "importance_share": shares,
        }
    )

    # Sort descending by importance and assign rank (1-indexed)
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1, dtype=int)
    return df_imp


def train_model_with_inner_early_stopping(
    config: ModelConfig,
    X_inner_tr: pd.DataFrame,
    y_inner_tr: np.ndarray,
    X_inner_val: pd.DataFrame,
    y_inner_val: np.ndarray,
    X_outer_tr: pd.DataFrame,
    y_outer_tr: np.ndarray,
) -> TrainingResult:
    """Execute two-stage training workflow: inner early-stopping followed by full outer refit.

    Workflow:
    1. Fit model on inner training split with early stopping monitored on inner validation split.
    2. Determine best_iteration and record best inner validation MAE score.
    3. Refit fresh model on all outer training data (2014-01-02 to 2016-12-31) using best_iteration.
    4. Extract gain-based feature importances.

    Parameters
    ----------
    config : ModelConfig
        Model configuration specification.
    X_inner_tr : pd.DataFrame
        Inner fit feature matrix (2014-01-02 to 2016-09-30).
    y_inner_tr : np.ndarray
        Inner fit target vector.
    X_inner_val : pd.DataFrame
        Inner early-stopping validation feature matrix (2016-10-01 to 2016-12-31).
    y_inner_val : np.ndarray
        Inner early-stopping target vector.
    X_outer_tr : pd.DataFrame
        Full outer training feature matrix (2014-01-02 to 2016-12-31).
    y_outer_tr : np.ndarray
        Full outer training target vector.

    Returns
    -------
    TrainingResult
        Complete training outcome with fitted outer model and metadata.
    """
    t_start = time.time()
    feature_names = list(X_outer_tr.columns)

    logger.info(
        "Starting training for '%s' (features: %d)...",
        config.config_name,
        len(feature_names),
    )

    if config.model_family == "xgboost":
        # 1. Inner tuning with early stopping
        tune_model = build_xgboost_estimator(
            config,
            n_estimators=config.max_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
            eval_metric="mae",
        )
        tune_model.fit(
            X_inner_tr,
            y_inner_tr,
            eval_set=[(X_inner_val, y_inner_val)],
            verbose=False,
        )
        best_iter = int(tune_model.best_iteration)
        # Ensure at least 1 tree
        best_iter = max(1, best_iter)
        best_score = float(tune_model.best_score)

        logger.info(
            "XGBoost inner tuning completed: best_iteration=%d, best_inner_score=%.4f",
            best_iter,
            best_score,
        )

        # 2. Refit on full outer training data
        final_model = build_xgboost_estimator(config, n_estimators=best_iter)
        final_model.fit(X_outer_tr, y_outer_tr)

    elif config.model_family == "lightgbm":
        # 1. Inner tuning with early stopping
        tune_model = build_lightgbm_estimator(config, n_estimators=config.max_estimators)
        callbacks = [lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False)]
        tune_model.fit(
            X_inner_tr,
            y_inner_tr,
            eval_X=X_inner_val,
            eval_y=y_inner_val,
            eval_metric="mae",
            callbacks=callbacks,
        )
        best_iter = int(tune_model.best_iteration_)
        best_iter = max(1, best_iter)
        # Extract best inner score from eval results
        eval_results = tune_model.evals_result_ if hasattr(tune_model, "evals_result_") else {}
        if eval_results and "valid_0" in eval_results:
            best_score = float(min(eval_results["valid_0"].get("l1", [np.nan])))
        elif hasattr(tune_model, "best_score_") and "valid_0" in tune_model.best_score_:
            best_score = float(tune_model.best_score_["valid_0"]["l1"])
        else:
            best_score = float(np.nan)

        logger.info(
            "LightGBM inner tuning completed: best_iteration=%d, best_inner_score=%.4f",
            best_iter,
            best_score,
        )

        # 2. Refit on full outer training data
        final_model = build_lightgbm_estimator(config, n_estimators=best_iter)
        final_model.fit(X_outer_tr, y_outer_tr)

    else:
        raise ValueError(f"Unknown model family: {config.model_family}")

    duration = time.time() - t_start

    # Extract gain-based feature importances from the refit model
    feat_imp = extract_gain_feature_importance(
        final_model,
        config.model_family,
        feature_names,
        config.feature_variant,
    )

    logger.info(
        "Refit completed for '%s' in %.2f seconds.",
        config.config_name,
        duration,
    )

    return TrainingResult(
        config=config,
        fitted_model=final_model,
        best_iteration=best_iter,
        best_inner_score=best_score,
        training_seconds=duration,
        feature_names=feature_names,
        feature_importances=feat_imp,
    )
