"""Training and evaluation pipeline for DemandIQ XGBoost and LightGBM models.

Orchestrates:
1. Feature matrix preparation across promotion variants.
2. Chronological inner early stopping (2014-01-02 to 2016-09-30 fit, 2016-10-01 to 2016-12-31 early stopping).
3. Full outer training refit (2014-01-02 to 2016-12-31).
4. Generation of 2017 outer validation predictions (42,716 rows across 118 SKUs).
5. Evaluation of portfolio and SKU-level metrics using full precision.
6. Promotion delta analysis (predictive association, non-causal).
7. Feature importance and training metadata extraction.
8. Baseline comparison against Stage 7 classical benchmarks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.features.feature_config import (
    DEVELOPMENT_CUTOFF_DATE,
    TRAINING_END_DATE,
    TRAINING_START_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
)
from src.models.tree_models import (
    ModelConfig,
    TrainingResult,
    get_model_feature_columns,
    post_process_predictions,
    prepare_categorical_features,
    train_model_with_inner_early_stopping,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_tree_models")

UnionPath = Union[Path, str]

# Inner chronological split boundaries (strictly inside 2014-2016 training period)
INNER_FIT_START: str = "2014-01-02"
INNER_FIT_END: str = "2016-09-30"
INNER_VAL_START: str = "2016-10-01"
INNER_VAL_END: str = "2016-12-31"

MODEL_CONFIGS: List[Tuple[str, str]] = [
    ("xgboost", "no_target_promo"),
    ("xgboost", "with_target_promo"),
    ("lightgbm", "no_target_promo"),
    ("lightgbm", "with_target_promo"),
]


def load_development_feature_data(
    data_path: UnionPath = "data/processed/forecasting_features_development.csv",
) -> pd.DataFrame:
    """Load canonical feature table and enforce development boundaries.

    Parameters
    ----------
    data_path : str or Path
        Path to processed feature table.

    Returns
    -------
    pd.DataFrame
        Development-period feature DataFrame.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"date", "brand_id", "sku_id", "target_quantity", "split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in feature table: {required_cols - set(df.columns)}")

    # Strictly verify development cutoff: zero 2018 rows allowed
    max_date = df["date"].max()
    if max_date > DEVELOPMENT_CUTOFF_DATE:
        raise ValueError(
            f"Holdout leakage detected: maximum date in dataset is {max_date}, "
            f"exceeding development cutoff {DEVELOPMENT_CUTOFF_DATE}."
        )

    # Sort deterministically
    df = df.sort_values(["date", "brand_id", "sku_id"]).reset_index(drop=True)
    df = prepare_categorical_features(df)

    logger.info(
        "Loaded %d feature rows (SKUs: %d, dates: %s to %s).",
        len(df),
        df["sku_id"].nunique(),
        df["date"].min(),
        df["date"].max(),
    )
    return df


def split_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition feature table into inner-fit, inner-val, outer-train, and outer-val.

    Parameters
    ----------
    df : pd.DataFrame
        Full development feature table.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_inner_tr, df_inner_val, df_outer_tr, df_outer_val)
    """
    df_outer_tr = df[(df["date"] >= TRAINING_START_DATE) & (df["date"] <= TRAINING_END_DATE)].copy()
    df_outer_val = df[(df["date"] >= VALIDATION_START_DATE) & (df["date"] <= VALIDATION_END_DATE)].copy()

    df_inner_tr = df_outer_tr[(df_outer_tr["date"] >= INNER_FIT_START) & (df_outer_tr["date"] <= INNER_FIT_END)].copy()
    df_inner_val = df_outer_tr[(df_outer_tr["date"] >= INNER_VAL_START) & (df_outer_tr["date"] <= INNER_VAL_END)].copy()

    logger.info(
        "Data splits created: Outer Train=%d rows, Inner Fit=%d rows, Inner Val=%d rows, Outer Val=%d rows.",
        len(df_outer_tr),
        len(df_inner_tr),
        len(df_inner_val),
        len(df_outer_val),
    )
    return df_inner_tr, df_inner_val, df_outer_tr, df_outer_val


def train_all_tree_models(
    df_inner_tr: pd.DataFrame,
    df_inner_val: pd.DataFrame,
    df_outer_tr: pd.DataFrame,
) -> Dict[str, TrainingResult]:
    """Train all four model configurations using inner early stopping and outer refit.

    Parameters
    ----------
    df_inner_tr : pd.DataFrame
        Inner fit data.
    df_inner_val : pd.DataFrame
        Inner early-stopping validation data.
    df_outer_tr : pd.DataFrame
        Full outer training data.

    Returns
    -------
    Dict[str, TrainingResult]
        Mapping from config_name to TrainingResult.
    """
    y_inner_tr = df_inner_tr["target_quantity"].values.astype(float)
    y_inner_val = df_inner_val["target_quantity"].values.astype(float)
    y_outer_tr = df_outer_tr["target_quantity"].values.astype(float)

    results: Dict[str, TrainingResult] = {}

    for model_family, feature_variant in MODEL_CONFIGS:
        config = ModelConfig(
            model_family=model_family,
            feature_variant=feature_variant,
            early_stopping_rounds=50,
            random_state=42,
        )
        feature_cols = get_model_feature_columns(feature_variant)

        X_inner_tr = df_inner_tr[feature_cols]
        X_inner_val = df_inner_val[feature_cols]
        X_outer_tr = df_outer_tr[feature_cols]

        result = train_model_with_inner_early_stopping(
            config,
            X_inner_tr,
            y_inner_tr,
            X_inner_val,
            y_inner_val,
            X_outer_tr,
            y_outer_tr,
        )
        results[config.config_name] = result

    return results


def generate_ml_validation_predictions(
    models: Dict[str, TrainingResult],
    df_outer_val: pd.DataFrame,
) -> pd.DataFrame:
    """Generate 2017 outer-validation predictions for all four model configurations.

    Parameters
    ----------
    models : Dict[str, TrainingResult]
        Fitted model results.
    df_outer_val : pd.DataFrame
        Outer validation feature dataframe.

    Returns
    -------
    pd.DataFrame
        Table with actuals and predictions for all 4 ML configurations.
    """
    preds_df = pd.DataFrame(
        {
            "date": df_outer_val["date"].values,
            "brand_id": df_outer_val["brand_id"].astype(str).values,
            "sku_id": df_outer_val["sku_id"].astype(str).values,
            "actual": df_outer_val["target_quantity"].values.astype(float),
        }
    )

    for config_name, result in models.items():
        feature_cols = result.feature_names
        X_val = df_outer_val[feature_cols]
        raw_preds = result.fitted_model.predict(X_val)
        preds_df[config_name] = post_process_predictions(raw_preds)

    # Sort deterministically
    preds_df = preds_df.sort_values(["date", "brand_id", "sku_id"]).reset_index(drop=True)
    logger.info("Generated predictions for %d validation rows.", len(preds_df))
    return preds_df


def calculate_ml_portfolio_metrics(
    preds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate portfolio-level WAPE, MAE, and RMSE for all ML configurations.

    Internal metric values remain full-precision floats.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Predictions table.

    Returns
    -------
    pd.DataFrame
        Portfolio metrics DataFrame.
    """
    total_targets = len(preds_df)
    rows = []

    for model_family, feature_variant in MODEL_CONFIGS:
        col = f"{model_family}_{feature_variant}"
        actual = preds_df["actual"].values
        pred = preds_df[col].values
        count = int(np.sum(~np.isnan(pred)))
        cov = count / total_targets if total_targets > 0 else 0.0

        rows.append(
            {
                "model": model_family,
                "feature_variant": feature_variant,
                "valid_prediction_count": count,
                "total_validation_targets": total_targets,
                "coverage": cov,
                "wape": float(calculate_wape(actual, pred)),
                "mae": float(calculate_mae(actual, pred)),
                "rmse": float(calculate_rmse(actual, pred)),
            }
        )

    return pd.DataFrame(rows)


def calculate_ml_sku_metrics(
    preds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate SKU-level metrics for all four configurations using full precision.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Predictions table.

    Returns
    -------
    pd.DataFrame
        SKU metrics DataFrame.
    """
    rows = []

    for sku_id, g in preds_df.groupby("sku_id", sort=True):
        brand_id = g["brand_id"].iloc[0]
        sku_targets = len(g)
        actual = g["actual"].values

        for model_family, feature_variant in MODEL_CONFIGS:
            col = f"{model_family}_{feature_variant}"
            pred = g[col].values
            count = int(np.sum(~np.isnan(pred)))
            cov = count / sku_targets if sku_targets > 0 else 0.0

            rows.append(
                {
                    "brand_id": brand_id,
                    "sku_id": sku_id,
                    "model": model_family,
                    "feature_variant": feature_variant,
                    "valid_prediction_count": count,
                    "coverage": cov,
                    "wape": float(calculate_wape(actual, pred)),
                    "mae": float(calculate_mae(actual, pred)),
                    "rmse": float(calculate_rmse(actual, pred)),
                }
            )

    df_sku = pd.DataFrame(rows)
    df_sku = df_sku.sort_values(["brand_id", "sku_id", "model", "feature_variant"]).reset_index(drop=True)
    return df_sku


def calculate_ml_winner_summary(
    sku_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify winning ML configuration per SKU based on lowest full-precision WAPE.

    Deterministic tie-break order:
    1. xgboost_with_target_promo
    2. lightgbm_with_target_promo
    3. xgboost_no_target_promo
    4. lightgbm_no_target_promo

    Parameters
    ----------
    sku_metrics_df : pd.DataFrame
        SKU metrics table.

    Returns
    -------
    pd.DataFrame
        Winner summary table.
    """
    configs = [f"{m}_{v}" for m, v in MODEL_CONFIGS]
    tie_break_priority = {c: i for i, c in enumerate(configs)}
    wins: Dict[str, int] = {c: 0 for c in configs}

    unique_skus = sku_metrics_df["sku_id"].unique()
    total_skus = len(unique_skus)

    # Pivot WAPE per SKU
    for sku_id, g in sku_metrics_df.groupby("sku_id"):
        wapes = {}
        for _, row in g.iterrows():
            cfg_name = f"{row['model']}_{row['feature_variant']}"
            wapes[cfg_name] = float(row["wape"])

        min_w = min(wapes.values())
        candidates = [
            c for c, w in wapes.items()
            if w == min_w or np.isclose(w, min_w, rtol=1e-14, atol=1e-14)
        ]
        candidates.sort(key=lambda c: tie_break_priority[c])
        winner = candidates[0]
        wins[winner] += 1

    rows = []
    for model_family, feature_variant in MODEL_CONFIGS:
        cfg_name = f"{model_family}_{feature_variant}"
        cnt = wins[cfg_name]
        share = cnt / total_skus if total_skus > 0 else 0.0
        rows.append(
            {
                "model": model_family,
                "feature_variant": feature_variant,
                "sku_wins": cnt,
                "sku_win_share": share,
            }
        )

    return pd.DataFrame(rows)


def build_training_summary(
    results: Dict[str, TrainingResult],
    inner_tr_count: int,
    inner_val_count: int,
) -> pd.DataFrame:
    """Compile training metadata, hyperparameters, and inner validation scores.

    Parameters
    ----------
    results : Dict[str, TrainingResult]
        Fitted model outcomes.
    inner_tr_count : int
        Inner fit row count.
    inner_val_count : int
        Inner validation row count.

    Returns
    -------
    pd.DataFrame
        Training summary table.
    """
    rows = []
    for cfg_name, res in results.items():
        rows.append(
            {
                "model": res.config.model_family,
                "feature_variant": res.config.feature_variant,
                "outer_train_start": TRAINING_START_DATE,
                "outer_train_end": TRAINING_END_DATE,
                "inner_fit_start": INNER_FIT_START,
                "inner_fit_end": INNER_FIT_END,
                "inner_validation_start": INNER_VAL_START,
                "inner_validation_end": INNER_VAL_END,
                "training_rows": inner_tr_count,
                "inner_validation_rows": inner_val_count,
                "best_iteration": res.best_iteration,
                "best_inner_score": round(res.best_inner_score, 6),
                "random_state": res.config.random_state,
                "training_seconds": round(res.training_seconds, 2),
            }
        )
    return pd.DataFrame(rows)


def build_consolidated_feature_importance(
    results: Dict[str, TrainingResult],
) -> pd.DataFrame:
    """Concatenate feature importance tables across all four configurations.

    Parameters
    ----------
    results : Dict[str, TrainingResult]
        Fitted model results.

    Returns
    -------
    pd.DataFrame
        Consolidated feature importance table.
    """
    imp_dfs = [res.feature_importances for res in results.values()]
    all_imp = pd.concat(imp_dfs, ignore_index=True)
    return all_imp


def compute_promotion_comparison(
    portfolio_metrics: pd.DataFrame,
    sku_metrics: pd.DataFrame,
    tol: float = 1e-4,
) -> Dict[str, Dict[str, Any]]:
    """Compute descriptive promotion ablation comparison (with vs no target promo).

    Operational assumption: Target-date promotion status is known or planned
    at the time the next-day forecast is generated. This comparison reflects
    predictive association, not causal impact.

    Parameters
    ----------
    portfolio_metrics : pd.DataFrame
        Portfolio metrics table.
    sku_metrics : pd.DataFrame
        SKU-level metrics table.
    tol : float, default 1e-4
        Tolerance for classifying an SKU's WAPE as unchanged.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Comparison statistics per model family.
    """
    out = {}
    for model_family in ["xgboost", "lightgbm"]:
        m_port = portfolio_metrics[portfolio_metrics["model"] == model_family]
        wape_no = float(m_port[m_port["feature_variant"] == "no_target_promo"]["wape"].iloc[0])
        wape_with = float(m_port[m_port["feature_variant"] == "with_target_promo"]["wape"].iloc[0])

        abs_diff = wape_with - wape_no
        rel_change = (abs_diff / wape_no) if wape_no > 0 else 0.0

        # SKU level comparison
        m_sku = sku_metrics[sku_metrics["model"] == model_family]
        piv = m_sku.pivot(index="sku_id", columns="feature_variant", values="wape")
        diff = piv["with_target_promo"] - piv["no_target_promo"]

        improved = int((diff < -tol).sum())
        worsened = int((diff > tol).sum())
        unchanged = int((np.abs(diff) <= tol).sum())

        out[model_family] = {
            "wape_no_promo": wape_no,
            "wape_with_promo": wape_with,
            "absolute_wape_diff": abs_diff,
            "relative_wape_change": rel_change,
            "skus_improved": improved,
            "skus_worsened": worsened,
            "skus_unchanged": unchanged,
        }
    return out


def save_ml_deliverables(
    preds_df: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    sku_metrics: pd.DataFrame,
    winner_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_dir: UnionPath = "data/processed",
) -> None:
    """Persist all required Stage 8 CSV deliverables deterministically.

    Output files:
    - data/processed/ml_predictions_validation.csv
    - data/processed/ml_metrics.csv
    - data/processed/ml_sku_metrics.csv
    - data/processed/ml_winner_summary.csv
    - data/processed/ml_training_summary.csv
    - data/processed/ml_feature_importance.csv

    Parameters
    ----------
    preds_df : pd.DataFrame
        Predictions table.
    portfolio_metrics : pd.DataFrame
        Portfolio metrics table.
    sku_metrics : pd.DataFrame
        SKU metrics table.
    winner_summary : pd.DataFrame
        Winner summary table.
    training_summary : pd.DataFrame
        Training metadata table.
    feature_importance : pd.DataFrame
        Feature importance table.
    output_dir : str or Path
        Target directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Round presentation columns for clean CSV persistence
    port_csv = portfolio_metrics.copy()
    for col in ["coverage", "wape", "mae", "rmse"]:
        if col in port_csv.columns:
            port_csv[col] = port_csv[col].round(6)

    sku_csv = sku_metrics.copy()
    for col in ["coverage", "wape", "mae", "rmse"]:
        if col in sku_csv.columns:
            sku_csv[col] = sku_csv[col].round(6)

    win_csv = winner_summary.copy()
    if "sku_win_share" in win_csv.columns:
        win_csv["sku_win_share"] = win_csv["sku_win_share"].round(6)

    imp_csv = feature_importance.copy()
    if "importance" in imp_csv.columns:
        imp_csv["importance"] = imp_csv["importance"].round(4)
    if "importance_share" in imp_csv.columns:
        imp_csv["importance_share"] = imp_csv["importance_share"].round(6)

    preds_df.to_csv(out_path / "ml_predictions_validation.csv", index=False)
    port_csv.to_csv(out_path / "ml_metrics.csv", index=False)
    sku_csv.to_csv(out_path / "ml_sku_metrics.csv", index=False)
    win_csv.to_csv(out_path / "ml_winner_summary.csv", index=False)
    training_summary.to_csv(out_path / "ml_training_summary.csv", index=False)
    imp_csv.to_csv(out_path / "ml_feature_importance.csv", index=False)

    logger.info("Saved all Stage 8 ML deliverables to %s", out_path)


def main() -> None:
    """Execute complete XGBoost & LightGBM forecasting workflow."""
    logger.info("=== DemandIQ Stage 8: XGBoost & LightGBM Global Forecasting ===")
    df = load_development_feature_data()
    df_inner_tr, df_inner_val, df_outer_tr, df_outer_val = split_training_data(df)

    # Train all 4 model configurations
    results = train_all_tree_models(df_inner_tr, df_inner_val, df_outer_tr)

    # Generate 2017 validation predictions
    preds_df = generate_ml_validation_predictions(results, df_outer_val)

    # Calculate metrics
    portfolio_metrics = calculate_ml_portfolio_metrics(preds_df)
    sku_metrics = calculate_ml_sku_metrics(preds_df)
    winner_summary = calculate_ml_winner_summary(sku_metrics)
    training_summary = build_training_summary(results, len(df_inner_tr), len(df_inner_val))
    feature_importance = build_consolidated_feature_importance(results)

    # Save deliverables
    save_ml_deliverables(
        preds_df,
        portfolio_metrics,
        sku_metrics,
        winner_summary,
        training_summary,
        feature_importance,
    )

    # Promotion ablation analysis
    promo_comp = compute_promotion_comparison(portfolio_metrics, sku_metrics)

    # Terminal summary display
    print("\n" + "=" * 80)
    print("STAGE 8: XGBOOST & LIGHTGBM PORTFOLIO METRICS (2017 VALIDATION)")
    print("=" * 80)
    port_display = portfolio_metrics.copy()
    for col in ["coverage", "wape", "mae", "rmse"]:
        port_display[col] = port_display[col].round(6)
    print(port_display.to_string(index=False))

    print("\n" + "=" * 80)
    print("STAGE 8: SKU WINS SUMMARY (AMONG 4 ML CONFIGURATIONS)")
    print("=" * 80)
    win_display = winner_summary.copy()
    win_display["sku_win_share"] = (win_display["sku_win_share"] * 100).round(2).astype(str) + "%"
    print(win_display.to_string(index=False))

    print("\n" + "=" * 80)
    print("PROMOTION ABLATION COMPARISON (WITH vs NO TARGET PROMOTION)")
    print("=" * 80)
    for model_family, stats in promo_comp.items():
        print(f"\nModel Family: {model_family.upper()}")
        print(f"  No Target Promotion WAPE:   {stats['wape_no_promo']:.6f}")
        print(f"  With Target Promotion WAPE: {stats['wape_with_promo']:.6f}")
        print(f"  Absolute Difference:        {stats['absolute_wape_diff']:.6f}")
        print(f"  Relative Change:            {stats['relative_wape_change'] * 100:.2f}%")
        print(f"  SKUs Improved:              {stats['skus_improved']} / 118")
        print(f"  SKUs Worsened:              {stats['skus_worsened']} / 118")
        print(f"  SKUs Unchanged:             {stats['skus_unchanged']} / 118")

    # Baseline comparison summary
    baseline_path = Path("data/processed/baseline_metrics.csv")
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        common_base = df_base[df_base["evaluation_scope"] == "common_valid"].copy()
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON: ML MODELS vs STAGE 7 CLASSICAL BASELINES")
        print("=" * 80)
        print("Baselines (common_valid):")
        for _, b_row in common_base.iterrows():
            print(f"  {b_row['baseline']:25s} | WAPE: {b_row['wape']:.6f} | MAE: {b_row['mae']:.4f} | RMSE: {b_row['rmse']:.4f}")
        print("\nML Configurations (100% coverage):")
        for _, m_row in portfolio_metrics.iterrows():
            cfg_label = f"{m_row['model']}_{m_row['feature_variant']}"
            print(f"  {cfg_label:25s} | WAPE: {m_row['wape']:.6f} | MAE: {m_row['mae']:.4f} | RMSE: {m_row['rmse']:.4f}")

    print("\n" + "=" * 80)
    print("STAGE 8 WORKSTREAM EXECUTION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
