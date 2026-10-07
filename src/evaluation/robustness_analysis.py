"""Robustness analysis and post-hoc evaluation module for DemandIQ Stage 9.

Computes:
1. Rolling validation metrics across model_valid and common_valid evaluation scopes.
2. Model rank stability, fold wins, and pooled summary tables across the 8 rolling folds.
3. Brand-level and SKU-level performance distributions.
4. Deterministic model robustness flags.
5. 2017 within-year quarterly stability metrics from frozen Stage 8 predictions.
6. Publication-grade figures illustrating fold-level WAPE and rank trajectories.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.visualization.plot_utils import apply_clean_layout, save_figure

logger = logging.getLogger("robustness_analysis")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

UnionPath = Union[Path, str]

MODELS_ALL: List[str] = [
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

ML_MODELS: List[str] = [
    "xgboost_no_target_promo",
    "xgboost_with_target_promo",
    "lightgbm_no_target_promo",
    "lightgbm_with_target_promo",
]

CLASSICAL_MODELS: List[str] = [
    "naive",
    "seasonal_naive_7",
    "moving_average_7d",
    "ets_ses",
    "croston_sba",
]


def calculate_fold_metrics(df_preds: pd.DataFrame) -> pd.DataFrame:
    """Calculate fold-level metrics under model_valid and common_valid scopes.

    Parameters
    ----------
    df_preds : pd.DataFrame
        Complete rolling validation predictions table (8 folds).

    Returns
    -------
    pd.DataFrame
        Fold metrics table with columns:
        [fold, evaluation_scope, model, valid_prediction_count, total_validation_targets, coverage, wape, mae, rmse].
    """
    rows = []
    folds = sorted(df_preds["fold"].unique())

    for f_id in folds:
        f_df = df_preds[df_preds["fold"] == f_id].copy()
        n_targets = len(f_df)
        common_mask = f_df[MODELS_ALL].notna().all(axis=1)
        f_common = f_df[common_mask]
        n_common = len(f_common)

        for model in MODELS_ALL:
            # 1. model_valid scope
            valid_mask = f_df[model].notna()
            v_actual = f_df.loc[valid_mask, "actual"].values
            v_pred = f_df.loc[valid_mask, model].values
            v_count = len(v_actual)
            v_cov = v_count / n_targets if n_targets > 0 else 0.0

            rows.append(
                {
                    "fold": f_id,
                    "evaluation_scope": "model_valid",
                    "model": model,
                    "valid_prediction_count": v_count,
                    "total_validation_targets": n_targets,
                    "coverage": round(v_cov, 6),
                    "wape": round(calculate_wape(v_actual, v_pred), 6),
                    "mae": round(calculate_mae(v_actual, v_pred), 6),
                    "rmse": round(calculate_rmse(v_actual, v_pred), 6),
                }
            )

            # 2. common_valid scope
            c_actual = f_common["actual"].values
            c_pred = f_common[model].values
            c_cov = n_common / n_targets if n_targets > 0 else 0.0

            rows.append(
                {
                    "fold": f_id,
                    "evaluation_scope": "common_valid",
                    "model": model,
                    "valid_prediction_count": n_common,
                    "total_validation_targets": n_targets,
                    "coverage": round(c_cov, 6),
                    "wape": round(calculate_wape(c_actual, c_pred), 6),
                    "mae": round(calculate_mae(c_actual, c_pred), 6),
                    "rmse": round(calculate_rmse(c_actual, c_pred), 6),
                }
            )

    return pd.DataFrame(rows)


def calculate_rolling_summary(
    df_preds: pd.DataFrame,
    df_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pooled and fold-distribution metrics, fold wins, and rank stability.

    Fold wins and rank stability are evaluated strictly on the common_valid population.

    Parameters
    ----------
    df_preds : pd.DataFrame
        Complete rolling validation predictions table.
    df_metrics : pd.DataFrame
        Fold metrics table.

    Returns
    -------
    pd.DataFrame
        Rolling validation summary table matching required schema.
    """
    # 1. Rank models within each fold on common_valid scope
    common_metrics = df_metrics[df_metrics["evaluation_scope"] == "common_valid"].copy()
    common_metrics["rank"] = common_metrics.groupby("fold")["wape"].rank(method="min", ascending=True)

    # 2. Pooled metrics on common_valid across all 8 folds combined
    common_all_mask = df_preds[MODELS_ALL].notna().all(axis=1)
    df_common_all = df_preds[common_all_mask].copy()

    summary_rows = []
    n_folds = df_preds["fold"].nunique()

    for model in MODELS_ALL:
        m_common_metrics = common_metrics[common_metrics["model"] == model]
        fold_wapes = m_common_metrics["wape"].values
        fold_ranks = m_common_metrics["rank"].values

        fold_wins = int((fold_ranks == 1.0).sum())
        fold_win_share = fold_wins / n_folds

        actual_all = df_common_all["actual"].values
        pred_all = df_common_all[model].values

        pooled_wape = calculate_wape(actual_all, pred_all)
        pooled_mae = calculate_mae(actual_all, pred_all)
        pooled_rmse = calculate_rmse(actual_all, pred_all)

        summary_rows.append(
            {
                "model": model,
                "pooled_wape": round(pooled_wape, 6),
                "pooled_mae": round(pooled_mae, 6),
                "pooled_rmse": round(pooled_rmse, 6),
                "mean_fold_wape": round(float(np.mean(fold_wapes)), 6),
                "median_fold_wape": round(float(np.median(fold_wapes)), 6),
                "std_fold_wape": round(float(np.std(fold_wapes, ddof=1)), 6),
                "min_fold_wape": round(float(np.min(fold_wapes)), 6),
                "max_fold_wape": round(float(np.max(fold_wapes)), 6),
                "fold_wins": fold_wins,
                "fold_win_share": round(fold_win_share, 4),
                "mean_rank": round(float(np.mean(fold_ranks)), 2),
                "median_rank": round(float(np.median(fold_ranks)), 2),
            }
        )

    # Sort ascending by pooled_wape
    summary_df = pd.DataFrame(summary_rows).sort_values("pooled_wape").reset_index(drop=True)
    return summary_df


def calculate_brand_metrics(df_preds: pd.DataFrame) -> pd.DataFrame:
    """Calculate brand-segmented metrics across all rolling validation targets.

    Parameters
    ----------
    df_preds : pd.DataFrame
        Complete rolling validation predictions table.

    Returns
    -------
    pd.DataFrame
        Brand metrics table with columns:
        [brand_id, model, valid_prediction_count, coverage, wape, mae, rmse].
    """
    rows = []
    brands = sorted(df_preds["brand_id"].unique())

    for b_id in brands:
        b_df = df_preds[df_preds["brand_id"] == b_id]
        n_targets = len(b_df)

        for model in MODELS_ALL:
            valid_mask = b_df[model].notna()
            v_actual = b_df.loc[valid_mask, "actual"].values
            v_pred = b_df.loc[valid_mask, model].values
            v_count = len(v_actual)
            v_cov = v_count / n_targets if n_targets > 0 else 0.0

            rows.append(
                {
                    "brand_id": b_id,
                    "model": model,
                    "valid_prediction_count": v_count,
                    "coverage": round(v_cov, 6),
                    "wape": round(calculate_wape(v_actual, v_pred), 6),
                    "mae": round(calculate_mae(v_actual, v_pred), 6),
                    "rmse": round(calculate_rmse(v_actual, v_pred), 6),
                }
            )

    return pd.DataFrame(rows).sort_values(["brand_id", "wape"]).reset_index(drop=True)


def calculate_sku_metrics(df_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute lightweight SKU-level robustness distributions across eligible folds.

    Parameters
    ----------
    df_preds : pd.DataFrame
        Complete rolling validation predictions table.

    Returns
    -------
    pd.DataFrame
        SKU robustness table with columns:
        [sku_id, model, eligible_fold_count, mean_fold_wape, median_fold_wape, min_fold_wape, max_fold_wape].
    """
    rows = []
    skus = sorted(df_preds["sku_id"].unique())

    for s_id in skus:
        s_df = df_preds[df_preds["sku_id"] == s_id]

        for model in MODELS_ALL:
            fold_wapes = []
            for _, g in s_df.groupby("fold"):
                v_mask = g[model].notna()
                if v_mask.sum() > 0:
                    y_true = g.loc[v_mask, "actual"].values
                    y_hat = g.loc[v_mask, model].values
                    w = calculate_wape(y_true, y_hat)
                    if pd.notna(w):
                        fold_wapes.append(w)

            if fold_wapes:
                rows.append(
                    {
                        "sku_id": s_id,
                        "model": model,
                        "eligible_fold_count": len(fold_wapes),
                        "mean_fold_wape": round(float(np.mean(fold_wapes)), 6),
                        "median_fold_wape": round(float(np.median(fold_wapes)), 6),
                        "min_fold_wape": round(float(np.min(fold_wapes)), 6),
                        "max_fold_wape": round(float(np.max(fold_wapes)), 6),
                    }
                )

    return pd.DataFrame(rows).sort_values(["sku_id", "model"]).reset_index(drop=True)


def assign_robustness_flag(
    fold_wins: int,
    mean_rank: float,
    std_fold_wape: float,
    median_std_wape: float,
) -> str:
    """Assign deterministic robustness flag based on fold wins, rank, and variability.

    Flags:
    - frequent_fold_win: Wins >= 4 folds (50%+ of rolling periods)
    - high_variability: Standard deviation of fold WAPE noticeably exceeds median variability
    - frequent_fold_loss: Mean rank >= 7.0 among 9 evaluated candidates
    - stable_wape: Consistent performance across periods without sharp degradation

    Parameters
    ----------
    fold_wins : int
        Number of fold wins.
    mean_rank : float
        Mean rank across folds.
    std_fold_wape : float
        Standard deviation of fold WAPEs.
    median_std_wape : float
        Median standard deviation across all candidates.

    Returns
    -------
    str
        Assigned robustness flag.
    """
    if fold_wins >= 4:
        return "frequent_fold_win"
    elif std_fold_wape > (median_std_wape * 1.25):
        return "high_variability"
    elif mean_rank >= 7.0:
        return "frequent_fold_loss"
    else:
        return "stable_wape"


def calculate_model_robustness_summary(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Build concise model robustness summary table with deterministic stability flags.

    Parameters
    ----------
    df_summary : pd.DataFrame
        Rolling validation summary table.

    Returns
    -------
    pd.DataFrame
        Model robustness summary table.
    """
    median_std = float(df_summary["std_fold_wape"].median())

    rows = []
    for _, r in df_summary.iterrows():
        flag = assign_robustness_flag(
            fold_wins=int(r["fold_wins"]),
            mean_rank=float(r["mean_rank"]),
            std_fold_wape=float(r["std_fold_wape"]),
            median_std_wape=median_std,
        )

        rows.append(
            {
                "model": r["model"],
                "folds_evaluated": 8,
                "fold_wins": int(r["fold_wins"]),
                "mean_rank": r["mean_rank"],
                "median_rank": r["median_rank"],
                "pooled_wape": r["pooled_wape"],
                "mean_fold_wape": r["mean_fold_wape"],
                "std_fold_wape": r["std_fold_wape"],
                "worst_fold_wape": r["max_fold_wape"],
                "best_fold_wape": r["min_fold_wape"],
                "robustness_flag": flag,
            }
        )

    return pd.DataFrame(rows).sort_values("pooled_wape").reset_index(drop=True)


def calculate_2017_quarterly_metrics(
    ml_preds_path: UnionPath = "data/processed/ml_predictions_validation.csv",
    baseline_preds_path: UnionPath = "data/processed/baseline_predictions_validation.csv",
) -> pd.DataFrame:
    """Calculate quarterly stability metrics for 2017 validation using frozen Stage 8 predictions.

    Does not retrain or retune models. Uses date calendar quarters:
    Q1: 2017-01-01 to 2017-03-31
    Q2: 2017-04-01 to 2017-06-30
    Q3: 2017-07-01 to 2017-09-30
    Q4: 2017-10-01 to 2017-12-31

    Parameters
    ----------
    ml_preds_path : UnionPath
        Path to Stage 8 frozen ML validation predictions CSV.
    baseline_preds_path : UnionPath
        Path to Stage 7 baseline predictions CSV (optional).

    Returns
    -------
    pd.DataFrame
        Quarterly metrics table with columns:
        [quarter, model, valid_prediction_count, wape, mae, rmse].
    """
    p_ml = Path(ml_preds_path)
    if not p_ml.exists():
        raise FileNotFoundError(f"ML predictions file not found: {p_ml}")

    df_ml = pd.read_csv(p_ml)

    # Optionally merge Stage 7 baseline predictions if available
    p_base = Path(baseline_preds_path)
    if p_base.exists():
        df_base = pd.read_csv(p_base)
        common_cols = [c for c in CLASSICAL_MODELS if c in df_base.columns]
        merge_cols = ["date", "sku_id"] + common_cols
        df = pd.merge(df_ml, df_base[merge_cols], on=["date", "sku_id"], how="left")
    else:
        df = df_ml.copy()

    df["date_dt"] = pd.to_datetime(df["date"])
    quarter_boundaries = [
        ("Q1", "2017-01-01", "2017-03-31"),
        ("Q2", "2017-04-01", "2017-06-30"),
        ("Q3", "2017-07-01", "2017-09-30"),
        ("Q4", "2017-10-01", "2017-12-31"),
    ]

    models_to_evaluate = [m for m in df.columns if m in MODELS_ALL]

    rows = []
    for q_label, q_start, q_end in quarter_boundaries:
        mask = (df["date"] >= q_start) & (df["date"] <= q_end)
        q_df = df[mask]

        for model in models_to_evaluate:
            v_mask = q_df[model].notna()
            v_actual = q_df.loc[v_mask, "actual"].values
            v_pred = q_df.loc[v_mask, model].values

            rows.append(
                {
                    "quarter": q_label,
                    "model": model,
                    "valid_prediction_count": len(v_actual),
                    "wape": round(calculate_wape(v_actual, v_pred), 6),
                    "mae": round(calculate_mae(v_actual, v_pred), 6),
                    "rmse": round(calculate_rmse(v_actual, v_pred), 6),
                }
            )

    return pd.DataFrame(rows).sort_values(["quarter", "model"]).reset_index(drop=True)


def create_robustness_figures(
    df_metrics: pd.DataFrame,
    df_summary: pd.DataFrame,
    figures_dir: UnionPath = "reports/figures",
) -> None:
    """Create publication-grade figures for rolling WAPE trajectories and rank stability.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Fold metrics table.
    df_summary : pd.DataFrame
        Rolling validation summary table.
    figures_dir : UnionPath
        Directory where figure PNGs are saved.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    common_df = df_metrics[df_metrics["evaluation_scope"] == "common_valid"].copy()

    # Identify strongest classical benchmark based on pooled WAPE in summary
    classical_summary = df_summary[df_summary["model"].isin(CLASSICAL_MODELS)]
    best_classical = (
        classical_summary.sort_values("pooled_wape").iloc[0]["model"]
        if not classical_summary.empty
        else "ets_ses"
    )

    models_to_plot = ML_MODELS + [best_classical]
    color_map = {
        "xgboost_with_target_promo": "#1E40AF",  # Deep Blue
        "xgboost_no_target_promo": "#60A5FA",    # Light Blue
        "lightgbm_with_target_promo": "#059669", # Emerald
        "lightgbm_no_target_promo": "#34D399",   # Mint
        best_classical: "#D97706",                # Amber
    }

    # 1. Figure: Rolling Validation WAPE Trajectory
    fig, ax = plt.subplots(figsize=(10, 5.5))
    folds = sorted(common_df["fold"].unique())
    fold_labels = [f"Fold {f}\n(2015-Q{f})" if f <= 4 else f"Fold {f}\n(2016-Q{f-4})" for f in folds]

    for model in models_to_plot:
        m_df = common_df[common_df["model"] == model].sort_values("fold")
        label_text = model.replace("_", " ").title()
        if model == best_classical:
            label_text += f" (Best Baseline)"

        ax.plot(
            m_df["fold"],
            m_df["wape"],
            marker="o",
            linewidth=2.2 if "with_target_promo" in model else 1.6,
            markersize=6,
            color=color_map.get(model, "#64748B"),
            label=label_text,
        )

    ax.set_title("Rolling-Origin Backtest: Fold-Level WAPE Trajectories (2015-2016)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Quarterly Validation Window", fontsize=11, labelpad=8)
    ax.set_ylabel("Portfolio WAPE (Common-Valid)", fontsize=11, labelpad=8)
    ax.set_xticks(folds)
    ax.set_xticklabels(fold_labels, fontsize=9)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9, loc="upper right")
    apply_clean_layout(ax, grid_axis="both")

    wape_fig_path = fig_dir / "rolling_validation_wape.png"
    save_figure(fig, wape_fig_path)
    plt.close(fig)
    logger.info("Saved rolling WAPE figure to %s", wape_fig_path)

    # 2. Figure: Model Rank Stability
    common_df["rank"] = common_df.groupby("fold")["wape"].rank(method="min", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in models_to_plot:
        m_df = common_df[common_df["model"] == model].sort_values("fold")
        label_text = model.replace("_", " ").title()

        ax.plot(
            m_df["fold"],
            m_df["rank"],
            marker="s",
            linewidth=2.0 if "with_target_promo" in model else 1.5,
            markersize=6,
            color=color_map.get(model, "#64748B"),
            label=label_text,
        )

    ax.set_title("Model Rank Stability Across Historical Rolling Folds (1 = Best)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Quarterly Validation Window", fontsize=11, labelpad=8)
    ax.set_ylabel("Rank on Common-Valid WAPE", fontsize=11, labelpad=8)
    ax.set_xticks(folds)
    ax.set_xticklabels(fold_labels, fontsize=9)
    ax.set_yticks(range(1, len(MODELS_ALL) + 1))
    ax.invert_yaxis()  # Rank 1 at the top
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9, loc="lower right")
    apply_clean_layout(ax, grid_axis="both")

    rank_fig_path = fig_dir / "model_rank_stability.png"
    save_figure(fig, rank_fig_path)
    plt.close(fig)
    logger.info("Saved model rank stability figure to %s", rank_fig_path)


def run_robustness_analysis_pipeline(
    df_preds: pd.DataFrame,
    output_dir: UnionPath = "data/processed",
    figures_dir: UnionPath = "reports/figures",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete post-processing robustness analysis and save all deliverables.

    Parameters
    ----------
    df_preds : pd.DataFrame
        Complete rolling validation predictions table.
    output_dir : UnionPath
        Directory for CSV artifacts.
    figures_dir : UnionPath
        Directory for figure artifacts.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_metrics, df_summary, df_robustness)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Fold Metrics (model_valid and common_valid)
    df_metrics = calculate_fold_metrics(df_preds)
    metrics_path = out_path / "rolling_validation_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info("Saved %s (%d rows).", metrics_path, len(df_metrics))

    # 2. Rolling Summary
    df_summary = calculate_rolling_summary(df_preds, df_metrics)
    summary_path = out_path / "rolling_validation_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info("Saved %s (%d rows).", summary_path, len(df_summary))

    # 3. Brand Metrics
    df_brand = calculate_brand_metrics(df_preds)
    brand_path = out_path / "rolling_validation_brand_metrics.csv"
    df_brand.to_csv(brand_path, index=False)
    logger.info("Saved %s (%d rows).", brand_path, len(df_brand))

    # 4. SKU Metrics
    df_sku = calculate_sku_metrics(df_preds)
    sku_path = out_path / "rolling_validation_sku_metrics.csv"
    df_sku.to_csv(sku_path, index=False)
    logger.info("Saved %s (%d rows).", sku_path, len(df_sku))

    # 5. Model Robustness Summary
    df_robustness = calculate_model_robustness_summary(df_summary)
    robustness_path = out_path / "model_robustness_summary.csv"
    df_robustness.to_csv(robustness_path, index=False)
    logger.info("Saved %s (%d rows).", robustness_path, len(df_robustness))

    # 6. 2017 Quarterly Stability Metrics
    try:
        df_quarterly = calculate_2017_quarterly_metrics()
        quarterly_path = out_path / "ml_2017_quarterly_metrics.csv"
        df_quarterly.to_csv(quarterly_path, index=False)
        logger.info("Saved %s (%d rows).", quarterly_path, len(df_quarterly))
    except Exception as exc:
        logger.warning("Could not compute 2017 quarterly metrics: %s", exc)

    # 7. Figures
    try:
        create_robustness_figures(df_metrics, df_summary, figures_dir=figures_dir)
    except Exception as exc:
        logger.warning("Could not generate robustness figures: %s", exc)

    return df_metrics, df_summary, df_robustness
