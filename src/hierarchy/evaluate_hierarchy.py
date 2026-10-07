"""Hierarchical forecast evaluation and Bottom-Up reconciliation module for DemandIQ.

Evaluates coherent hierarchical forecasts across Total, Brand, and SKU levels for:
1. Chronological rolling validation (8 folds, 2015-2016)
2. Outer validation holdout (2017)

Core Responsibilities:
- Load frozen rolling and 2017 bottom-level predictions.
- Apply deterministic Bottom-Up reconciliation using summing matrix S.
- Verify exact mathematical coherence and SKU forecast preservation.
- Calculate WAPE, MAE, and RMSE at SKU, Brand, and Total levels.
- Evaluate model rank stability, fold wins, and promotion-awareness deltas.
- Persist required deliverables under data/processed/ and publication figures under reports/figures/.

Methodological Principles:
- Coherence vs Accuracy distinction: Bottom-Up guarantees coherence by construction;
  aggregate forecasting accuracy is an empirical finding, not an assumed property.
- Existing bottom-level SKU forecasts are preserved without modification.
- Promotion-aware forecasts assume scheduled/planned target-date promotion status.
- 2018 holdout remains strictly sealed and untouched.
- Full precision metric comparison before presentation rounding.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.hierarchy.reconcile import (
    EXPECTED_TOTAL_NODES,
    TOTAL_NODE_ID,
    HierarchySpec,
    build_hierarchy_predictions,
    build_summing_matrix,
    load_sku_hierarchy_mapping,
    verify_forecast_coherence,
)
from src.visualization.plot_utils import apply_clean_layout, save_figure

logger = logging.getLogger("evaluate_hierarchy")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

UnionPath = Union[Path, str]

ML_MODELS: List[str] = [
    "xgboost_no_target_promo",
    "xgboost_with_target_promo",
    "lightgbm_no_target_promo",
    "lightgbm_with_target_promo",
]

HIERARCHY_LEVELS: List[str] = ["total", "brand", "sku"]

LEVEL_NODE_COUNTS: Dict[str, int] = {
    "total": 1,
    "brand": 4,
    "sku": 118,
}


def calculate_hierarchy_metrics(
    df_hierarchy: pd.DataFrame,
    is_rolling: bool = False,
    model_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate hierarchical evaluation metrics (WAPE, MAE, RMSE) by level and model.

    Parameters
    ----------
    df_hierarchy : pd.DataFrame
        Reconciled hierarchy predictions table containing:
        ['date', 'hierarchy_level', 'node_id', 'actual'] + model_cols (and 'fold' if is_rolling).
    is_rolling : bool, default False
        If True, calculates fold-level and pooled metrics. If False, calculates annual metrics.
    model_cols : Optional[List[str]], default None
        List of forecast model column names to evaluate. Defaults to the four Stage 8 ML models.

    Returns
    -------
    pd.DataFrame
        Hierarchical metrics DataFrame matching required output schema.
    """
    if model_cols is None:
        model_cols = ML_MODELS

    rows: List[Dict[str, Any]] = []

    if is_rolling:
        folds = sorted(df_hierarchy["fold"].unique())

        # 1. Fold-level metrics
        for f_id in folds:
            df_fold = df_hierarchy[df_hierarchy["fold"] == f_id]
            for level in HIERARCHY_LEVELS:
                sub = df_fold[df_fold["hierarchy_level"] == level]
                y_true = sub["actual"].values
                n_targets = len(sub)
                node_cnt = LEVEL_NODE_COUNTS[level]

                for model in model_cols:
                    y_pred = sub[model].values
                    rows.append(
                        {
                            "fold": f_id,
                            "evaluation_scope": "fold",
                            "hierarchy_level": level,
                            "model": model,
                            "node_count": node_cnt,
                            "target_count": n_targets,
                            "wape": round(calculate_wape(y_true, y_pred), 6),
                            "mae": round(calculate_mae(y_true, y_pred), 6),
                            "rmse": round(calculate_rmse(y_true, y_pred), 6),
                        }
                    )

        # 2. Pooled metrics across all 8 folds
        for level in HIERARCHY_LEVELS:
            sub = df_hierarchy[df_hierarchy["hierarchy_level"] == level]
            y_true = sub["actual"].values
            n_targets = len(sub)
            node_cnt = LEVEL_NODE_COUNTS[level]

            for model in model_cols:
                y_pred = sub[model].values
                rows.append(
                    {
                        "fold": "pooled",
                        "evaluation_scope": "pooled",
                        "hierarchy_level": level,
                        "model": model,
                        "node_count": node_cnt,
                        "target_count": n_targets,
                        "wape": round(calculate_wape(y_true, y_pred), 6),
                        "mae": round(calculate_mae(y_true, y_pred), 6),
                        "rmse": round(calculate_rmse(y_true, y_pred), 6),
                    }
                )

    else:
        # Annual metrics (e.g. 2017)
        for level in HIERARCHY_LEVELS:
            sub = df_hierarchy[df_hierarchy["hierarchy_level"] == level]
            y_true = sub["actual"].values
            n_targets = len(sub)
            node_cnt = LEVEL_NODE_COUNTS[level]

            for model in model_cols:
                y_pred = sub[model].values
                rows.append(
                    {
                        "hierarchy_level": level,
                        "model": model,
                        "node_count": node_cnt,
                        "target_count": n_targets,
                        "wape": round(calculate_wape(y_true, y_pred), 6),
                        "mae": round(calculate_mae(y_true, y_pred), 6),
                        "rmse": round(calculate_rmse(y_true, y_pred), 6),
                    }
                )

    return pd.DataFrame(rows)


def calculate_hierarchy_summary(df_metrics_rolling: pd.DataFrame) -> pd.DataFrame:
    """Compute compact hierarchy summary with pooled metrics, fold wins, and mean ranks.

    Parameters
    ----------
    df_metrics_rolling : pd.DataFrame
        Rolling metrics table containing both 'fold' and 'pooled' evaluation scopes.

    Returns
    -------
    pd.DataFrame
        Hierarchy summary DataFrame with columns:
        ['hierarchy_level', 'model', 'pooled_wape', 'pooled_mae', 'pooled_rmse',
         'mean_fold_wape', 'median_fold_wape', 'fold_wins', 'mean_rank'].
    """
    fold_metrics = df_metrics_rolling[df_metrics_rolling["evaluation_scope"] == "fold"].copy()
    pooled_metrics = df_metrics_rolling[df_metrics_rolling["evaluation_scope"] == "pooled"].copy()

    # Rank within each hierarchy level and fold
    fold_metrics["rank"] = fold_metrics.groupby(["hierarchy_level", "fold"])["wape"].rank(
        method="min", ascending=True
    )

    summary_rows: List[Dict[str, Any]] = []

    for level in HIERARCHY_LEVELS:
        l_fold = fold_metrics[fold_metrics["hierarchy_level"] == level]
        l_pooled = pooled_metrics[pooled_metrics["hierarchy_level"] == level]

        for model in ML_MODELS:
            m_fold = l_fold[l_fold["model"] == model]
            m_pooled = l_pooled[l_pooled["model"] == model]

            fold_wapes = m_fold["wape"].values
            fold_ranks = m_fold["rank"].values

            fold_wins = int((fold_ranks == 1.0).sum())
            mean_rank = float(np.mean(fold_ranks))

            p_wape = float(m_pooled["wape"].values[0]) if len(m_pooled) > 0 else np.nan
            p_mae = float(m_pooled["mae"].values[0]) if len(m_pooled) > 0 else np.nan
            p_rmse = float(m_pooled["rmse"].values[0]) if len(m_pooled) > 0 else np.nan

            summary_rows.append(
                {
                    "hierarchy_level": level,
                    "model": model,
                    "pooled_wape": round(p_wape, 6),
                    "pooled_mae": round(p_mae, 6),
                    "pooled_rmse": round(p_rmse, 6),
                    "mean_fold_wape": round(float(np.mean(fold_wapes)), 6),
                    "median_fold_wape": round(float(np.median(fold_wapes)), 6),
                    "fold_wins": fold_wins,
                    "mean_rank": round(mean_rank, 2),
                }
            )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["hierarchy_level", "pooled_wape"])
        .reset_index(drop=True)
    )
    return summary_df


def calculate_hierarchy_promotion_comparison(
    df_rolling_hierarchy: pd.DataFrame,
    df_2017_hierarchy: pd.DataFrame,
) -> pd.DataFrame:
    """Compare promotion-aware vs non-promotion models at SKU, Brand, and Total levels.

    Evaluates full precision WAPE deltas:
        abs_wape_diff = wape_with_promo - wape_no_promo
        rel_wape_diff = (wape_with_promo - wape_no_promo) / wape_no_promo
    for XGBoost and LightGBM across pooled 2015-2016 and 2017 annual periods.

    Parameters
    ----------
    df_rolling_hierarchy : pd.DataFrame
        Reconciled rolling hierarchy predictions.
    df_2017_hierarchy : pd.DataFrame
        Reconciled 2017 hierarchy predictions.

    Returns
    -------
    pd.DataFrame
        Promotion comparison DataFrame matching required schema.
    """
    rows: List[Dict[str, Any]] = []

    datasets = [
        ("pooled_rolling", df_rolling_hierarchy),
        ("2017_annual", df_2017_hierarchy),
    ]

    for period_name, df_h in datasets:
        for level in HIERARCHY_LEVELS:
            sub = df_h[df_h["hierarchy_level"] == level]
            y_true = sub["actual"].values

            for fam in ["xgboost", "lightgbm"]:
                with_col = f"{fam}_with_target_promo"
                no_col = f"{fam}_no_target_promo"

                w_with = calculate_wape(y_true, sub[with_col].values)
                w_no = calculate_wape(y_true, sub[no_col].values)

                abs_diff = w_with - w_no
                rel_diff = abs_diff / w_no if w_no != 0.0 else np.nan

                rows.append(
                    {
                        "evaluation_period": period_name,
                        "hierarchy_level": level,
                        "model_family": fam,
                        "wape_with_promo": round(w_with, 6),
                        "wape_no_promo": round(w_no, 6),
                        "abs_wape_diff": round(abs_diff, 6),
                        "rel_wape_diff": round(rel_diff, 6),
                    }
                )

    return pd.DataFrame(rows)


def plot_hierarchy_wape_comparison(
    df_summary: pd.DataFrame,
    output_path: UnionPath = "reports/figures/hierarchy_wape_comparison.png",
) -> Path:
    """Generate publication-grade bar chart comparing WAPE across hierarchy levels.

    Parameters
    ----------
    df_summary : pd.DataFrame
        Hierarchy summary DataFrame with pooled WAPE per level and model.
    output_path : UnionPath, default "reports/figures/hierarchy_wape_comparison.png"
        Destination file path.

    Returns
    -------
    Path
        Resolved figure path.
    """
    path = Path(output_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = ["total", "brand", "sku"]
    level_labels = ["Total\n(1 Node)", "Brand\n(4 Nodes)", "SKU\n(118 Nodes)"]
    x = np.arange(len(levels))
    width = 0.18

    model_configs = [
        ("xgboost_no_target_promo", "XGBoost (No Promo)", "#93C5FD"),
        ("xgboost_with_target_promo", "XGBoost (With Promo)", "#1D4ED8"),
        ("lightgbm_no_target_promo", "LightGBM (No Promo)", "#94A3B8"),
        ("lightgbm_with_target_promo", "LightGBM (With Promo)", "#059669"),
    ]

    for idx, (m_col, m_label, color) in enumerate(model_configs):
        wapes = []
        for lvl in levels:
            val = df_summary[
                (df_summary["hierarchy_level"] == lvl) & (df_summary["model"] == m_col)
            ]["pooled_wape"].values[0]
            wapes.append(val)

        x_pos = x + (idx - 1.5) * width
        rects = ax.bar(x_pos, wapes, width, label=m_label, color=color, edgecolor="#1E293B", linewidth=0.6)

        # Annotate bar values
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height*100:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#0F172A",
            )

    ax.set_ylabel("Pooled WAPE (2015-2016)", fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_title(
        "Hierarchical Forecasting Performance Across Hierarchy Levels",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#0F172A",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=10, fontweight="bold", color="#0F172A")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9, loc="upper left")

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, path)


def plot_hierarchy_promotion_delta(
    df_promo_comp: pd.DataFrame,
    output_path: UnionPath = "reports/figures/hierarchy_promotion_delta.png",
) -> Path:
    """Generate publication-grade bar chart showing promotion WAPE delta by hierarchy level.

    Parameters
    ----------
    df_promo_comp : pd.DataFrame
        Promotion comparison table.
    output_path : UnionPath, default "reports/figures/hierarchy_promotion_delta.png"
        Destination file path.

    Returns
    -------
    Path
        Resolved figure path.
    """
    path = Path(output_path)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sub = df_promo_comp[df_promo_comp["evaluation_period"] == "pooled_rolling"]
    levels = ["total", "brand", "sku"]
    level_labels = ["Total\n(1 Node)", "Brand\n(4 Nodes)", "SKU\n(118 Nodes)"]
    x = np.arange(len(levels))
    width = 0.32

    families = [
        ("xgboost", "XGBoost Delta", "#1D4ED8"),
        ("lightgbm", "LightGBM Delta", "#059669"),
    ]

    for idx, (fam, label, color) in enumerate(families):
        deltas = []
        for lvl in levels:
            val = sub[(sub["hierarchy_level"] == lvl) & (sub["model_family"] == fam)]["abs_wape_diff"].values[0]
            deltas.append(val)

        x_pos = x + (idx - 0.5) * width
        rects = ax.bar(x_pos, deltas, width, label=label, color=color, edgecolor="#1E293B", linewidth=0.6)

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height*100:.2f} pp",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
            )

    ax.axhline(0, color="#64748B", linestyle="-", linewidth=1.0)
    ax.set_ylabel("WAPE Delta (With Promo - No Promo)", fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_title(
        "Benefit of Promotion Awareness Across Forecast Hierarchy Levels\n(Pooled 2015-2016 Rolling Validation)",
        fontsize=12,
        fontweight="bold",
        pad=14,
        color="#0F172A",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=10, fontweight="bold", color="#0F172A")
    ax.set_ylim(-0.06, 0.01)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5, loc="lower right")

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, path)


def run_hierarchy_pipeline(
    rolling_preds_path: UnionPath = "data/processed/rolling_validation_predictions.csv",
    val_preds_path: UnionPath = "data/processed/ml_predictions_validation.csv",
    dim_path: UnionPath = "data/processed/sku_dimension.csv",
    output_dir: UnionPath = "data/processed",
    figures_dir: UnionPath = "reports/figures",
) -> Dict[str, Any]:
    """Execute complete hierarchical forecasting and reconciliation pipeline.

    Parameters
    ----------
    rolling_preds_path : UnionPath
        Path to rolling validation predictions CSV.
    val_preds_path : UnionPath
        Path to 2017 outer validation ML predictions CSV.
    dim_path : UnionPath
        Path to canonical SKU dimension CSV.
    output_dir : UnionPath
        Output directory for processed CSV files.
    figures_dir : UnionPath
        Output directory for generated figures.

    Returns
    -------
    Dict[str, Any]
        Pipeline execution report including validation results, checks, and artifact paths.
    """
    out_dir = Path(output_dir)
    fig_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading hierarchy mapping and building deterministic summing matrix S...")
    sku_mapping = load_sku_hierarchy_mapping(dim_path)
    spec = build_summing_matrix(sku_mapping)

    logger.info("Step 2: Loading frozen rolling predictions (2015-2016)...")
    df_roll_raw = pd.read_csv(rolling_preds_path)
    df_h_roll = build_hierarchy_predictions(df_roll_raw, spec, is_rolling=True, model_cols=ML_MODELS)

    logger.info("Step 3: Loading frozen outer validation predictions (2017)...")
    df_2017_raw = pd.read_csv(val_preds_path)
    df_h_2017 = build_hierarchy_predictions(df_2017_raw, spec, is_rolling=False, model_cols=ML_MODELS)

    logger.info("Step 4: Running exact mathematical coherence checks...")
    checks_roll = verify_forecast_coherence(
        df_h_roll, ML_MODELS, dataset_name="rolling_2015_2016", df_sku_source=df_roll_raw
    )
    checks_2017 = verify_forecast_coherence(
        df_h_2017, ML_MODELS, dataset_name="2017_annual", df_sku_source=df_2017_raw
    )
    all_checks = checks_roll + checks_2017
    df_checks = pd.DataFrame(all_checks)

    # Assert that all checks passed
    if not df_checks["passed"].all():
        failed = df_checks[~df_checks["passed"]]
        raise AssertionError(f"Coherence checks failed!\n{failed}")
    logger.info("All %d coherence checks passed with zero tolerance violations.", len(df_checks))

    logger.info("Step 5: Evaluating hierarchical metrics (WAPE, MAE, RMSE)...")
    df_metrics_roll = calculate_hierarchy_metrics(df_h_roll, is_rolling=True, model_cols=ML_MODELS)
    df_metrics_2017 = calculate_hierarchy_metrics(df_h_2017, is_rolling=False, model_cols=ML_MODELS)

    logger.info("Step 6: Generating hierarchy summary and rank stability...")
    df_summary = calculate_hierarchy_summary(df_metrics_roll)

    logger.info("Step 7: Generating promotion-awareness hierarchical comparison...")
    df_promo_comp = calculate_hierarchy_promotion_comparison(df_h_roll, df_h_2017)

    logger.info("Step 8: Persisting CSV deliverables under %s...", out_dir)
    p_h_roll = out_dir / "hierarchy_predictions_rolling.csv"
    p_h_2017 = out_dir / "hierarchy_predictions_2017.csv"
    p_m_roll = out_dir / "hierarchy_metrics_rolling.csv"
    p_m_2017 = out_dir / "hierarchy_metrics_2017.csv"
    p_checks = out_dir / "hierarchy_coherence_checks.csv"
    p_summary = out_dir / "hierarchy_summary.csv"
    p_promo = out_dir / "hierarchy_promotion_comparison.csv"

    df_h_roll.to_csv(p_h_roll, index=False)
    df_h_2017.to_csv(p_h_2017, index=False)
    df_metrics_roll.to_csv(p_m_roll, index=False)
    df_metrics_2017.to_csv(p_m_2017, index=False)
    df_checks.to_csv(p_checks, index=False)
    df_summary.to_csv(p_summary, index=False)
    df_promo_comp.to_csv(p_promo, index=False)

    logger.info("Step 9: Generating publication-grade figures under %s...", fig_dir)
    p_fig_wape = plot_hierarchy_wape_comparison(df_summary, fig_dir / "hierarchy_wape_comparison.png")
    p_fig_promo = plot_hierarchy_promotion_delta(df_promo_comp, fig_dir / "hierarchy_promotion_delta.png")

    logger.info("Hierarchical reconciliation and evaluation pipeline completed successfully.")
    return {
        "status": "success",
        "predictions_rolling_rows": len(df_h_roll),
        "predictions_2017_rows": len(df_h_2017),
        "metrics_rolling_rows": len(df_metrics_roll),
        "metrics_2017_rows": len(df_metrics_2017),
        "coherence_checks_count": len(df_checks),
        "all_coherence_passed": bool(df_checks["passed"].all()),
        "max_coherence_residual": float(df_checks["max_abs_residual"].max()),
        "summary_rows": len(df_summary),
        "promo_comparison_rows": len(df_promo_comp),
        "figures": [str(p_fig_wape), str(p_fig_promo)],
    }


if __name__ == "__main__":
    run_hierarchy_pipeline()
