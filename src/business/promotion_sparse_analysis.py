"""Promotion & Sparse-Demand Analysis module for DemandIQ.

Investigates:
1. When planned target-date promotion information improves next-day forecast accuracy
   across promotion and non-promotion operational conditions.
2. Where demand sparsity presents forecasting difficulty and how tree models compare
   against classical intermittent-demand baselines (Croston SBA).

Methodological Principles:
- Strictly predictive association, not causal claims (no 'uplift', 'ROI', or 'causation').
- Temporal isolation: sparsity classification uses strictly training observations.
- Holdout sealing: 2018 is completely sealed and untouched.
- Full precision metric comparison before presentation rounding.
- Common-valid evaluation population across all nine forecasting methods.
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
from src.evaluation.rolling_validation import ROLLING_FOLDS, RollingFold
from src.features.feature_config import DEVELOPMENT_CUTOFF_DATE
from src.visualization.plot_utils import apply_clean_layout, save_figure

logger = logging.getLogger("promotion_sparse_analysis")
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

SPARSITY_BUCKETS: List[str] = [
    "low_sparsity",
    "moderate_sparsity",
    "high_sparsity",
    "very_high_sparsity",
]

SPARSITY_LABELS: Dict[str, str] = {
    "low_sparsity": "< 10%",
    "moderate_sparsity": "10% - 25%",
    "high_sparsity": "25% - 50%",
    "very_high_sparsity": ">= 50%",
}


def assign_sparsity_bucket(zero_rate: float) -> str:
    """Classify zero-demand rate into predefined deterministic sparsity buckets.

    Thresholds:
    - low_sparsity: zero_rate < 0.10
    - moderate_sparsity: 0.10 <= zero_rate < 0.25
    - high_sparsity: 0.25 <= zero_rate < 0.50
    - very_high_sparsity: zero_rate >= 0.50

    Parameters
    ----------
    zero_rate : float
        Proportion of observed training periods with zero demand.

    Returns
    -------
    str
        Assigned bucket identifier.
    """
    if zero_rate < 0.10:
        return "low_sparsity"
    elif zero_rate < 0.25:
        return "moderate_sparsity"
    elif zero_rate < 0.50:
        return "high_sparsity"
    else:
        return "very_high_sparsity"


def load_and_prepare_analysis_datasets(
    rolling_preds_path: UnionPath = "data/processed/rolling_validation_predictions.csv",
    ml_preds_path: UnionPath = "data/processed/ml_predictions_validation.csv",
    baseline_preds_path: UnionPath = "data/processed/baseline_predictions_validation.csv",
    features_path: UnionPath = "data/processed/forecasting_features_development.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, validate, and join prediction datasets with promotion status.

    Parameters
    ----------
    rolling_preds_path : UnionPath
        Path to Stage 9 rolling validation predictions CSV (2015-2016).
    ml_preds_path : UnionPath
        Path to Stage 8 frozen ML validation predictions CSV (2017).
    baseline_preds_path : UnionPath
        Path to Stage 7 baseline predictions CSV (2017).
    features_path : UnionPath
        Path to canonical feature table CSV.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_rolling, df_2017, df_features)
    """
    p_roll = Path(rolling_preds_path)
    p_ml = Path(ml_preds_path)
    p_base = Path(baseline_preds_path)
    p_feat = Path(features_path)

    for p in [p_roll, p_ml, p_base, p_feat]:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    df_rolling = pd.read_csv(p_roll)
    df_ml_2017 = pd.read_csv(p_ml)
    df_base_2017 = pd.read_csv(p_base)
    df_feat = pd.read_csv(p_feat)

    # 1. Enforce holdout isolation: strictly no 2018 in any input
    for name, df_check in [("rolling", df_rolling), ("ml_2017", df_ml_2017), ("base_2017", df_base_2017), ("feat", df_feat)]:
        if (df_check["date"] >= "2018-01-01").any():
            raise ValueError(f"2018 holdout leakage detected in {name} dataset!")

    # 2. Build complete 2017 predictions table containing all 9 models
    base_cols = [c for c in CLASSICAL_MODELS if c in df_base_2017.columns]
    df_2017 = pd.merge(
        df_ml_2017,
        df_base_2017[["date", "sku_id"] + base_cols],
        on=["date", "sku_id"],
        how="inner",
    )
    if len(df_2017) != len(df_ml_2017):
        raise ValueError(f"Merge between ML 2017 and Baseline 2017 altered row count: {len(df_2017)} vs {len(df_ml_2017)}")

    # 3. Extract promotion mapping from canonical feature table
    promo_map = df_feat[["date", "sku_id", "target_promotion"]].copy()
    if promo_map.duplicated(subset=["date", "sku_id"]).any():
        raise ValueError("Duplicate (date, sku_id) keys found in feature table promotion mapping!")

    # 4. Perform 1-to-1 join to attach target_promotion
    initial_roll_len = len(df_rolling)
    df_rolling = pd.merge(df_rolling, promo_map, on=["date", "sku_id"], how="inner")
    if len(df_rolling) != initial_roll_len:
        raise ValueError(f"Promotion join altered rolling prediction row count: {len(df_rolling)} vs {initial_roll_len}")

    initial_2017_len = len(df_2017)
    df_2017 = pd.merge(df_2017, promo_map, on=["date", "sku_id"], how="inner")
    if len(df_2017) != initial_2017_len:
        raise ValueError(f"Promotion join altered 2017 prediction row count: {len(df_2017)} vs {initial_2017_len}")

    # Validate promotion values are strictly binary
    assert set(df_rolling["target_promotion"].unique()).issubset({0, 1})
    assert set(df_2017["target_promotion"].unique()).issubset({0, 1})

    logger.info("Loaded datasets: Rolling=%d rows, 2017=%d rows, Features=%d rows.", len(df_rolling), len(df_2017), len(df_feat))
    return df_rolling, df_2017, df_feat


def compute_training_sparsity_map(
    df_features: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Dict[str, str]:
    """Calculate SKU zero-demand rates and assign sparsity buckets strictly on training data.

    Parameters
    ----------
    df_features : pd.DataFrame
        Development feature table.
    start_date : str
        Start of training period (inclusive).
    end_date : str
        End of training period (inclusive).

    Returns
    -------
    Dict[str, str]
        Mapping from sku_id to assigned sparsity bucket.
    """
    tr_mask = (df_features["date"] >= start_date) & (df_features["date"] <= end_date)
    df_tr = df_features[tr_mask]

    sku_zeros = df_tr.groupby("sku_id")["target_quantity"].apply(lambda s: float((s == 0.0).mean()))
    return {sku: assign_sparsity_bucket(rate) for sku, rate in sku_zeros.items()}


def attach_sparsity_buckets(
    df_rolling: pd.DataFrame,
    df_2017: pd.DataFrame,
    df_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach training-only sparsity buckets to rolling and 2017 validation predictions.

    For rolling folds, each fold uses strictly its own training window.
    For 2017, uses the full Stage 8 outer-training window (2014-01-02 to 2016-12-31).

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Rolling validation predictions with target_promotion.
    df_2017 : pd.DataFrame
        2017 validation predictions with target_promotion.
    df_features : pd.DataFrame
        Development feature table.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames with added 'sparsity_bucket' column.
    """
    # 1. Rolling fold sparsity: training-only per fold
    fold_dfs = []
    for fold in ROLLING_FOLDS:
        f_mask = df_rolling["fold"] == fold.fold_id
        f_val = df_rolling[f_mask].copy()

        sparsity_map = compute_training_sparsity_map(df_features, fold.train_start, fold.train_end)
        f_val["sparsity_bucket"] = f_val["sku_id"].map(sparsity_map)

        if f_val["sparsity_bucket"].isna().any():
            raise ValueError(f"Unmapped SKU sparsity bucket in fold {fold.fold_id}")

        fold_dfs.append(f_val)

    df_rolling_with_sparse = pd.concat(fold_dfs, ignore_index=True)
    df_rolling_with_sparse = df_rolling_with_sparse.sort_values(["fold", "date", "brand_id", "sku_id"]).reset_index(drop=True)

    # 2. 2017 outer training sparsity (2014-01-02 to 2016-12-31)
    sparse_map_2017 = compute_training_sparsity_map(df_features, "2014-01-02", "2016-12-31")
    df_2017_with_sparse = df_2017.copy()
    df_2017_with_sparse["sparsity_bucket"] = df_2017_with_sparse["sku_id"].map(sparse_map_2017)

    if df_2017_with_sparse["sparsity_bucket"].isna().any():
        raise ValueError("Unmapped SKU sparsity bucket in 2017 validation data")

    df_2017_with_sparse = df_2017_with_sparse.sort_values(["date", "brand_id", "sku_id"]).reset_index(drop=True)
    return df_rolling_with_sparse, df_2017_with_sparse


def compute_promotion_condition_metrics(
    df_rolling: pd.DataFrame,
    df_2017: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate forecast error metrics segmented by promotion vs non-promotion conditions.

    Evaluates:
    - pooled_2015_2016 across all 8 folds
    - fold_1 through fold_8 individually
    - annual_2017
    - 2017_Q1 through 2017_Q4

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Rolling predictions with target_promotion.
    df_2017 : pd.DataFrame
        2017 predictions with target_promotion.

    Returns
    -------
    pd.DataFrame
        Promotion condition metrics table.
    """
    rows = []

    # Helper: process a single evaluation population slice
    def process_period_slice(period_label: str, df_slice: pd.DataFrame) -> None:
        # Enforce common-valid filter across all 9 models
        common_mask = df_slice[MODELS_ALL].notna().all(axis=1)
        df_common = df_slice[common_mask]
        total_targets = len(df_common)
        total_demand = df_common["actual"].sum()

        for promo_val, promo_cond in [(0, "non_promotion"), (1, "promotion")]:
            cond_df = df_common[df_common["target_promotion"] == promo_val]
            target_count = len(cond_df)
            actual_sum = float(cond_df["actual"].sum())
            target_share = target_count / total_targets if total_targets > 0 else 0.0
            demand_share = actual_sum / total_demand if total_demand > 0 else 0.0

            y_true = cond_df["actual"].values

            for model in MODELS_ALL:
                y_pred = cond_df[model].values
                rows.append(
                    {
                        "evaluation_period": period_label,
                        "target_promotion": promo_val,
                        "promotion_condition": promo_cond,
                        "model": model,
                        "target_count": target_count,
                        "target_share": round(target_share, 4),
                        "actual_sum": round(actual_sum, 2),
                        "demand_share": round(demand_share, 4),
                        "wape": round(calculate_wape(y_true, y_pred), 6),
                        "mae": round(calculate_mae(y_true, y_pred), 6),
                        "rmse": round(calculate_rmse(y_true, y_pred), 6),
                    }
                )

    # 1. Pooled 2015-2016
    process_period_slice("pooled_2015_2016", df_rolling)

    # 2. Individual Rolling Folds 1..8
    for f_id in sorted(df_rolling["fold"].unique()):
        process_period_slice(f"fold_{f_id}", df_rolling[df_rolling["fold"] == f_id])

    # 3. Annual 2017
    process_period_slice("annual_2017", df_2017)

    # 4. 2017 Calendar Quarters
    df_2017_q = df_2017.copy()
    quarter_spans = [
        ("2017_Q1", "2017-01-01", "2017-03-31"),
        ("2017_Q2", "2017-04-01", "2017-06-30"),
        ("2017_Q3", "2017-07-01", "2017-09-30"),
        ("2017_Q4", "2017-10-01", "2017-12-31"),
    ]
    for q_label, q_start, q_end in quarter_spans:
        q_mask = (df_2017_q["date"] >= q_start) & (df_2017_q["date"] <= q_end)
        process_period_slice(q_label, df_2017_q[q_mask])

    out_df = pd.DataFrame(rows)
    return out_df.sort_values(["evaluation_period", "target_promotion", "wape"]).reset_index(drop=True)


def compute_promotion_model_comparison(
    df_rolling: pd.DataFrame,
    df_2017: pd.DataFrame,
) -> pd.DataFrame:
    """Compare promotion-aware vs non-promotion-aware models at portfolio and SKU level.

    Calculates:
    - Overall portfolio WAPE with vs without target promotion
    - Absolute and relative WAPE difference
    - SKU win distribution (improved, worsened, unchanged)

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Rolling predictions.
    df_2017 : pd.DataFrame
        2017 predictions.

    Returns
    -------
    pd.DataFrame
        Model comparison table.
    """
    periods = []

    # 1. Pooled 2015-2016
    periods.append(("pooled_2015_2016", df_rolling))

    # 2. Folds 1..8
    for f_id in sorted(df_rolling["fold"].unique()):
        periods.append((f"fold_{f_id}", df_rolling[df_rolling["fold"] == f_id]))

    # 3. Annual 2017
    periods.append(("annual_2017", df_2017))

    # 4. 2017 Quarters
    df_2017_q = df_2017.copy()
    quarter_spans = [
        ("2017_Q1", "2017-01-01", "2017-03-31"),
        ("2017_Q2", "2017-04-01", "2017-06-30"),
        ("2017_Q3", "2017-07-01", "2017-09-30"),
        ("2017_Q4", "2017-10-01", "2017-12-31"),
    ]
    for q_label, q_start, q_end in quarter_spans:
        q_mask = (df_2017_q["date"] >= q_start) & (df_2017_q["date"] <= q_end)
        periods.append((q_label, df_2017_q[q_mask]))

    rows = []
    comparisons = [
        ("xgboost", "xgboost_with_target_promo", "xgboost_no_target_promo"),
        ("lightgbm", "lightgbm_with_target_promo", "lightgbm_no_target_promo"),
    ]

    for p_label, df_slice in periods:
        common_mask = df_slice[MODELS_ALL].notna().all(axis=1)
        df_common = df_slice[common_mask]
        y_true = df_common["actual"].values

        for family, col_with, col_no in comparisons:
            wape_with_raw = calculate_wape(y_true, df_common[col_with].values)
            wape_no_raw = calculate_wape(y_true, df_common[col_no].values)

            abs_diff = wape_with_raw - wape_no_raw
            rel_diff = (abs_diff / wape_no_raw) * 100.0 if wape_no_raw > 0 else 0.0

            # SKU-level win forensics (full precision)
            improved = 0
            worsened = 0
            unchanged = 0
            skus_in_slice = sorted(df_common["sku_id"].unique())

            for sku in skus_in_slice:
                sku_df = df_common[df_common["sku_id"] == sku]
                y_sku = sku_df["actual"].values
                w_with = calculate_wape(y_sku, sku_df[col_with].values)
                w_no = calculate_wape(y_sku, sku_df[col_no].values)

                # Full-precision comparison
                if w_with < w_no:
                    improved += 1
                elif w_with > w_no:
                    worsened += 1
                else:
                    unchanged += 1

            rows.append(
                {
                    "evaluation_period": p_label,
                    "model_family": family,
                    "wape_with_promo": round(wape_with_raw, 6),
                    "wape_no_promo": round(wape_no_raw, 6),
                    "abs_wape_diff": round(abs_diff, 6),
                    "rel_wape_diff": round(rel_diff, 2),
                    "skus_improved": improved,
                    "skus_worsened": worsened,
                    "skus_unchanged": unchanged,
                    "total_skus_evaluated": len(skus_in_slice),
                }
            )

    return pd.DataFrame(rows)


def compute_sparse_demand_metrics(
    df_rolling: pd.DataFrame,
    df_2017: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate error metrics across four training-only sparsity buckets.

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Rolling predictions with attached sparsity_bucket.
    df_2017 : pd.DataFrame
        2017 predictions with attached sparsity_bucket.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_metrics, df_summary)
    """
    metric_rows = []
    summary_rows = []

    periods = [
        ("pooled_2015_2016", df_rolling),
        ("annual_2017", df_2017),
    ]

    for p_label, df_slice in periods:
        common_mask = df_slice[MODELS_ALL].notna().all(axis=1)
        df_common = df_slice[common_mask]
        period_total_demand = df_common["actual"].sum()

        for bucket in SPARSITY_BUCKETS:
            b_df = df_common[df_common["sparsity_bucket"] == bucket]
            sku_count = b_df["sku_id"].nunique()
            target_count = len(b_df)
            actual_sum = float(b_df["actual"].sum())
            demand_share = actual_sum / period_total_demand if period_total_demand > 0 else 0.0

            y_true = b_df["actual"].values
            bucket_wapes = {}

            for model in MODELS_ALL:
                y_pred = b_df[model].values
                w = calculate_wape(y_true, y_pred)
                mae = calculate_mae(y_true, y_pred)
                rmse = calculate_rmse(y_true, y_pred)

                bucket_wapes[model] = w

                metric_rows.append(
                    {
                        "evaluation_period": p_label,
                        "sparsity_bucket": bucket,
                        "model": model,
                        "sku_count": sku_count,
                        "target_count": target_count,
                        "actual_sum": round(actual_sum, 2),
                        "wape": round(w, 6),
                        "mae": round(mae, 6),
                        "rmse": round(rmse, 6),
                    }
                )

            # Build summary row for this bucket
            # Identify best classical baseline
            classical_wapes = {m: bucket_wapes[m] for m in CLASSICAL_MODELS}
            best_cls_model = min(classical_wapes, key=classical_wapes.get)
            best_cls_wape = classical_wapes[best_cls_model]

            # Overall best model in bucket
            best_model = min(bucket_wapes, key=bucket_wapes.get)
            best_wape = bucket_wapes[best_model]

            summary_rows.append(
                {
                    "evaluation_period": p_label,
                    "sparsity_bucket": bucket,
                    "zero_demand_range": SPARSITY_LABELS[bucket],
                    "sku_count": sku_count,
                    "target_count": target_count,
                    "actual_demand_share": round(demand_share, 4),
                    "croston_wape": round(bucket_wapes["croston_sba"], 6),
                    "best_baseline_model": best_cls_model,
                    "best_baseline_wape": round(best_cls_wape, 6),
                    "xgboost_with_promo_wape": round(bucket_wapes["xgboost_with_target_promo"], 6),
                    "lightgbm_with_promo_wape": round(bucket_wapes["lightgbm_with_target_promo"], 6),
                    "best_model": best_model,
                    "best_wape": round(best_wape, 6),
                }
            )

    df_metrics = pd.DataFrame(metric_rows).sort_values(["evaluation_period", "sparsity_bucket", "wape"]).reset_index(drop=True)
    df_summary = pd.DataFrame(summary_rows).sort_values(["evaluation_period", "sparsity_bucket"]).reset_index(drop=True)
    return df_metrics, df_summary


def compute_promotion_sparse_cross_diagnostic(
    df_rolling: pd.DataFrame,
    df_2017: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tabulate promotion condition by sparsity bucket as a descriptive diagnostic.

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Rolling predictions.
    df_2017 : pd.DataFrame
        2017 predictions.

    Returns
    -------
    pd.DataFrame
        Diagnostic cross-tabulation table.
    """
    rows = []
    periods = [
        ("pooled_2015_2016", df_rolling),
        ("annual_2017", df_2017),
    ]

    for p_label, df_slice in periods:
        common_mask = df_slice[MODELS_ALL].notna().all(axis=1)
        df_common = df_slice[common_mask]

        for promo_val, promo_cond in [(0, "non_promotion"), (1, "promotion")]:
            for bucket in SPARSITY_BUCKETS:
                cell_df = df_common[(df_common["target_promotion"] == promo_val) & (df_common["sparsity_bucket"] == bucket)]
                if len(cell_df) == 0:
                    continue

                y_true = cell_df["actual"].values
                rows.append(
                    {
                        "evaluation_period": p_label,
                        "target_promotion": promo_val,
                        "promotion_condition": promo_cond,
                        "sparsity_bucket": bucket,
                        "target_count": len(cell_df),
                        "actual_sum": round(float(y_true.sum()), 2),
                        "xgboost_with_promo_wape": round(calculate_wape(y_true, cell_df["xgboost_with_target_promo"].values), 6),
                        "lightgbm_with_promo_wape": round(calculate_wape(y_true, cell_df["lightgbm_with_target_promo"].values), 6),
                        "croston_wape": round(calculate_wape(y_true, cell_df["croston_sba"].values), 6),
                        "ets_ses_wape": round(calculate_wape(y_true, cell_df["ets_ses"].values), 6),
                    }
                )

    return pd.DataFrame(rows).sort_values(["evaluation_period", "target_promotion", "sparsity_bucket"]).reset_index(drop=True)


# =====================================================================
# VISUALIZATION GENERATORS
# =====================================================================

def create_promotion_condition_figure(
    df_promo_metrics: pd.DataFrame,
    output_path: UnionPath = "reports/figures/promotion_condition_wape.png",
) -> None:
    """Generate grouped bar chart comparing WAPE by promotion condition."""
    fig_path = Path(output_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to pooled 2015-2016
    df_plot = df_promo_metrics[df_promo_metrics["evaluation_period"] == "pooled_2015_2016"].copy()

    models_focus = [
        "xgboost_with_target_promo",
        "lightgbm_with_target_promo",
        "xgboost_no_target_promo",
        "lightgbm_no_target_promo",
        "ets_ses",
    ]
    df_plot = df_plot[df_plot["model"].isin(models_focus)]

    labels = [m.replace("_", " ").title().replace("Target Promo", "Promo") for m in models_focus]
    non_promo_wape = [df_plot[(df_plot["model"] == m) & (df_plot["target_promotion"] == 0)]["wape"].iloc[0] for m in models_focus]
    promo_wape = [df_plot[(df_plot["model"] == m) & (df_plot["target_promotion"] == 1)]["wape"].iloc[0] for m in models_focus]

    x = np.arange(len(models_focus))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width / 2, non_promo_wape, width, label="Non-Promotion (62.4% Days / 33.3% Demand)", color="#64748B", alpha=0.9)
    bars2 = ax.bar(x + width / 2, promo_wape, width, label="Promotion (37.6% Days / 66.7% Demand)", color="#EA580C", alpha=0.95)

    ax.set_title("Forecast Error by Promotion Condition (Pooled Rolling 2015-2016)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Portfolio WAPE (Common-Valid)", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5, loc="upper right")
    apply_clean_layout(ax, grid_axis="y")

    # Add value labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)

    save_figure(fig, fig_path)
    plt.close(fig)
    logger.info("Saved promotion condition figure to %s", fig_path)


def create_promotion_delta_figure(
    df_promo_comp: pd.DataFrame,
    output_path: UnionPath = "reports/figures/promotion_model_delta.png",
) -> None:
    """Generate chart showing fold-level WAPE error reduction from target promotion."""
    fig_path = Path(output_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to 8 rolling folds and 2017 annual
    periods_order = [f"fold_{i}" for i in range(1, 9)] + ["annual_2017"]
    df_plot = df_promo_comp[df_promo_comp["evaluation_period"].isin(periods_order)].copy()

    period_labels = [f"Fold {i}\n(2015-Q{i})" if i <= 4 else f"Fold {i}\n(2016-Q{i-4})" for i in range(1, 9)] + ["Annual 2017\n(Outer Val)"]

    xgb_deltas = [df_plot[(df_plot["evaluation_period"] == p) & (df_plot["model_family"] == "xgboost")]["abs_wape_diff"].iloc[0] for p in periods_order]
    lgb_deltas = [df_plot[(df_plot["evaluation_period"] == p) & (df_plot["model_family"] == "lightgbm")]["abs_wape_diff"].iloc[0] for p in periods_order]

    x = np.arange(len(periods_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x - width / 2, xgb_deltas, width, label="XGBoost (With Promo vs No Promo)", color="#1E40AF", alpha=0.9)
    ax.bar(x + width / 2, lgb_deltas, width, label="LightGBM (With Promo vs No Promo)", color="#059669", alpha=0.9)

    ax.axhline(0, color="#0F172A", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_title("WAPE Difference: Target Promotion vs Non-Promotion Model Variant", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("WAPE Delta (Negative = Promotion Awareness Improves Accuracy)", fontsize=10.5, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, fontsize=8.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5, loc="lower right")
    apply_clean_layout(ax, grid_axis="y")

    save_figure(fig, fig_path)
    plt.close(fig)
    logger.info("Saved promotion delta figure to %s", fig_path)


def create_sparse_demand_figure(
    df_sparse_metrics: pd.DataFrame,
    output_path: UnionPath = "reports/figures/sparse_demand_wape.png",
) -> None:
    """Generate grouped bar chart comparing WAPE across training-defined sparsity buckets."""
    fig_path = Path(output_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = df_sparse_metrics[df_sparse_metrics["evaluation_period"] == "pooled_2015_2016"].copy()

    models_compare = [
        "xgboost_with_target_promo",
        "lightgbm_with_target_promo",
        "ets_ses",
        "croston_sba",
    ]
    color_map = {
        "xgboost_with_target_promo": "#1E40AF",
        "lightgbm_with_target_promo": "#059669",
        "ets_ses": "#D97706",
        "croston_sba": "#7C3AED",
    }
    model_labels = {
        "xgboost_with_target_promo": "XGBoost (With Promo)",
        "lightgbm_with_target_promo": "LightGBM (With Promo)",
        "ets_ses": "ETS / SES (Top Baseline)",
        "croston_sba": "Croston SBA (Intermittent Ref)",
    }

    buckets = SPARSITY_BUCKETS
    bucket_labels = [f"{b.replace('_', ' ').title()}\n({SPARSITY_LABELS[b]} Zeros)" for b in buckets]

    x = np.arange(len(buckets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for idx, model in enumerate(models_compare):
        wapes = [df_plot[(df_plot["sparsity_bucket"] == b) & (df_plot["model"] == model)]["wape"].iloc[0] for b in buckets]
        offset = (idx - 1.5) * width
        ax.bar(x + offset, wapes, width, label=model_labels[model], color=color_map[model], alpha=0.9)

    ax.set_title("Forecast Accuracy Across Demand Sparsity Buckets (Pooled 2015-2016)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Portfolio WAPE (Common-Valid)", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5, loc="upper left")
    apply_clean_layout(ax, grid_axis="y")

    save_figure(fig, fig_path)
    plt.close(fig)
    logger.info("Saved sparse demand figure to %s", fig_path)


# =====================================================================
# MAIN PIPELINE RUNNER
# =====================================================================

def run_promotion_sparse_analysis_pipeline(
    output_dir: UnionPath = "data/processed",
    figures_dir: UnionPath = "reports/figures",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete promotion and sparse-demand analysis and generate all deliverables.

    Parameters
    ----------
    output_dir : UnionPath, default "data/processed"
        Directory for CSV output artifacts.
    figures_dir : UnionPath, default "reports/figures"
        Directory for figure PNG artifacts.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_promo_metrics, df_promo_comp, df_sparse_metrics, df_sparse_summary, df_diag)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting DemandIQ Promotion & Sparse-Demand Analysis...")

    # 1. Load and validate prediction artifacts and features
    df_rolling, df_2017, df_features = load_and_prepare_analysis_datasets()

    # 2. Attach training-only sparsity classifications
    df_rolling, df_2017 = attach_sparsity_buckets(df_rolling, df_2017, df_features)

    # 3. Promotion condition metrics
    df_promo_metrics = compute_promotion_condition_metrics(df_rolling, df_2017)
    promo_metrics_file = out_path / "promotion_condition_metrics.csv"
    df_promo_metrics.to_csv(promo_metrics_file, index=False)
    logger.info("Saved promotion condition metrics to %s (%d rows).", promo_metrics_file, len(df_promo_metrics))

    # 4. Promotion-aware model comparison
    df_promo_comp = compute_promotion_model_comparison(df_rolling, df_2017)
    promo_comp_file = out_path / "promotion_model_comparison.csv"
    df_promo_comp.to_csv(promo_comp_file, index=False)
    logger.info("Saved promotion model comparison to %s (%d rows).", promo_comp_file, len(df_promo_comp))

    # 5. Sparse-demand metrics & summary
    df_sparse_metrics, df_sparse_summary = compute_sparse_demand_metrics(df_rolling, df_2017)
    sparse_metrics_file = out_path / "sparse_demand_metrics.csv"
    df_sparse_metrics.to_csv(sparse_metrics_file, index=False)
    logger.info("Saved sparse demand metrics to %s (%d rows).", sparse_metrics_file, len(df_sparse_metrics))

    sparse_summary_file = out_path / "sparse_demand_summary.csv"
    df_sparse_summary.to_csv(sparse_summary_file, index=False)
    logger.info("Saved sparse demand summary to %s (%d rows).", sparse_summary_file, len(df_sparse_summary))

    # 6. Combined diagnostic cross-tabulation
    df_diag = compute_promotion_sparse_cross_diagnostic(df_rolling, df_2017)
    diag_file = out_path / "promotion_sparse_metrics.csv"
    df_diag.to_csv(diag_file, index=False)
    logger.info("Saved promotion-by-sparsity diagnostic to %s (%d rows).", diag_file, len(df_diag))

    # 7. Generate publication-grade figures
    try:
        create_promotion_condition_figure(df_promo_metrics, fig_path / "promotion_condition_wape.png")
        create_promotion_delta_figure(df_promo_comp, fig_path / "promotion_model_delta.png")
        create_sparse_demand_figure(df_sparse_metrics, fig_path / "sparse_demand_wape.png")
    except Exception as exc:
        logger.warning("Could not generate figures: %s", exc)

    logger.info("=== Promotion & Sparse-Demand Analysis Complete ===")
    return df_promo_metrics, df_promo_comp, df_sparse_metrics, df_sparse_summary, df_diag


if __name__ == "__main__":
    run_promotion_sparse_analysis_pipeline()
