"""Baseline forecast evaluation pipeline for DemandIQ.

Executes and evaluates the five classical forecasting baselines across all 118 SKUs:
1. Naive
2. Seasonal Naive 7-Day
3. 7-Day Calendar Moving Average
4. Simple Exponential Smoothing / ETS(A,N,N)
5. Croston SBA

Enforces strict chronological validation (Train: 2014-01-02 to 2016-12-31, Val: 2017-01-01 to 2017-12-31).
2018 holdout remains sealed and untouched.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UnionPath = Union[Path, str]

from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.features.feature_config import (
    DEVELOPMENT_CUTOFF_DATE,
    TRAINING_END_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
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
from src.visualization.plot_utils import apply_clean_layout, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("evaluate_baselines")

BASELINES: List[str] = [
    "naive",
    "seasonal_naive_7",
    "moving_average_7d",
    "ets_ses",
    "croston_sba",
]

PREDICTION_COLUMNS: List[str] = [
    "date",
    "brand_id",
    "sku_id",
    "actual",
    "naive",
    "seasonal_naive_7",
    "moving_average_7d",
    "ets_ses",
    "croston_sba",
]


def load_baseline_data(
    data_path: UnionPath = "data/processed/sku_demand_daily.csv",
) -> pd.DataFrame:
    """Load canonical SKU daily demand data and enforce development cutoff.

    Parameters
    ----------
    data_path : str or Path
        Path to canonical dataset csv.

    Returns
    -------
    pd.DataFrame
        Validated development-period demand dataframe.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Canonical daily demand file not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"date", "brand_id", "sku_id", "quantity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: {required_cols - set(df.columns)}")

    # Check for holdout leakage before filtering
    cutoff_dt = pd.Timestamp(DEVELOPMENT_CUTOFF_DATE)
    dates_dt = pd.to_datetime(df["date"])

    # Strictly filter to development period
    df = df[dates_dt <= cutoff_dt].copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df["quantity"] = df["quantity"].astype(float)

    # Sort deterministically
    df = df.sort_values(["sku_id", "date_dt"]).reset_index(drop=True)
    logger.info("Loaded %d development-period rows across %d SKUs.", len(df), df["sku_id"].nunique())
    return df


def generate_validation_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 1-step ahead validation forecasts for all five baselines.

    Parameters
    ----------
    df : pd.DataFrame
        Development-period SKU daily demand dataset.

    Returns
    -------
    pd.DataFrame
        Validation predictions with actuals and baseline forecasts, sorted deterministically.
    """
    train_end_dt = pd.Timestamp(TRAINING_END_DATE)
    val_start_dt = pd.Timestamp(VALIDATION_START_DATE)
    val_end_dt = pd.Timestamp(VALIDATION_END_DATE)

    # Build full history lookup dictionary for exact calendar lag matching: (sku_id, date_dt) -> quantity
    sku_demand_map: Dict[Tuple[str, pd.Timestamp], float] = df.set_index(
        ["sku_id", "date_dt"]
    )["quantity"].to_dict()

    val_records: List[pd.DataFrame] = []

    for sku_id, g in df.groupby("sku_id", sort=True):
        g = g.sort_values("date_dt").reset_index(drop=True)

        g_train = g[g["date_dt"] <= train_end_dt]
        val_mask = (g["date_dt"] >= val_start_dt) & (g["date_dt"] <= val_end_dt)
        g_val = g[val_mask].copy()

        if len(g_val) == 0:
            continue

        val_indices = g_val.index

        # 1. Naive: shift(1) across full history
        g["naive"] = generate_naive_forecast(g["quantity"])

        # 2. Seasonal Naive 7: exact calendar date t - 7 lookup
        g["seasonal_naive_7"] = generate_seasonal_naive_forecast(
            g["date_dt"], sku_demand_map, sku_id
        )

        # 3. Moving Average 7D: exact trailing window [t - 7D, t - 1D]
        g["moving_average_7d"] = generate_moving_average_forecast(
            g["date_dt"], g["quantity"], window="7D", min_periods=1
        )

        # 4. Simple Exponential Smoothing / ETS(A,N,N): fit alpha on training only
        y_train = g_train["quantity"].values.astype(float)
        alpha_ses, init_level_ses = fit_ses_training_model(y_train)
        y_val = g_val["quantity"].values.astype(float)
        ses_preds = generate_ses_validation_forecast(y_val, alpha_ses, init_level_ses)

        # 5. Croston SBA: fit on training only with fixed alpha=0.1
        croston_state = fit_croston_sba_training_model(y_train, alpha=0.1)
        croston_preds = generate_croston_sba_validation_forecast(
            y_val, croston_state, start_pos=len(y_train), alpha=0.1
        )

        # Slice validation rows and assign stateful forecasts
        sku_val = g.loc[val_indices].copy()
        sku_val["ets_ses"] = ses_preds
        sku_val["croston_sba"] = croston_preds

        val_records.append(sku_val)

    if not val_records:
        raise ValueError("No validation records generated across the portfolio.")

    all_val = pd.concat(val_records, ignore_index=True)
    all_val = all_val.rename(columns={"quantity": "actual"})
    all_val = all_val[PREDICTION_COLUMNS]

    # Deterministic sort: date, brand_id, sku_id
    all_val = all_val.sort_values(["date", "brand_id", "sku_id"]).reset_index(drop=True)
    logger.info("Generated validation predictions for %d rows.", len(all_val))
    return all_val


def calculate_baseline_metrics(val_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate portfolio-level metrics under baseline_valid and common_valid scopes.

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation predictions DataFrame.

    Returns
    -------
    pd.DataFrame
        Portfolio metrics table.
    """
    total_val_targets = len(val_df)
    rows: List[Dict[str, object]] = []

    # Scope 1: baseline_valid (each baseline evaluated on its own valid predictions)
    for b in BASELINES:
        sub = val_df[val_df[b].notna()]
        actual = sub["actual"].values
        pred = sub[b].values
        valid_count = len(sub)
        cov = valid_count / total_val_targets if total_val_targets > 0 else 0.0

        rows.append(
            {
                "evaluation_scope": "baseline_valid",
                "baseline": b,
                "valid_prediction_count": valid_count,
                "total_validation_targets": total_val_targets,
                "coverage": round(cov, 6),
                "wape": round(calculate_wape(actual, pred), 6),
                "mae": round(calculate_mae(actual, pred), 6),
                "rmse": round(calculate_rmse(actual, pred), 6),
            }
        )

    # Scope 2: common_valid (strict intersection where all 5 baselines are valid)
    common_mask = val_df[BASELINES].notna().all(axis=1)
    common_sub = val_df[common_mask]
    common_count = len(common_sub)
    common_cov = common_count / total_val_targets if total_val_targets > 0 else 0.0

    for b in BASELINES:
        actual = common_sub["actual"].values
        pred = common_sub[b].values

        rows.append(
            {
                "evaluation_scope": "common_valid",
                "baseline": b,
                "valid_prediction_count": common_count,
                "total_validation_targets": total_val_targets,
                "coverage": round(common_cov, 6),
                "wape": round(calculate_wape(actual, pred), 6),
                "mae": round(calculate_mae(actual, pred), 6),
                "rmse": round(calculate_rmse(actual, pred), 6),
            }
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df


def calculate_sku_metrics(val_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate SKU-level metrics for all baselines under both scopes.

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation predictions DataFrame.

    Returns
    -------
    pd.DataFrame
        SKU metrics table.
    """
    common_mask = val_df[BASELINES].notna().all(axis=1)
    rows: List[Dict[str, object]] = []

    # Group by SKU deterministically
    for sku_id, g in val_df.groupby("sku_id", sort=True):
        brand_id = g["brand_id"].iloc[0]
        sku_total_targets = len(g)

        # Baseline-valid scope
        for b in BASELINES:
            sub = g[g[b].notna()]
            actual = sub["actual"].values
            pred = sub[b].values
            count = len(sub)
            cov = count / sku_total_targets if sku_total_targets > 0 else 0.0

            rows.append(
                {
                    "evaluation_scope": "baseline_valid",
                    "brand_id": brand_id,
                    "sku_id": sku_id,
                    "baseline": b,
                    "valid_prediction_count": count,
                    "coverage": round(cov, 6),
                    "wape": round(calculate_wape(actual, pred), 6),
                    "mae": round(calculate_mae(actual, pred), 6),
                    "rmse": round(calculate_rmse(actual, pred), 6),
                }
            )

        # Common-valid scope
        g_common = g[common_mask.loc[g.index]]
        common_count = len(g_common)
        common_cov = common_count / sku_total_targets if sku_total_targets > 0 else 0.0

        for b in BASELINES:
            actual = g_common["actual"].values
            pred = g_common[b].values

            rows.append(
                {
                    "evaluation_scope": "common_valid",
                    "brand_id": brand_id,
                    "sku_id": sku_id,
                    "baseline": b,
                    "valid_prediction_count": common_count,
                    "coverage": round(common_cov, 6),
                    "wape": round(calculate_wape(actual, pred), 6),
                    "mae": round(calculate_mae(actual, pred), 6),
                    "rmse": round(calculate_rmse(actual, pred), 6),
                }
            )

    sku_metrics_df = pd.DataFrame(rows)
    sku_metrics_df = sku_metrics_df.sort_values(
        ["evaluation_scope", "brand_id", "sku_id", "baseline"]
    ).reset_index(drop=True)
    return sku_metrics_df


def calculate_baseline_winners(sku_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Identify winning baseline per SKU based on lowest WAPE in common_valid scope.

    Documented deterministic tie-breaker priority order:
    1. ets_ses
    2. croston_sba
    3. moving_average_7d
    4. naive
    5. seasonal_naive_7

    Parameters
    ----------
    sku_metrics_df : pd.DataFrame
        SKU-level metrics table.

    Returns
    -------
    pd.DataFrame
        Winner summary table with sku_wins and sku_win_share.
    """
    common_df = sku_metrics_df[sku_metrics_df["evaluation_scope"] == "common_valid"].copy()
    unique_skus = common_df["sku_id"].unique()
    total_skus = len(unique_skus)

    tie_break_priority = {b: i for i, b in enumerate(BASELINES)}
    wins: Dict[str, int] = {b: 0 for b in BASELINES}

    for sku_id, g in common_df.groupby("sku_id"):
        # Map baseline -> wape
        b_wape = g.set_index("baseline")["wape"].to_dict()

        # Find minimum WAPE
        min_wape = min(b_wape.values())
        candidates = [b for b, w in b_wape.items() if np.isclose(w, min_wape, atol=1e-9)]

        # Apply deterministic tie-break priority
        candidates.sort(key=lambda b: tie_break_priority[b])
        winner = candidates[0]
        wins[winner] += 1

    rows = []
    for b in BASELINES:
        cnt = wins[b]
        share = cnt / total_skus if total_skus > 0 else 0.0
        rows.append(
            {
                "baseline": b,
                "sku_wins": cnt,
                "sku_win_share": round(share, 6),
            }
        )

    winner_df = pd.DataFrame(rows)
    return winner_df


def save_baseline_outputs(
    val_preds: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    sku_metrics: pd.DataFrame,
    winner_summary: pd.DataFrame,
    output_dir: UnionPath = "data/processed",
) -> None:
    """Save all baseline prediction and metric deliverables deterministically.

    Parameters
    ----------
    val_preds : pd.DataFrame
        Validation predictions DataFrame.
    portfolio_metrics : pd.DataFrame
        Portfolio metrics DataFrame.
    sku_metrics : pd.DataFrame
        SKU metrics DataFrame.
    winner_summary : pd.DataFrame
        Winner summary DataFrame.
    output_dir : str or Path
        Target directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    val_preds.to_csv(out_path / "baseline_predictions_validation.csv", index=False)
    portfolio_metrics.to_csv(out_path / "baseline_metrics.csv", index=False)
    sku_metrics.to_csv(out_path / "baseline_sku_metrics.csv", index=False)
    winner_summary.to_csv(out_path / "baseline_winner_summary.csv", index=False)

    logger.info("Saved baseline deliverables to %s", out_path)


def generate_baseline_diagnostic_plot(
    val_preds: pd.DataFrame,
    sku_id: str = "B2_15",
    output_path: UnionPath = "reports/figures/baseline_example_forecasts.png",
) -> None:
    """Generate diagnostic forecast comparison chart for a representative SKU.

    Parameters
    ----------
    val_preds : pd.DataFrame
        Validation predictions DataFrame.
    sku_id : str, default "B2_15"
        Deterministic representative dense SKU.
    output_path : str or Path
        Target figure file path.
    """
    sku_df = val_preds[val_preds["sku_id"] == sku_id].copy()
    if len(sku_df) == 0:
        logger.warning("SKU %s not found for diagnostic plot; skipping.", sku_id)
        return

    # Focus on the first 60 days of validation for clean visual readability
    sku_df["date_dt"] = pd.to_datetime(sku_df["date"])
    slice_df = sku_df.iloc[:60].copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Actuals
    ax.plot(
        slice_df["date_dt"],
        slice_df["actual"],
        color="#1A202C",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="Actual Demand",
        zorder=5,
    )

    # Baselines
    colors = {
        "naive": "#718096",
        "seasonal_naive_7": "#ED8936",
        "moving_average_7d": "#3182CE",
        "ets_ses": "#38A169",
        "croston_sba": "#805AD5",
    }
    labels = {
        "naive": "Naive",
        "seasonal_naive_7": "Seasonal Naive (7D)",
        "moving_average_7d": "Moving Average (7D)",
        "ets_ses": "ETS SES",
        "croston_sba": "Croston SBA",
    }

    for b in BASELINES:
        ax.plot(
            slice_df["date_dt"],
            slice_df[b],
            color=colors[b],
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
            label=labels[b],
        )

    brand = slice_df["brand_id"].iloc[0]
    ax.set_title(
        f"Baseline Forecast Comparison — Representative Dense SKU ({sku_id}, Brand {brand})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Validation Date (Early 2017)", fontsize=11, labelpad=8)
    ax.set_ylabel("Demand Units", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right")

    apply_clean_layout(ax, grid_axis="both")
    fig.tight_layout()
    save_figure(fig, Path(output_path))
    logger.info("Saved diagnostic baseline visual to %s", output_path)


def main() -> None:
    """Execute complete baseline forecast evaluation workflow."""
    logger.info("Starting DemandIQ forecasting baselines workflow...")
    df = load_baseline_data()
    val_preds = generate_validation_predictions(df)
    portfolio_metrics = calculate_baseline_metrics(val_preds)
    sku_metrics = calculate_sku_metrics(val_preds)
    winner_summary = calculate_baseline_winners(sku_metrics)

    save_baseline_outputs(val_preds, portfolio_metrics, sku_metrics, winner_summary)

    # Optional diagnostic visual
    generate_baseline_diagnostic_plot(val_preds, sku_id="B2_15")

    # Log summary
    print("\n" + "=" * 60)
    print("DEMANDIQ BASELINES EVALUATION SUMMARY")
    print("=" * 60)
    print("\nPORTFOLIO METRICS (common_valid):")
    cv_metrics = portfolio_metrics[portfolio_metrics["evaluation_scope"] == "common_valid"]
    print(cv_metrics[["baseline", "valid_prediction_count", "coverage", "wape", "mae", "rmse"]].to_string(index=False))

    print("\nSKU WINS SUMMARY (common_valid):")
    print(winner_summary.to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
