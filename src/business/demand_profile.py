"""Demand analytics and profiling module for DemandIQ.

Calculates portfolio statistics, SKU demand profiles, brand contributions,
weekly/calendar patterns, zero-demand distributions, and promotion dynamics
strictly within the development period (default: date <= 2017-12-31).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DEV_CUTOFF = "2017-12-31"


def filter_development_period(
    canonical_df: pd.DataFrame,
    cutoff_date: Optional[str] = DEFAULT_DEV_CUTOFF,
) -> pd.DataFrame:
    """Filter canonical dataset to development period (strictly excluding holdout).

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Full canonical demand DataFrame.
    cutoff_date : str, optional
        Maximum inclusive date for development analysis (default: '2017-12-31').

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with dates <= cutoff_date.
    """
    if cutoff_date is not None:
        filtered = canonical_df[canonical_df["date"] <= cutoff_date].copy()
    else:
        filtered = canonical_df.copy()
    filtered.sort_values(by=["date", "brand_id", "sku_id"], inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def calculate_portfolio_metrics(dev_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate development-period daily portfolio demand statistics.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary with portfolio-level demand metrics.
    """
    daily_portfolio = dev_df.groupby("date")["quantity"].sum().sort_index()

    obs_days = len(daily_portfolio)
    total_demand = int(daily_portfolio.sum())
    mean_daily = float(daily_portfolio.mean())
    median_daily = float(daily_portfolio.median())
    std_daily = float(daily_portfolio.std())
    min_daily = int(daily_portfolio.min())
    max_daily = int(daily_portfolio.max())

    min_date = str(daily_portfolio.idxmin())
    max_date = str(daily_portfolio.idxmax())

    return {
        "observation_days": obs_days,
        "total_demand": total_demand,
        "mean_daily_demand": mean_daily,
        "median_daily_demand": median_daily,
        "std_daily_demand": std_daily,
        "min_daily_demand": min_daily,
        "min_demand_date": min_date,
        "max_daily_demand": max_daily,
        "max_demand_date": max_date,
    }


def calculate_sku_profiles(dev_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate development-period demand profiles per SKU.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.

    Returns
    -------
    pd.DataFrame
        Per-SKU profile DataFrame with columns:
        [brand_id, sku_id, total_demand, mean_daily_demand, median_daily_demand,
         std_daily_demand, min_daily_demand, max_daily_demand, zero_demand_rate,
         nonzero_demand_rate], sorted deterministically by [brand_id, sku_id].
    """
    grouped = dev_df.groupby(["brand_id", "sku_id"])["quantity"]

    total_demand = grouped.sum()
    mean_daily = grouped.mean()
    median_daily = grouped.median()
    std_daily = grouped.std().fillna(0.0)
    min_daily = grouped.min()
    max_daily = grouped.max()

    obs_count = grouped.count()
    zero_count = grouped.apply(lambda s: (s == 0).sum())
    zero_rate = (zero_count / obs_count).round(6)
    nonzero_rate = (1.0 - zero_rate).round(6)

    profiles = pd.DataFrame({
        "total_demand": total_demand.astype(int),
        "mean_daily_demand": mean_daily.round(4),
        "median_daily_demand": median_daily.round(4),
        "std_daily_demand": std_daily.round(4),
        "min_daily_demand": min_daily.astype(int),
        "max_daily_demand": max_daily.astype(int),
        "zero_demand_rate": zero_rate,
        "nonzero_demand_rate": nonzero_rate,
    }).reset_index()

    profiles.sort_values(by=["brand_id", "sku_id"], ascending=[True, True], inplace=True)
    profiles.reset_index(drop=True, inplace=True)
    return profiles


def calculate_brand_summary(dev_df: pd.DataFrame, sku_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate brand-level development summary statistics.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.
    sku_profiles_df : pd.DataFrame
        Per-SKU profiles DataFrame.

    Returns
    -------
    pd.DataFrame
        Brand summary DataFrame with columns:
        [brand_id, sku_count, total_demand, mean_daily_demand, portfolio_demand_share,
         avg_zero_demand_rate], sorted deterministically by [brand_id].
    """
    total_portfolio_demand = dev_df["quantity"].sum()
    obs_days = dev_df["date"].nunique()

    brand_grp = dev_df.groupby("brand_id")["quantity"]
    brand_totals = brand_grp.sum()
    brand_means = (brand_totals / obs_days).round(4)
    brand_shares = (brand_totals / total_portfolio_demand).round(6)

    brand_sku_counts = dev_df.groupby("brand_id")["sku_id"].nunique()
    brand_avg_zero = sku_profiles_df.groupby("brand_id")["zero_demand_rate"].mean().round(6)

    summary = pd.DataFrame({
        "brand_id": sorted(brand_totals.index),
        "sku_count": [brand_sku_counts[b] for b in sorted(brand_totals.index)],
        "total_demand": [int(brand_totals[b]) for b in sorted(brand_totals.index)],
        "mean_daily_demand": [brand_means[b] for b in sorted(brand_totals.index)],
        "portfolio_demand_share": [brand_shares[b] for b in sorted(brand_totals.index)],
        "avg_zero_demand_rate": [brand_avg_zero[b] for b in sorted(brand_totals.index)],
    })

    summary.sort_values(by="brand_id", inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def calculate_weekly_analysis(dev_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze development demand by day of week across portfolio and brands.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.

    Returns
    -------
    pd.DataFrame
        Weekly analysis by day of week.
    """
    df = dev_df.copy()
    df["weekday_num"] = pd.to_datetime(df["date"]).dt.dayofweek  # 0=Monday, 6=Sunday
    df["weekday_name"] = pd.to_datetime(df["date"]).dt.day_name()

    daily_totals = df.groupby(["date", "weekday_num", "weekday_name"])["quantity"].sum().reset_index()

    weekly_summary = daily_totals.groupby(["weekday_num", "weekday_name"])["quantity"].agg(
        day_count="count",
        total_demand="sum",
        mean_daily_demand="mean",
        std_daily_demand="std",
        median_daily_demand="median",
    ).reset_index()

    total_demand = weekly_summary["total_demand"].sum()
    weekly_summary["demand_share"] = (weekly_summary["total_demand"] / total_demand).round(6)
    weekly_summary["mean_daily_demand"] = weekly_summary["mean_daily_demand"].round(4)
    weekly_summary["std_daily_demand"] = weekly_summary["std_daily_demand"].round(4)

    weekly_summary.sort_values(by="weekday_num", inplace=True)
    weekly_summary.reset_index(drop=True, inplace=True)
    return weekly_summary


def calculate_calendar_seasonality(dev_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze development demand by calendar month.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.

    Returns
    -------
    pd.DataFrame
        Monthly seasonality summary.
    """
    df = dev_df.copy()
    dates = pd.to_datetime(df["date"])
    df["month"] = dates.dt.month
    df["month_name"] = dates.dt.month_name()

    daily_totals = df.groupby(["date", "month", "month_name"])["quantity"].sum().reset_index()

    monthly_summary = daily_totals.groupby(["month", "month_name"])["quantity"].agg(
        day_count="count",
        total_demand="sum",
        mean_daily_demand="mean",
        std_daily_demand="std",
    ).reset_index()

    total_demand = monthly_summary["total_demand"].sum()
    monthly_summary["demand_share"] = (monthly_summary["total_demand"] / total_demand).round(6)
    monthly_summary["mean_daily_demand"] = monthly_summary["mean_daily_demand"].round(4)
    monthly_summary["std_daily_demand"] = monthly_summary["std_daily_demand"].round(4)

    monthly_summary.sort_values(by="month", inplace=True)
    monthly_summary.reset_index(drop=True, inplace=True)
    return monthly_summary


def calculate_promotion_summary(dev_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze promotion coverage and demand differences per SKU and overall.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.

    Returns
    -------
    pd.DataFrame
        Per-SKU promotion summary with promotion frequency and demand on
        promotion vs non-promotion days.
    """
    sku_records = []
    for (brand_id, sku_id), group in dev_df.groupby(["brand_id", "sku_id"]):
        total_obs = len(group)
        promo_obs = int(group["promotion"].sum())
        promo_rate = round(promo_obs / total_obs, 6)

        promo_demand = group.loc[group["promotion"] == 1, "quantity"]
        non_promo_demand = group.loc[group["promotion"] == 0, "quantity"]

        mean_promo_demand = round(float(promo_demand.mean()), 4) if len(promo_demand) > 0 else 0.0
        median_promo_demand = round(float(promo_demand.median()), 4) if len(promo_demand) > 0 else 0.0
        mean_non_promo_demand = round(float(non_promo_demand.mean()), 4) if len(non_promo_demand) > 0 else 0.0
        median_non_promo_demand = round(float(non_promo_demand.median()), 4) if len(non_promo_demand) > 0 else 0.0

        sku_records.append({
            "brand_id": brand_id,
            "sku_id": sku_id,
            "total_obs": total_obs,
            "promo_obs": promo_obs,
            "promo_rate": promo_rate,
            "mean_promo_demand": mean_promo_demand,
            "median_promo_demand": median_promo_demand,
            "mean_non_promo_demand": mean_non_promo_demand,
            "median_non_promo_demand": median_non_promo_demand,
        })

    promo_df = pd.DataFrame(sku_records)
    promo_df.sort_values(by=["brand_id", "sku_id"], ascending=[True, True], inplace=True)
    promo_df.reset_index(drop=True, inplace=True)
    return promo_df


def generate_demand_analytics(
    canonical_path: Path = Path("data/processed/sku_demand_daily.csv"),
    processed_dir: Path = Path("data/processed"),
    cutoff_date: Optional[str] = DEFAULT_DEV_CUTOFF,
) -> Dict[str, Any]:
    """Execute complete demand analytics pipeline and save analytical outputs.

    Parameters
    ----------
    canonical_path : Path
        Path to canonical dataset CSV.
    processed_dir : Path
        Directory where analytical summary artifacts will be written.
    cutoff_date : str, optional
        Development cutoff date (default: '2017-12-31').

    Returns
    -------
    Dict[str, Any]
        Dictionary with all analytical findings and DataFrames.
    """
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical dataset not found at: {canonical_path}")

    logger.info("Loading canonical dataset from %s", canonical_path)
    canonical_df = pd.read_csv(canonical_path)

    logger.info("Filtering to development period (date <= %s)...", cutoff_date)
    dev_df = filter_development_period(canonical_df, cutoff_date=cutoff_date)
    logger.info(
        "Development period filtered: %d rows (%d observed days, %d SKUs).",
        len(dev_df),
        dev_df["date"].nunique(),
        dev_df["sku_id"].nunique(),
    )

    # 1. Portfolio Metrics
    portfolio_metrics = calculate_portfolio_metrics(dev_df)
    logger.info(
        "Portfolio demand: %d total units across %d days (mean=%.2f, median=%.2f, std=%.2f).",
        portfolio_metrics["total_demand"],
        portfolio_metrics["observation_days"],
        portfolio_metrics["mean_daily_demand"],
        portfolio_metrics["median_daily_demand"],
        portfolio_metrics["std_daily_demand"],
    )

    # 2. SKU Profiles
    sku_profiles_df = calculate_sku_profiles(dev_df)
    demand_profile_path = processed_dir / "demand_profile.csv"
    sku_profiles_df.to_csv(demand_profile_path, index=False)
    logger.info("SKU demand profiles saved to %s", demand_profile_path)

    # 3. Brand Summary
    brand_summary_df = calculate_brand_summary(dev_df, sku_profiles_df)
    brand_summary_path = processed_dir / "brand_summary.csv"
    brand_summary_df.to_csv(brand_summary_path, index=False)
    logger.info("Brand summary saved to %s", brand_summary_path)

    # 4. Promotion Summary
    promo_summary_df = calculate_promotion_summary(dev_df)
    promo_summary_path = processed_dir / "promotion_summary.csv"
    promo_summary_df.to_csv(promo_summary_path, index=False)
    logger.info("Promotion summary saved to %s", promo_summary_path)

    # 5. Weekly & Calendar Analysis
    weekly_summary_df = calculate_weekly_analysis(dev_df)
    calendar_seasonality_df = calculate_calendar_seasonality(dev_df)

    # 6. Zero-Demand Analysis
    total_cells = len(dev_df)
    total_zero_cells = int((dev_df["quantity"] == 0).sum())
    overall_zero_rate = round(total_zero_cells / total_cells, 6)

    zero_distribution = {
        "overall_zero_rate": overall_zero_rate,
        "total_observations": total_cells,
        "total_zero_observations": total_zero_cells,
        "skus_lt_10_pct_zero": int((sku_profiles_df["zero_demand_rate"] < 0.10).sum()),
        "skus_10_to_25_pct_zero": int(
            ((sku_profiles_df["zero_demand_rate"] >= 0.10) & (sku_profiles_df["zero_demand_rate"] < 0.25)).sum()
        ),
        "skus_25_to_50_pct_zero": int(
            ((sku_profiles_df["zero_demand_rate"] >= 0.25) & (sku_profiles_df["zero_demand_rate"] < 0.50)).sum()
        ),
        "skus_gt_50_pct_zero": int((sku_profiles_df["zero_demand_rate"] >= 0.50).sum()),
    }

    logger.info(
        "Zero-demand analysis: overall rate = %.2f%% (%d of %d cells are zero).",
        overall_zero_rate * 100,
        total_zero_cells,
        total_cells,
    )

    return {
        "portfolio": portfolio_metrics,
        "sku_profiles": sku_profiles_df,
        "brand_summary": brand_summary_df,
        "promotion_summary": promo_summary_df,
        "weekly_summary": weekly_summary_df,
        "calendar_seasonality": calendar_seasonality_df,
        "zero_distribution": zero_distribution,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    generate_demand_analytics()
