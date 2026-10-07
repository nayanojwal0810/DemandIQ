"""Exploratory Data Analysis and Demand Storytelling Visualization Module.

Generates reproducible, publication-grade static figures communicating portfolio demand,
seasonality, brand contributions, SKU concentration, sparsity, promotion dynamics,
and representative dense/sparse SKU behavior strictly within the development period
(2014-01-02 to 2017-12-31). 2018 is strictly held out and untouched.
"""

from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from src.features.feature_config import DEVELOPMENT_CUTOFF_DATE
from src.business.demand_profile import (
    filter_development_period,
    calculate_sku_profiles,
    calculate_brand_summary,
    calculate_weekly_analysis,
    calculate_calendar_seasonality,
)
from src.visualization.plot_utils import (
    BRAND_COLORS,
    PROMO_COLORS,
    PRIMARY_COLOR,
    NEUTRAL_DARK,
    NEUTRAL_MUTED,
    FIGURE_DPI,
    apply_clean_layout,
    save_figure,
    select_representative_dense_sku,
    select_representative_sparse_sku,
)

logger = logging.getLogger(__name__)

FIGURE_NAMES: List[str] = [
    "portfolio_daily_demand.png",
    "day_of_week_seasonality.png",
    "monthly_seasonality.png",
    "brand_demand_share.png",
    "sku_demand_rank.png",
    "sku_zero_demand_distribution.png",
    "sparsity_vs_demand_volume.png",
    "promotion_demand_by_brand.png",
    "representative_dense_sku.png",
    "representative_sparse_sku.png",
]


def plot_portfolio_daily_demand(dev_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure 1: Portfolio Daily Demand Over Time across Development Period.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    daily = dev_df.groupby("date")["quantity"].sum().reset_index()
    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily.sort_values(by="date_dt", inplace=True)
    daily["rolling_28"] = daily["quantity"].rolling(28, min_periods=7).mean()

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=FIGURE_DPI)
    ax.plot(
        daily["date_dt"],
        daily["quantity"],
        color="#93C5FD",
        linewidth=0.75,
        alpha=0.85,
        label="Daily Total Demand",
    )
    ax.plot(
        daily["date_dt"],
        daily["rolling_28"],
        color=PRIMARY_COLOR,
        linewidth=1.8,
        label="28-Day Rolling Average",
    )

    ax.set_title(
        "Portfolio Daily Demand — Development Period (2014–2017)",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Date", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Total Demand (Units)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(
        daily["date_dt"].min() - pd.Timedelta(days=5),
        daily["date_dt"].max() + pd.Timedelta(days=5),
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    ax.legend(frameon=False, loc="upper right", fontsize=9)
    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, output_path)


def plot_day_of_week_seasonality(dev_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure 2: Day-of-Week Seasonality (Semantic Monday -> Sunday order).

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    weekly = calculate_weekly_analysis(dev_df)
    # Ensure strict semantic order Monday (0) through Sunday (6)
    weekly.sort_values(by="weekday_num", inplace=True)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=FIGURE_DPI)
    bars = ax.bar(
        weekly["weekday_name"],
        weekly["mean_daily_demand"],
        color=PRIMARY_COLOR,
        width=0.55,
        edgecolor="none",
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 10,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=NEUTRAL_DARK,
        )

    ax.set_title(
        "Day-of-Week Mean Portfolio Demand — Development Period",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Day of Week", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Mean Daily Demand (Units)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylim(0, weekly["mean_daily_demand"].max() * 1.15)

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, output_path)


def plot_monthly_seasonality(dev_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure 3: Monthly Seasonality (Calendar January -> December order).

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    monthly = calculate_calendar_seasonality(dev_df)
    monthly.sort_values(by="month", inplace=True)
    month_abbrs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=FIGURE_DPI)
    bars = ax.bar(
        month_abbrs,
        monthly["mean_daily_demand"],
        color=PRIMARY_COLOR,
        width=0.55,
        edgecolor="none",
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 7,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=NEUTRAL_DARK,
        )

    ax.set_title(
        "Monthly Mean Portfolio Daily Demand — Development Period",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Calendar Month", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Mean Daily Demand (Units)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylim(0, monthly["mean_daily_demand"].max() * 1.15)

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, output_path)


def plot_brand_demand_share(brand_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure 4: Brand Demand Contribution (Horizontal bar chart).

    Parameters
    ----------
    brand_df : pd.DataFrame
        Brand summary DataFrame with total_demand and portfolio_demand_share.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    # Sort ascending for horizontal bar chart (largest at the top)
    sorted_df = brand_df.sort_values(by="total_demand", ascending=True).reset_index(drop=True)
    colors = [BRAND_COLORS.get(b, PRIMARY_COLOR) for b in sorted_df["brand_id"]]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=FIGURE_DPI)
    bars = ax.barh(
        sorted_df["brand_id"],
        sorted_df["total_demand"],
        color=colors,
        height=0.55,
        edgecolor="none",
    )

    for bar, (_, row) in zip(bars, sorted_df.iterrows()):
        w = bar.get_width()
        pct = row["portfolio_demand_share"] * 100
        total_str = f"{int(row['total_demand']):,} units"
        ax.text(
            w + 4000,
            bar.get_y() + bar.get_height() / 2.0,
            f"{total_str} ({pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=9,
            color=NEUTRAL_DARK,
        )

    ax.set_title(
        "Portfolio Demand Contribution by Brand — Development Period",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Total Demand Units (2014–2017)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Brand", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_xlim(0, sorted_df["total_demand"].max() * 1.30)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    apply_clean_layout(ax, grid_axis="x")
    return save_figure(fig, output_path)


def plot_sku_demand_rank(profiles_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> Path:
    """Figure 5: SKU Demand Concentration (Ranked top 15 SKUs).

    Parameters
    ----------
    profiles_df : pd.DataFrame
        SKU profiles DataFrame.
    output_path : Path
        Destination file path.
    top_n : int, optional
        Number of top SKUs to display (default: 15).

    Returns
    -------
    Path
        Saved figure path.
    """
    top_df = profiles_df.sort_values(by="total_demand", ascending=False).head(top_n).copy()
    # Invert order for horizontal bars so highest is on top
    top_df.sort_values(by="total_demand", ascending=True, inplace=True)

    labels = [f"{row['sku_id']} ({row['brand_id']})" for _, row in top_df.iterrows()]
    colors = [BRAND_COLORS.get(row["brand_id"], PRIMARY_COLOR) for _, row in top_df.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=FIGURE_DPI)
    bars = ax.barh(
        labels,
        top_df["total_demand"],
        color=colors,
        height=0.6,
        edgecolor="none",
    )

    for bar, val in zip(bars, top_df["total_demand"]):
        w = bar.get_width()
        ax.text(
            w + 400,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(val):,} units",
            ha="left",
            va="center",
            fontsize=8.5,
            color=NEUTRAL_DARK,
        )

    ax.set_title(
        f"Top {top_n} SKUs by Total Demand — Development Period Concentration",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Total Demand Units (2014–2017)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("SKU (Brand)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_xlim(0, top_df["total_demand"].max() * 1.20)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    apply_clean_layout(ax, grid_axis="x")
    return save_figure(fig, output_path)


def plot_sku_zero_demand_distribution(profiles_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure 6: SKU Sparsity Distribution (Histogram of zero-demand rates).

    Parameters
    ----------
    profiles_df : pd.DataFrame
        SKU profiles DataFrame with zero_demand_rate.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    zero_pct = profiles_df["zero_demand_rate"] * 100.0
    bins = np.arange(0, 85, 5)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=FIGURE_DPI)
    ax.hist(
        zero_pct,
        bins=bins,
        color=PRIMARY_COLOR,
        edgecolor="white",
        linewidth=1.0,
    )

    # Reference analytical boundary thresholds (<10%, 10-25%, 25-50%, >=50%)
    ax.axvline(10, color="#DC2626", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axvline(25, color="#D97706", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axvline(50, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.85)

    # Analytical bucket counts from SKU profiles
    rates = profiles_df["zero_demand_rate"]
    count_lt10 = (rates < 0.10).sum()
    count_10_25 = ((rates >= 0.10) & (rates < 0.25)).sum()
    count_25_50 = ((rates >= 0.25) & (rates < 0.50)).sum()
    count_ge50 = (rates >= 0.50).sum()

    # Provide headroom up to 25 so annotations sit above the tallest bar (height 19)
    ax.set_ylim(0, 25)

    ax.text(5, 22.2, f"<10% Low\n({count_lt10} SKUs)", ha="center", fontsize=8.5, color="#DC2626", fontweight="bold")
    ax.text(17.5, 22.2, f"10–25% Mod\n({count_10_25} SKUs)", ha="center", fontsize=8.5, color="#D97706", fontweight="bold")
    ax.text(37.5, 22.2, f"25–50% High\n({count_25_50} SKUs)", ha="center", fontsize=8.5, color="#475569", fontweight="bold")
    ax.text(62.5, 22.2, f"≥50% V. High\n({count_ge50} SKUs)", ha="center", fontsize=8.5, color=NEUTRAL_DARK, fontweight="bold")

    ax.set_title(
        "Distribution of SKU Zero-Demand Rates — Development Period",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Zero-Demand Rate (%)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Number of SKUs", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_xlim(0, 80)

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, output_path)


def plot_sparsity_vs_demand_volume(profiles_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure 7: Sparsity vs Demand Volume (Descriptive scatter plot).

    Parameters
    ----------
    profiles_df : pd.DataFrame
        SKU profiles DataFrame.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=FIGURE_DPI)

    for brand_id in sorted(profiles_df["brand_id"].unique()):
        sub = profiles_df[profiles_df["brand_id"] == brand_id]
        ax.scatter(
            sub["zero_demand_rate"] * 100.0,
            sub["mean_daily_demand"],
            color=BRAND_COLORS.get(brand_id, PRIMARY_COLOR),
            label=f"Brand {brand_id}",
            s=45,
            alpha=0.8,
            edgecolors="none",
        )

    ax.set_title(
        "SKU Zero-Demand Rate vs Mean Daily Demand — Development Period",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Zero-Demand Rate (%)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Mean Daily Demand (Units)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_xlim(-2, 80)
    ax.set_ylim(-0.5, profiles_df["mean_daily_demand"].max() * 1.08)

    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right", fontsize=8.5)
    apply_clean_layout(ax, grid_axis="both")
    return save_figure(fig, output_path)


def plot_promotion_demand_by_brand(dev_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure 8: Observed Mean Demand by Promotion Status and Brand.

    Includes explicit descriptive non-causal disclaimer.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    brands = sorted(dev_df["brand_id"].unique())
    non_promo_means = [
        float(dev_df[(dev_df["brand_id"] == b) & (dev_df["promotion"] == 0)]["quantity"].mean())
        for b in brands
    ]
    promo_means = [
        float(dev_df[(dev_df["brand_id"] == b) & (dev_df["promotion"] == 1)]["quantity"].mean())
        for b in brands
    ]

    x = np.arange(len(brands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=FIGURE_DPI)
    bars_np = ax.bar(
        x - width / 2.0,
        non_promo_means,
        width,
        label="Non-Promotion Observations",
        color=PROMO_COLORS[0],
        edgecolor="none",
    )
    bars_p = ax.bar(
        x + width / 2.0,
        promo_means,
        width,
        label="Promotion Observations",
        color=PROMO_COLORS[1],
        edgecolor="none",
    )

    for bar, val in zip(bars_np, non_promo_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=NEUTRAL_DARK,
        )

    for bar, val in zip(bars_p, promo_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=NEUTRAL_DARK,
        )

    ax.set_title(
        "Observed Mean Daily Demand: Promotion vs Non-Promotion by Brand",
        fontsize=12,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Brand", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Mean SKU Daily Demand (Units)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(brands)
    ax.set_ylim(0, max(promo_means) * 1.18)

    ax.legend(frameon=False, loc="upper left", fontsize=9)

    # Mandatory visible non-causal note
    ax.text(
        0.5,
        -0.16,
        "Descriptive comparison only; promotion indicator does not establish causal impact.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        fontstyle="italic",
        color=NEUTRAL_MUTED,
    )

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, output_path)


def plot_representative_dense_sku(
    dev_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Figure 9: Representative Dense SKU Time Series (Deterministically Selected).

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.
    profiles_df : pd.DataFrame
        SKU profiles DataFrame.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    brand_id, sku_id = select_representative_dense_sku(profiles_df)
    sku_data = dev_df[(dev_df["brand_id"] == brand_id) & (dev_df["sku_id"] == sku_id)].copy()
    sku_data["date_dt"] = pd.to_datetime(sku_data["date"])
    sku_data.sort_values(by="date_dt", inplace=True)

    profile_row = profiles_df[(profiles_df["brand_id"] == brand_id) & (profiles_df["sku_id"] == sku_id)].iloc[0]
    zero_rate_pct = profile_row["zero_demand_rate"] * 100.0
    mean_daily = profile_row["mean_daily_demand"]
    total_vol = int(profile_row["total_demand"])

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=FIGURE_DPI)
    ax.plot(
        sku_data["date_dt"],
        sku_data["quantity"],
        color=BRAND_COLORS.get(brand_id, PRIMARY_COLOR),
        linewidth=1.0,
        alpha=0.85,
    )

    ax.set_title(
        f"Representative Dense SKU: {sku_id} (Brand {brand_id}) — Development Period\n"
        f"Zero-Demand Rate: {zero_rate_pct:.2f}% | Mean Daily Demand: {mean_daily:.2f} units | Total: {total_vol:,} units",
        fontsize=11,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Date", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Daily Demand (Units)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(
        sku_data["date_dt"].min() - pd.Timedelta(days=5),
        sku_data["date_dt"].max() + pd.Timedelta(days=5),
    )

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, output_path)


def plot_representative_sparse_sku(
    dev_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Figure 10: Representative Sparse SKU Time Series (Deterministically Selected).

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development-period SKU demand DataFrame.
    profiles_df : pd.DataFrame
        SKU profiles DataFrame.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        Saved figure path.
    """
    brand_id, sku_id = select_representative_sparse_sku(profiles_df)
    sku_data = dev_df[(dev_df["brand_id"] == brand_id) & (dev_df["sku_id"] == sku_id)].copy()
    sku_data["date_dt"] = pd.to_datetime(sku_data["date"])
    sku_data.sort_values(by="date_dt", inplace=True)

    profile_row = profiles_df[(profiles_df["brand_id"] == brand_id) & (profiles_df["sku_id"] == sku_id)].iloc[0]
    zero_rate_pct = profile_row["zero_demand_rate"] * 100.0
    mean_daily = profile_row["mean_daily_demand"]
    total_vol = int(profile_row["total_demand"])

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=FIGURE_DPI)
    ax.plot(
        sku_data["date_dt"],
        sku_data["quantity"],
        color=BRAND_COLORS.get(brand_id, PRIMARY_COLOR),
        linewidth=1.0,
        alpha=0.85,
    )

    ax.set_title(
        f"Representative Sparse SKU: {sku_id} (Brand {brand_id}) — Development Period\n"
        f"Zero-Demand Rate: {zero_rate_pct:.2f}% | Mean Daily Demand: {mean_daily:.2f} units | Total: {total_vol:,} units",
        fontsize=11,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=10,
    )
    ax.set_xlabel("Date", fontsize=10, color=NEUTRAL_DARK, labelpad=8)
    ax.set_ylabel("Daily Demand (Units)", fontsize=10, color=NEUTRAL_DARK, labelpad=8)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(
        sku_data["date_dt"].min() - pd.Timedelta(days=5),
        sku_data["date_dt"].max() + pd.Timedelta(days=5),
    )

    apply_clean_layout(ax, grid_axis="y")
    return save_figure(fig, output_path)


def generate_all_eda_figures(
    canonical_path: Path = Path("data/processed/sku_demand_daily.csv"),
    output_dir: Path = Path("reports/figures"),
    cutoff_date: str = DEVELOPMENT_CUTOFF_DATE,
) -> List[Path]:
    """Execute end-to-end visualization pipeline generating all 10 approved EDA figures.

    Parameters
    ----------
    canonical_path : Path
        Path to canonical SKU demand dataset.
    output_dir : Path
        Directory where generated figures will be stored.
    cutoff_date : str
        Development cutoff date (default: DEVELOPMENT_CUTOFF_DATE = '2017-12-31').
        Strictly excludes 2018 holdout data.

    Returns
    -------
    List[Path]
        List of generated figure file paths.

    Raises
    ------
    FileNotFoundError
        If canonical dataset is missing.
    ValueError
        If holdout data (date > cutoff_date) leaks into the development set.
    """
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical dataset not found at {canonical_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading canonical dataset from %s", canonical_path)
    canonical_df = pd.read_csv(canonical_path)

    # Strictly filter to development period (date <= cutoff_date)
    dev_df = filter_development_period(canonical_df, cutoff_date=cutoff_date)

    if (dev_df["date"] > cutoff_date).any():
        raise ValueError(
            f"Holdout leakage detected! Maximum date in development subset is {dev_df['date'].max()}, "
            f"exceeding cutoff {cutoff_date}."
        )

    logger.info(
        "Development period verified: %d rows from %s to %s (%d observed days, %d SKUs).",
        len(dev_df),
        dev_df["date"].min(),
        dev_df["date"].max(),
        dev_df["date"].nunique(),
        dev_df["sku_id"].nunique(),
    )

    # Reuse existing demand profiling logic
    profiles_df = calculate_sku_profiles(dev_df)
    brand_df = calculate_brand_summary(dev_df, profiles_df)

    # Validate that chart data contains no infinite values
    if np.isinf(profiles_df["mean_daily_demand"]).any() or np.isinf(profiles_df["zero_demand_rate"]).any():
        raise ValueError("Infinite values detected in SKU profiles.")

    generated: List[Path] = []

    # 1. Portfolio Demand Over Time
    p1 = plot_portfolio_daily_demand(dev_df, output_dir / "portfolio_daily_demand.png")
    generated.append(p1)

    # 2. Day-of-Week Seasonality
    p2 = plot_day_of_week_seasonality(dev_df, output_dir / "day_of_week_seasonality.png")
    generated.append(p2)

    # 3. Monthly Seasonality
    p3 = plot_monthly_seasonality(dev_df, output_dir / "monthly_seasonality.png")
    generated.append(p3)

    # 4. Brand Demand Share
    p4 = plot_brand_demand_share(brand_df, output_dir / "brand_demand_share.png")
    generated.append(p4)

    # 5. SKU Demand Rank
    p5 = plot_sku_demand_rank(profiles_df, output_dir / "sku_demand_rank.png")
    generated.append(p5)

    # 6. SKU Zero-Demand Distribution
    p6 = plot_sku_zero_demand_distribution(profiles_df, output_dir / "sku_zero_demand_distribution.png")
    generated.append(p6)

    # 7. Sparsity vs Demand Volume
    p7 = plot_sparsity_vs_demand_volume(profiles_df, output_dir / "sparsity_vs_demand_volume.png")
    generated.append(p7)

    # 8. Promotion Demand by Brand
    p8 = plot_promotion_demand_by_brand(dev_df, output_dir / "promotion_demand_by_brand.png")
    generated.append(p8)

    # 9. Representative Dense SKU
    p9 = plot_representative_dense_sku(dev_df, profiles_df, output_dir / "representative_dense_sku.png")
    generated.append(p9)

    # 10. Representative Sparse SKU
    p10 = plot_representative_sparse_sku(dev_df, profiles_df, output_dir / "representative_sparse_sku.png")
    generated.append(p10)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

    logger.info("Successfully generated all %d figures in %s", len(generated), output_dir)
    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
    generate_all_eda_figures()
