"""Plot utilities, style configurations, and deterministic SKU selection for DemandIQ."""

from pathlib import Path
from typing import Dict, Tuple
import logging

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Centrally defined color palette
BRAND_COLORS: Dict[str, str] = {
    "B1": "#2563EB",  # Royal Blue
    "B2": "#059669",  # Emerald Green
    "B3": "#D97706",  # Amber
    "B4": "#7C3AED",  # Violet
}

PROMO_COLORS: Dict[int, str] = {
    0: "#64748B",  # Slate Gray (Non-Promotion)
    1: "#EA580C",  # Warm Orange (Promotion)
}

PRIMARY_COLOR: str = "#1E40AF"    # Deep Blue
SECONDARY_COLOR: str = "#0D9488"  # Teal
NEUTRAL_DARK: str = "#0F172A"     # Slate 900
NEUTRAL_MUTED: str = "#475569"    # Slate 600
GRID_COLOR: str = "#E2E8F0"       # Slate 200
FIGURE_DPI: int = 300


def apply_clean_layout(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply clean, publication-grade styling to a matplotlib Axes instance.

    Removes top and right spines, tints remaining spines, applies subtle
    gridlines, and styles tick labels.

    Parameters
    ----------
    ax : plt.Axes
        Target matplotlib axes.
    grid_axis : str, optional
        Axis on which to draw subtle gridlines ('y', 'x', or 'both'). Default 'y'.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if grid_axis in ("y", "x", "both"):
        ax.grid(
            axis=grid_axis,
            linestyle="--",
            linewidth=0.7,
            alpha=0.7,
            color=GRID_COLOR,
        )
    ax.set_axisbelow(True)
    ax.tick_params(colors=NEUTRAL_MUTED, labelsize=9)


def save_figure(fig: plt.Figure, filepath: Path, dpi: int = FIGURE_DPI) -> Path:
    """Save matplotlib figure deterministically with standard options and close figure.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure instance.
    filepath : Path
        Destination path.
    dpi : int, optional
        DPI resolution (default: 300).

    Returns
    -------
    Path
        Resolved saved filepath.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    logger.info("Saved figure: %s (%dx%d dpi=%d)", filepath, fig.get_figwidth(), fig.get_figheight(), dpi)
    return filepath


def select_representative_dense_sku(profiles_df: pd.DataFrame) -> Tuple[str, str]:
    """Deterministically select a representative dense SKU from development profiles.

    Selection Rule:
    1. Minimum zero-demand rate (highest continuous sales presence).
    2. Highest total demand (tie-break).
    3. Lexicographical sku_id ascending (deterministic tie-break).

    Parameters
    ----------
    profiles_df : pd.DataFrame
        SKU demand profiles DataFrame containing [brand_id, sku_id, zero_demand_rate, total_demand].

    Returns
    -------
    Tuple[str, str]
        (brand_id, sku_id) of the selected dense SKU.
    """
    sorted_df = profiles_df.sort_values(
        by=["zero_demand_rate", "total_demand", "sku_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    selected = sorted_df.iloc[0]
    return str(selected["brand_id"]), str(selected["sku_id"])


def select_representative_sparse_sku(profiles_df: pd.DataFrame) -> Tuple[str, str]:
    """Deterministically select a representative sparse/intermittent SKU from development profiles.

    Selection Rule:
    1. Maximum zero-demand rate (highest proportion of non-sales days).
    2. Lowest total demand (tie-break).
    3. Lexicographical sku_id ascending (deterministic tie-break).

    Parameters
    ----------
    profiles_df : pd.DataFrame
        SKU demand profiles DataFrame containing [brand_id, sku_id, zero_demand_rate, total_demand].

    Returns
    -------
    Tuple[str, str]
        (brand_id, sku_id) of the selected sparse SKU.
    """
    sorted_df = profiles_df.sort_values(
        by=["zero_demand_rate", "total_demand", "sku_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    selected = sorted_df.iloc[0]
    return str(selected["brand_id"]), str(selected["sku_id"])
