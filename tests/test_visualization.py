"""Tests for visualization and exploratory analysis figures.

Verifies behavioral integrity, development cutoff enforcement, deterministic SKU selection,
artifact completeness, and non-mutation of canonical source data.
"""

from pathlib import Path
import hashlib
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.features.feature_config import DEVELOPMENT_CUTOFF_DATE
from src.business.demand_profile import (
    filter_development_period,
    calculate_sku_profiles,
    calculate_brand_summary,
)
from src.visualization.plot_utils import (
    select_representative_dense_sku,
    select_representative_sparse_sku,
    BRAND_COLORS,
    PROMO_COLORS,
)
from src.visualization.create_eda_figures import (
    FIGURE_NAMES,
    generate_all_eda_figures,
    calculate_portfolio_calendar_rolling_mean,
)

CANONICAL_PATH = Path("data/processed/sku_demand_daily.csv")


def test_visualization_module_imports():
    """Verify visualization modules import cleanly with required attributes."""
    import src.visualization.create_eda_figures as eda
    import src.visualization.plot_utils as utils

    assert hasattr(eda, "generate_all_eda_figures")
    assert hasattr(eda, "calculate_portfolio_calendar_rolling_mean")
    assert hasattr(eda, "FIGURE_NAMES")
    assert hasattr(utils, "select_representative_dense_sku")
    assert hasattr(utils, "select_representative_sparse_sku")
    assert hasattr(utils, "BRAND_COLORS")
    assert hasattr(utils, "PROMO_COLORS")
    assert len(eda.FIGURE_NAMES) == 10


def test_development_cutoff_enforced():
    """Verify that development filtering strictly bounds data at 2017-12-31 with no 2018 data."""
    assert CANONICAL_PATH.exists(), f"Missing canonical data at {CANONICAL_PATH}"
    canonical_df = pd.read_csv(CANONICAL_PATH)

    # Raw canonical contains full history including 2018 holdout
    assert canonical_df["date"].max() > DEVELOPMENT_CUTOFF_DATE

    # Filtered development subset must strictly exclude 2018
    dev_df = filter_development_period(canonical_df, cutoff_date=DEVELOPMENT_CUTOFF_DATE)
    assert dev_df["date"].max() <= DEVELOPMENT_CUTOFF_DATE
    assert not (dev_df["date"] > DEVELOPMENT_CUTOFF_DATE).any()
    assert not dev_df["date"].str.startswith("2018").any()


def test_all_required_figure_files_created_and_names_match(tmp_path):
    """Verify generate_all_eda_figures creates all 10 approved figures with expected names."""
    generated_paths = generate_all_eda_figures(
        canonical_path=CANONICAL_PATH,
        output_dir=tmp_path,
        cutoff_date=DEVELOPMENT_CUTOFF_DATE,
    )

    assert len(generated_paths) == 10
    generated_names = [p.name for p in generated_paths]
    assert sorted(generated_names) == sorted(FIGURE_NAMES)

    for fig_name in FIGURE_NAMES:
        fig_file = tmp_path / fig_name
        assert fig_file.exists(), f"Figure file {fig_name} was not created in output directory."


def test_expected_figure_count_and_output_directory_clean(tmp_path):
    """Verify that the output directory contains exactly the 10 expected figure files."""
    generate_all_eda_figures(
        canonical_path=CANONICAL_PATH,
        output_dir=tmp_path,
        cutoff_date=DEVELOPMENT_CUTOFF_DATE,
    )

    all_files = [p.name for p in tmp_path.iterdir() if p.is_file()]
    assert len(all_files) == 10
    assert set(all_files) == set(FIGURE_NAMES)


def test_generated_image_files_non_empty(tmp_path):
    """Verify all generated figure files are valid, non-empty PNG files."""
    generate_all_eda_figures(
        canonical_path=CANONICAL_PATH,
        output_dir=tmp_path,
        cutoff_date=DEVELOPMENT_CUTOFF_DATE,
    )

    for fig_name in FIGURE_NAMES:
        fig_path = tmp_path / fig_name
        size = fig_path.stat().st_size
        assert size > 10_000, f"Figure {fig_name} is unexpectedly small: {size} bytes"

        # Check PNG header signature: \x89PNG\r\n\x1a\n
        with open(fig_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"File {fig_name} does not have a valid PNG header"


def test_representative_dense_sku_selection_deterministic():
    """Verify dense SKU selection is deterministic and permutation-invariant."""
    canonical_df = pd.read_csv(CANONICAL_PATH)
    dev_df = filter_development_period(canonical_df, cutoff_date=DEVELOPMENT_CUTOFF_DATE)
    profiles_df = calculate_sku_profiles(dev_df)

    res1 = select_representative_dense_sku(profiles_df)
    res2 = select_representative_dense_sku(profiles_df)
    assert res1 == res2
    assert res1 == ("B2", "B2_15")

    # Permutation invariance: shuffling profiles DataFrame yields identical selection
    shuffled_df = profiles_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    res_shuffled = select_representative_dense_sku(shuffled_df)
    assert res_shuffled == res1


def test_representative_sparse_sku_selection_deterministic():
    """Verify sparse SKU selection is deterministic and permutation-invariant."""
    canonical_df = pd.read_csv(CANONICAL_PATH)
    dev_df = filter_development_period(canonical_df, cutoff_date=DEVELOPMENT_CUTOFF_DATE)
    profiles_df = calculate_sku_profiles(dev_df)

    res1 = select_representative_sparse_sku(profiles_df)
    res2 = select_representative_sparse_sku(profiles_df)
    assert res1 == res2
    assert res1 == ("B2", "B2_19")

    # Permutation invariance: shuffling profiles DataFrame yields identical selection
    shuffled_df = profiles_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    res_shuffled = select_representative_sparse_sku(shuffled_df)
    assert res_shuffled == res1


def test_representative_dense_sku_satisfies_density_rule():
    """Verify selected dense SKU satisfies strict density rule (< 5% zero-demand)."""
    canonical_df = pd.read_csv(CANONICAL_PATH)
    dev_df = filter_development_period(canonical_df, cutoff_date=DEVELOPMENT_CUTOFF_DATE)
    profiles_df = calculate_sku_profiles(dev_df)

    brand_id, sku_id = select_representative_dense_sku(profiles_df)
    row = profiles_df[(profiles_df["brand_id"] == brand_id) & (profiles_df["sku_id"] == sku_id)].iloc[0]

    # Must satisfy density rule (< 0.05 zero rate)
    assert row["zero_demand_rate"] < 0.05
    # Must be the minimum zero-demand rate in the portfolio
    assert row["zero_demand_rate"] == profiles_df["zero_demand_rate"].min()


def test_representative_sparse_sku_satisfies_sparsity_rule():
    """Verify selected sparse SKU satisfies strict sparsity rule (>= 50% zero-demand)."""
    canonical_df = pd.read_csv(CANONICAL_PATH)
    dev_df = filter_development_period(canonical_df, cutoff_date=DEVELOPMENT_CUTOFF_DATE)
    profiles_df = calculate_sku_profiles(dev_df)

    brand_id, sku_id = select_representative_sparse_sku(profiles_df)
    row = profiles_df[(profiles_df["brand_id"] == brand_id) & (profiles_df["sku_id"] == sku_id)].iloc[0]

    # Must satisfy sparsity rule (>= 0.50 zero rate)
    assert row["zero_demand_rate"] >= 0.50
    # Must be the maximum zero-demand rate in the portfolio
    assert row["zero_demand_rate"] == profiles_df["zero_demand_rate"].max()


def test_chart_summary_data_no_infinite_or_nan():
    """Verify analytical summaries contain no infinite values or unexpected NaNs."""
    canonical_df = pd.read_csv(CANONICAL_PATH)
    dev_df = filter_development_period(canonical_df, cutoff_date=DEVELOPMENT_CUTOFF_DATE)
    profiles_df = calculate_sku_profiles(dev_df)
    brand_df = calculate_brand_summary(dev_df, profiles_df)

    # Check profiles
    assert not np.isinf(profiles_df["mean_daily_demand"]).any()
    assert not np.isinf(profiles_df["zero_demand_rate"]).any()
    assert not np.isinf(profiles_df["total_demand"]).any()
    assert not profiles_df["mean_daily_demand"].isna().any()
    assert not profiles_df["zero_demand_rate"].isna().any()

    # Check brand summary
    assert not np.isinf(brand_df["total_demand"]).any()
    assert not np.isinf(brand_df["portfolio_demand_share"]).any()
    assert not brand_df["total_demand"].isna().any()
    assert not brand_df["portfolio_demand_share"].isna().any()


def test_source_canonical_dataset_not_mutated(tmp_path):
    """Verify running the visualization pipeline does not mutate the source dataset."""
    assert CANONICAL_PATH.exists()

    # Compute checksum and shape before run
    with open(CANONICAL_PATH, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest()
    shape_before = pd.read_csv(CANONICAL_PATH).shape

    # Execute visualization
    generate_all_eda_figures(
        canonical_path=CANONICAL_PATH,
        output_dir=tmp_path,
        cutoff_date=DEVELOPMENT_CUTOFF_DATE,
    )

    # Check checksum and shape after run
    with open(CANONICAL_PATH, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest()
    shape_after = pd.read_csv(CANONICAL_PATH).shape

    assert hash_before == hash_after
    assert shape_before == shape_after


def test_missing_input_file_fails_cleanly(tmp_path):
    """Verify generate_all_eda_figures raises FileNotFoundError on missing input."""
    nonexistent = tmp_path / "missing_sku_demand.csv"
    with pytest.raises(FileNotFoundError):
        generate_all_eda_figures(
            canonical_path=nonexistent,
            output_dir=tmp_path,
            cutoff_date=DEVELOPMENT_CUTOFF_DATE,
        )


def test_holdout_leakage_detection_raises_error(tmp_path, monkeypatch):
    """Verify ValueError is raised if holdout records leak past the development cutoff."""
    import src.visualization.create_eda_figures as eda

    mock_df = pd.DataFrame({
        "date": ["2017-12-31", "2018-01-01"],
        "brand_id": ["B1", "B1"],
        "sku_id": ["B1_1", "B1_1"],
        "quantity": [10, 20],
        "promotion": [0, 1],
    })
    mock_csv = tmp_path / "mock_demand.csv"
    mock_df.to_csv(mock_csv, index=False)

    # Simulate leakage by bypassing date filter
    monkeypatch.setattr(eda, "filter_development_period", lambda df, cutoff_date: df)

    with pytest.raises(ValueError, match="Holdout leakage detected"):
        eda.generate_all_eda_figures(
            canonical_path=mock_csv,
            output_dir=tmp_path / "figs",
            cutoff_date=DEVELOPMENT_CUTOFF_DATE,
        )


def test_portfolio_calendar_rolling_window_semantics():
    """Verify portfolio rolling average uses exact trailing calendar days [t-28D, t-1D].

    Tests behavioral differences between calendar window and observed rows:
    1. Calendar window [t - 28 days, t - 1 day] strictly bounds historical dates.
    2. Missing calendar dates are not imputed as zero demand.
    3. The target date itself is excluded (closed='left').
    4. Row-based rolling yields a different window and value across calendar gaps.
    """
    # Synthetic calendar with gap between 2020-01-03 and 2020-01-11 (7 missing calendar days)
    # Target date: 2020-02-01
    # Trailing 28-calendar-day window [t-28D, t-1D] is [2020-01-04, 2020-01-31].
    dates = [
        "2020-01-01",  # t - 31 days (outside 28-day calendar window)
        "2020-01-02",  # t - 30 days (outside 28-day calendar window)
        "2020-01-03",  # t - 29 days (outside 28-day calendar window)
        # Gap: 2020-01-04 through 2020-01-10 (7 calendar days unobserved)
        "2020-01-11",  # t - 21 days (inside 28-day window)
        "2020-01-12",
        "2020-01-13",
        "2020-01-14",
        "2020-01-15",
        "2020-01-16",
        "2020-01-17",
        "2020-01-18",
        "2020-01-19",
        "2020-01-20",
        "2020-01-21",
        "2020-01-22",
        "2020-01-23",
        "2020-01-24",
        "2020-01-25",
        "2020-01-26",
        "2020-01-27",
        "2020-01-28",
        "2020-01-29",
        "2020-01-30",
        "2020-01-31",  # t - 1 day (inside 28-day window)
        "2020-02-01",  # Target date t (must be excluded from its own window)
    ]
    # Set values: 100 for dates outside window, 10 for dates inside window, 500 for target date
    quantities = [100, 100, 100] + [10] * 21 + [500]
    synth_df = pd.DataFrame({"date": dates, "quantity": quantities})

    # Compute calendar rolling mean [t - 28D, t - 1D]
    rolling_series = calculate_portfolio_calendar_rolling_mean(
        synth_df, window_days=28, min_periods=7
    )

    target_dt = pd.to_datetime("2020-02-01")
    actual_rolling_mean = rolling_series.loc[target_dt]

    # Explicit calculation:
    # Observations inside [2020-01-04, 2020-01-31]: exactly 21 observations of 10
    # Mean of observed dates in window: (21 * 10) / 21 = 10.0
    expected_calendar_mean = 10.0
    assert actual_rolling_mean == expected_calendar_mean

    # Verification 1: Target date (quantity 500) is excluded
    assert actual_rolling_mean != 500.0

    # Verification 2: Missing calendar dates are NOT imputed as zero
    # (If missing 7 days were zeros, mean would be 210 / 28 = 7.5)
    zero_imputed_mean = 7.5
    assert actual_rolling_mean != zero_imputed_mean

    # Verification 3: 28 observed rows produces a different value (includes older 100s)
    # Prior to target, 24 rows exist: (3 * 100 + 21 * 10) / 24 = 21.25
    row_rolling_mean = 21.25
    assert actual_rolling_mean != row_rolling_mean
