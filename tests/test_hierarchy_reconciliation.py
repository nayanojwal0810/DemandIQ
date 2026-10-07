"""Unit and integration test suite for DemandIQ Hierarchical Bottom-Up Reconciliation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.hierarchy.reconcile import (
    EXPECTED_BRANDS,
    EXPECTED_SKU_COUNT,
    EXPECTED_TOTAL_NODES,
    TOTAL_NODE_ID,
    HierarchySpec,
    build_hierarchy_predictions,
    build_summing_matrix,
    load_sku_hierarchy_mapping,
    reconcile_bottom_up,
    verify_forecast_coherence,
)
from src.hierarchy.evaluate_hierarchy import (
    ML_MODELS,
    calculate_hierarchy_metrics,
    calculate_hierarchy_promotion_comparison,
    calculate_hierarchy_summary,
)


@pytest.fixture(scope="module")
def canonical_mapping() -> pd.DataFrame:
    """Fixture providing validated canonical SKU-to-brand mapping."""
    return load_sku_hierarchy_mapping("data/processed/sku_dimension.csv")


@pytest.fixture(scope="module")
def hierarchy_spec(canonical_mapping: pd.DataFrame) -> HierarchySpec:
    """Fixture providing canonical hierarchy specification."""
    return build_summing_matrix(canonical_mapping)


# =============================================================================
# 1. Hierarchy Structure Tests
# =============================================================================


def test_hierarchy_structure_sku_count(canonical_mapping: pd.DataFrame):
    """Verify hierarchy mapping contains exactly 118 unique SKUs."""
    assert len(canonical_mapping) == EXPECTED_SKU_COUNT
    assert canonical_mapping["sku_id"].nunique() == EXPECTED_SKU_COUNT


def test_hierarchy_structure_brand_count(canonical_mapping: pd.DataFrame):
    """Verify hierarchy mapping contains exactly 4 unique brands."""
    brands = sorted(canonical_mapping["brand_id"].unique())
    assert brands == EXPECTED_BRANDS
    assert len(brands) == 4


def test_hierarchy_structure_strict_one_to_one(canonical_mapping: pd.DataFrame):
    """Verify every SKU belongs to exactly one brand with no duplicates or missing values."""
    assert not canonical_mapping["sku_id"].isna().any()
    assert not canonical_mapping["brand_id"].isna().any()

    # No duplicate SKU rows
    assert not canonical_mapping["sku_id"].duplicated().any()

    # Each SKU maps to exactly one brand
    skus_per_brand = canonical_mapping.groupby("sku_id")["brand_id"].nunique()
    assert (skus_per_brand == 1).all()


# =============================================================================
# 2. Summing Matrix Tests
# =============================================================================


def test_summing_matrix_dimensions(hierarchy_spec: HierarchySpec):
    """Verify summing matrix has shape (123, 118)."""
    S = hierarchy_spec.summing_matrix
    assert S.shape == (EXPECTED_TOTAL_NODES, EXPECTED_SKU_COUNT)
    assert len(hierarchy_spec.node_ids) == EXPECTED_TOTAL_NODES
    assert len(hierarchy_spec.bottom_sku_order) == EXPECTED_SKU_COUNT


def test_summing_matrix_deterministic_ordering(hierarchy_spec: HierarchySpec):
    """Verify explicit deterministic node ordering: Total -> Brands -> SKUs."""
    node_ids = hierarchy_spec.node_ids
    assert node_ids[0] == TOTAL_NODE_ID
    assert node_ids[1:5] == EXPECTED_BRANDS
    assert node_ids[5:] == hierarchy_spec.bottom_sku_order
    assert hierarchy_spec.node_levels[0] == "total"
    assert hierarchy_spec.node_levels[1:5] == ["brand"] * 4
    assert hierarchy_spec.node_levels[5:] == ["sku"] * EXPECTED_SKU_COUNT


def test_summing_matrix_row_properties(hierarchy_spec: HierarchySpec, canonical_mapping: pd.DataFrame):
    """Verify mathematical properties of summing matrix S."""
    S = hierarchy_spec.summing_matrix
    n_skus = EXPECTED_SKU_COUNT
    n_brands = len(EXPECTED_BRANDS)

    # Total row sums all 118 SKUs
    assert np.allclose(S[0, :], np.ones(n_skus))
    assert S[0, :].sum() == float(n_skus)

    # Every SKU contributes to exactly one brand row
    brand_block = S[1 : 1 + n_brands, :]
    assert np.allclose(brand_block.sum(axis=0), np.ones(n_skus))

    # Bottom block is exactly the identity matrix (every SKU maps to itself)
    sku_block = S[1 + n_brands :, :]
    assert np.allclose(sku_block, np.eye(n_skus))

    # Verify each brand row has the exact count of its member SKUs
    for b_idx, b in enumerate(EXPECTED_BRANDS):
        expected_cnt = (canonical_mapping["brand_id"] == b).sum()
        actual_cnt = S[1 + b_idx, :].sum()
        assert actual_cnt == expected_cnt


# =============================================================================
# 3. Bottom-Up Mathematical Correctness (Synthetic)
# =============================================================================


def test_bottom_up_synthetic_mathematical_correctness():
    """Verify Bottom-Up reconciliation on a known synthetic toy hierarchy.

    Hierarchy:
        Total
         ├── B1: S1, S2
         └── B2: S3
    """
    mock_mapping = pd.DataFrame(
        {
            "brand_id": ["B1", "B1", "B2"],
            "sku_id": ["S1", "S2", "S3"],
        }
    )
    # Build synthetic spec
    skus = ["S1", "S2", "S3"]
    brands = ["B1", "B2"]
    # Nodes: Total (0), B1 (1), B2 (2), S1 (3), S2 (4), S3 (5)
    S_mock = np.array(
        [
            [1.0, 1.0, 1.0],  # Total
            [1.0, 1.0, 0.0],  # B1
            [0.0, 0.0, 1.0],  # B2
            [1.0, 0.0, 0.0],  # S1
            [0.0, 1.0, 0.0],  # S2
            [0.0, 0.0, 1.0],  # S3
        ]
    )

    # Forecasts for 2 dates:
    # Date 1: S1=10, S2=20, S3=30
    # Date 2: S1=5,  S2=15, S3=25
    B_synthetic = np.array(
        [
            [10.0, 20.0, 30.0],
            [5.0, 15.0, 25.0],
        ]
    )

    Y_synthetic = reconcile_bottom_up(B_synthetic, S_mock)

    assert Y_synthetic.shape == (2, 6)

    # Date 1 checks:
    assert Y_synthetic[0, 0] == 60.0  # Total = 10 + 20 + 30
    assert Y_synthetic[0, 1] == 30.0  # B1 = 10 + 20
    assert Y_synthetic[0, 2] == 30.0  # B2 = 30
    assert np.allclose(Y_synthetic[0, 3:], [10.0, 20.0, 30.0])  # SKUs unchanged

    # Date 2 checks:
    assert Y_synthetic[1, 0] == 45.0  # Total = 5 + 15 + 25
    assert Y_synthetic[1, 1] == 20.0  # B1 = 5 + 15
    assert Y_synthetic[1, 2] == 25.0  # B2 = 25
    assert np.allclose(Y_synthetic[1, 3:], [5.0, 15.0, 25.0])  # SKUs unchanged


# =============================================================================
# 4. Exact Mathematical Coherence & Preservation Tests
# =============================================================================


def test_coherence_verification_rolling(hierarchy_spec: HierarchySpec):
    """Verify zero tolerance violations on rolling validation hierarchy predictions."""
    path = Path("data/processed/hierarchy_predictions_rolling.csv")
    if not path.exists():
        pytest.skip("hierarchy_predictions_rolling.csv not yet generated.")

    df_roll = pd.read_csv(path)
    df_src = pd.read_csv("data/processed/rolling_validation_predictions.csv")

    checks = verify_forecast_coherence(
        df_roll,
        model_cols=ML_MODELS,
        tolerance=1e-10,
        dataset_name="rolling",
        df_sku_source=df_src,
    )
    df_checks = pd.DataFrame(checks)

    assert df_checks["passed"].all()
    assert (df_checks["violating_rows"] == 0).all()
    assert (df_checks["max_abs_residual"] <= 1e-10).all()


def test_coherence_verification_2017(hierarchy_spec: HierarchySpec):
    """Verify zero tolerance violations on 2017 validation hierarchy predictions."""
    path = Path("data/processed/hierarchy_predictions_2017.csv")
    if not path.exists():
        pytest.skip("hierarchy_predictions_2017.csv not yet generated.")

    df_2017 = pd.read_csv(path)
    df_src = pd.read_csv("data/processed/ml_predictions_validation.csv")

    checks = verify_forecast_coherence(
        df_2017,
        model_cols=ML_MODELS,
        tolerance=1e-10,
        dataset_name="2017",
        df_sku_source=df_src,
    )
    df_checks = pd.DataFrame(checks)

    assert df_checks["passed"].all()
    assert (df_checks["violating_rows"] == 0).all()
    assert (df_checks["max_abs_residual"] <= 1e-10).all()


# =============================================================================
# 5. Temporal Integrity & 2018 Sealing Tests
# =============================================================================


def test_temporal_integrity_no_2018_in_predictions():
    """Verify that neither rolling nor 2017 hierarchy predictions contain 2018 dates."""
    for filename in ["hierarchy_predictions_rolling.csv", "hierarchy_predictions_2017.csv"]:
        path = Path("data/processed") / filename
        if path.exists():
            df = pd.read_csv(path)
            dates = pd.to_datetime(df["date"])
            assert (dates < pd.Timestamp("2018-01-01")).all(), f"Found 2018 dates in {filename}!"


def test_temporal_integrity_loud_failure_on_2018_data(hierarchy_spec: HierarchySpec):
    """Verify build_hierarchy_predictions fails loudly if 2018 data is passed."""
    mock_df = pd.DataFrame(
        {
            "date": ["2018-01-02"] * EXPECTED_SKU_COUNT,
            "brand_id": ["B1"] * EXPECTED_SKU_COUNT,
            "sku_id": hierarchy_spec.bottom_sku_order,
            "actual": [1.0] * EXPECTED_SKU_COUNT,
            "xgboost_no_target_promo": [1.0] * EXPECTED_SKU_COUNT,
            "xgboost_with_target_promo": [1.0] * EXPECTED_SKU_COUNT,
            "lightgbm_no_target_promo": [1.0] * EXPECTED_SKU_COUNT,
            "lightgbm_with_target_promo": [1.0] * EXPECTED_SKU_COUNT,
        }
    )
    with pytest.raises(ValueError, match="2018 holdout is strictly sealed"):
        build_hierarchy_predictions(mock_df, hierarchy_spec, is_rolling=False)


def test_rolling_folds_unchanged():
    """Verify that rolling folds remain exactly 1 through 8 matching Stage 9."""
    path = Path("data/processed/hierarchy_predictions_rolling.csv")
    if not path.exists():
        pytest.skip("hierarchy_predictions_rolling.csv not yet generated.")

    df = pd.read_csv(path)
    assert sorted(df["fold"].unique().tolist()) == [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# 6. Forecast Integrity Tests
# =============================================================================


def test_forecast_integrity_non_negative_and_no_nan():
    """Verify that all four ML hierarchy forecasts are non-negative and finite."""
    for filename in ["hierarchy_predictions_rolling.csv", "hierarchy_predictions_2017.csv"]:
        path = Path("data/processed") / filename
        if path.exists():
            df = pd.read_csv(path)
            for m in ML_MODELS:
                assert not df[m].isna().any(), f"NaN values in {filename} for {m}"
                assert not np.isinf(df[m]).any(), f"Inf values in {filename} for {m}"
                assert (df[m] >= 0.0).all(), f"Negative values in {filename} for {m}"


def test_forecast_integrity_bottom_sku_identity(hierarchy_spec: HierarchySpec):
    """Verify bottom SKU predictions before and after reconciliation are numerically identical."""
    path_reconciled = Path("data/processed/hierarchy_predictions_rolling.csv")
    path_source = Path("data/processed/rolling_validation_predictions.csv")
    if not path_reconciled.exists():
        pytest.skip("hierarchy_predictions_rolling.csv not yet generated.")

    df_rec = pd.read_csv(path_reconciled)
    df_src = pd.read_csv(path_source)

    df_rec_sku = df_rec[df_rec["hierarchy_level"] == "sku"].sort_values(["date", "node_id"]).reset_index(drop=True)
    df_src_sorted = df_src.sort_values(["date", "sku_id"]).reset_index(drop=True)

    for m in ML_MODELS:
        diff = np.abs(df_rec_sku[m].values - df_src_sorted[m].values)
        assert diff.max() <= 1e-12, f"Bottom SKU predictions mutated for {m} (max diff {diff.max()})!"


# =============================================================================
# 7. Metric Integrity & Ranking Tests
# =============================================================================


def test_metrics_match_repository_utility_definitions():
    """Verify hierarchical metrics accurately reuse repo metric formulas."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    expected_wape = (2.0 + 2.0 + 3.0) / 60.0
    expected_mae = (2.0 + 2.0 + 3.0) / 3.0
    expected_rmse = np.sqrt((4.0 + 4.0 + 9.0) / 3.0)

    assert np.isclose(calculate_wape(y_true, y_pred), expected_wape)
    assert np.isclose(calculate_mae(y_true, y_pred), expected_mae)
    assert np.isclose(calculate_rmse(y_true, y_pred), expected_rmse)


def test_hierarchy_summary_structure_and_ranking():
    """Verify hierarchy summary correctly aggregates ranks and fold wins."""
    path = Path("data/processed/hierarchy_summary.csv")
    if not path.exists():
        pytest.skip("hierarchy_summary.csv not yet generated.")

    df_summary = pd.read_csv(path)
    assert len(df_summary) == 12  # 3 levels * 4 models
    assert set(df_summary["hierarchy_level"].unique()) == {"total", "brand", "sku"}

    for level in ["total", "brand", "sku"]:
        sub = df_summary[df_summary["hierarchy_level"] == level]
        # Sum of fold wins across 4 models must equal 8
        assert sub["fold_wins"].sum() == 8
        # Model with rank 1 should have lowest pooled_wape
        assert sub.iloc[0]["pooled_wape"] == sub["pooled_wape"].min()


# =============================================================================
# 8. Promotion Comparison Tests
# =============================================================================


def test_promotion_comparison_structure_and_deltas():
    """Verify promotion comparison pairs models properly and computes full precision deltas."""
    path = Path("data/processed/hierarchy_promotion_comparison.csv")
    if not path.exists():
        pytest.skip("hierarchy_promotion_comparison.csv not yet generated.")

    df_promo = pd.read_csv(path)
    assert len(df_promo) == 12  # 2 periods * 3 levels * 2 model families
    assert set(df_promo["model_family"].unique()) == {"xgboost", "lightgbm"}

    for _, row in df_promo.iterrows():
        expected_abs = row["wape_with_promo"] - row["wape_no_promo"]
        expected_rel = expected_abs / row["wape_no_promo"]
        assert np.isclose(row["abs_wape_diff"], expected_abs, atol=1e-5)
        assert np.isclose(row["rel_wape_diff"], expected_rel, atol=1e-5)
        # In this dataset, promotion-awareness consistently reduces WAPE across all levels
        assert row["abs_wape_diff"] < 0.0
