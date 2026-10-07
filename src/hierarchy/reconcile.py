"""Hierarchical forecasting Bottom-Up reconciliation and summing matrix module for DemandIQ.

Defines and enforces the mathematical hierarchy structure:
    Total (1 node)
      ├── Brand B1..B4 (4 nodes)
      │     └── SKUs (118 nodes)
Total nodes: 123 nodes (1 Total, 4 Brands, 118 SKUs).

Mathematical Definition:
Let b_t be the 118-dimensional vector of bottom-level SKU forecasts at date t.
Let S be the fixed (123, 118) summing matrix.
The coherent hierarchical forecast vector is:
    y_hat_t = S b_t
where:
    y_hat_t,Total = sum_{j=1}^{118} b_{t,j}
    y_hat_t,Brand = sum_{j in Brand} b_{t,j}
    y_hat_t,SKU   = b_{t,SKU}  (exact identity preservation)

Methodological Principle:
Bottom-Up reconciliation guarantees mathematical coherence by construction.
It preserves existing SKU forecasts exactly. Higher-level forecasting accuracy
is an empirical outcome to be evaluated, not an assumed property.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("hierarchy_reconcile")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

UnionPath = Union[Path, str]

TOTAL_NODE_ID: str = "Total"
EXPECTED_BRANDS: List[str] = ["B1", "B2", "B3", "B4"]
EXPECTED_SKU_COUNT: int = 118
EXPECTED_TOTAL_NODES: int = 123  # 1 Total + 4 Brands + 118 SKUs


@dataclass(frozen=True)
class HierarchySpec:
    """Specification of the full three-level demand forecasting hierarchy.

    Attributes
    ----------
    summing_matrix : np.ndarray
        Fixed summing matrix S of shape (123, 118).
    node_ids : List[str]
        Ordered list of all 123 hierarchy node identifiers.
    node_levels : List[str]
        Ordered list of hierarchy levels ('total', 'brand', 'sku').
    node_parent_ids : List[str]
        Ordered list of parent node identifiers ('' for Total, 'Total' for Brands, brand_id for SKUs).
    bottom_sku_order : List[str]
        Ordered list of 118 bottom-level SKU identifiers matching column order of S.
    """

    summing_matrix: np.ndarray
    node_ids: List[str]
    node_levels: List[str]
    node_parent_ids: List[str]
    bottom_sku_order: List[str]


def load_sku_hierarchy_mapping(
    dim_path: UnionPath = "data/processed/sku_dimension.csv",
) -> pd.DataFrame:
    """Load and validate the canonical SKU-to-brand mapping table.

    Validates:
    1. Exactly 118 unique SKUs
    2. Exactly 4 unique Brands ('B1', 'B2', 'B3', 'B4')
    3. Each SKU belongs to exactly one Brand (no multiple memberships)
    4. No missing values

    Parameters
    ----------
    dim_path : UnionPath, default "data/processed/sku_dimension.csv"
        Path to canonical SKU dimension CSV.

    Returns
    -------
    pd.DataFrame
        Validated mapping DataFrame with columns ['brand_id', 'sku_id'],
        sorted deterministically by ['brand_id', 'sku_id'].
    """
    path = Path(dim_path)
    if not path.exists():
        raise FileNotFoundError(f"SKU dimension file not found: {path}")

    df = pd.read_csv(path)
    required = {"brand_id", "sku_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: {required - set(df.columns)}")

    # Clean and check nulls
    df = df[["brand_id", "sku_id"]].dropna().drop_duplicates()

    # Structural validations
    n_skus = df["sku_id"].nunique()
    if n_skus != EXPECTED_SKU_COUNT:
        raise ValueError(f"Expected {EXPECTED_SKU_COUNT} unique SKUs, got {n_skus}")

    brands = sorted(df["brand_id"].unique())
    if brands != EXPECTED_BRANDS:
        raise ValueError(f"Expected brands {EXPECTED_BRANDS}, got {brands}")

    # Check 1-to-1 SKU -> Brand relationship
    skus_per_brand = df.groupby("sku_id")["brand_id"].nunique()
    if (skus_per_brand > 1).any():
        multi_brand_skus = skus_per_brand[skus_per_brand > 1].index.tolist()
        raise ValueError(f"SKUs mapped to multiple brands: {multi_brand_skus}")

    df_sorted = df.sort_values(["brand_id", "sku_id"]).reset_index(drop=True)
    logger.info("Validated hierarchy mapping: %d SKUs across %d brands.", len(df_sorted), len(brands))
    return df_sorted


def build_summing_matrix(sku_mapping: pd.DataFrame) -> HierarchySpec:
    """Construct the deterministic summing matrix S and node metadata.

    The summing matrix S has shape (123, 118):
    - Row 0: Total (all 1.0s)
    - Rows 1..4: Brands B1..B4 (1.0 for SKUs belonging to the Brand, else 0.0)
    - Rows 5..122: SKUs (118x118 Identity matrix)

    Parameters
    ----------
    sku_mapping : pd.DataFrame
        Validated SKU to Brand mapping.

    Returns
    -------
    HierarchySpec
        Dataclass containing S, node_ids, node_levels, node_parent_ids, and bottom_sku_order.
    """
    skus = sorted(sku_mapping["sku_id"].unique())
    brands = sorted(sku_mapping["brand_id"].unique())

    n_skus = len(skus)
    n_brands = len(brands)
    n_total_nodes = 1 + n_brands + n_skus

    assert n_total_nodes == EXPECTED_TOTAL_NODES, f"Expected {EXPECTED_TOTAL_NODES} nodes, got {n_total_nodes}"

    node_ids: List[str] = [TOTAL_NODE_ID] + brands + skus
    node_levels: List[str] = ["total"] + ["brand"] * n_brands + ["sku"] * n_skus

    # Build parent mapping
    sku_to_brand = dict(zip(sku_mapping["sku_id"], sku_mapping["brand_id"]))
    node_parent_ids: List[str] = [""] + [TOTAL_NODE_ID] * n_brands + [sku_to_brand[s] for s in skus]

    sku_idx = {s: i for i, s in enumerate(skus)}
    S = np.zeros((n_total_nodes, n_skus), dtype=float)

    # 1. Total row (sum of all SKUs)
    S[0, :] = 1.0

    # 2. Brand rows (sum of member SKUs)
    for b_i, b in enumerate(brands):
        member_skus = sku_mapping[sku_mapping["brand_id"] == b]["sku_id"].tolist()
        for s in member_skus:
            S[1 + b_i, sku_idx[s]] = 1.0

    # 3. SKU rows (identity matrix)
    for s_i, s in enumerate(skus):
        S[1 + n_brands + s_i, sku_idx[s]] = 1.0

    # Sanity checks on S
    assert S.shape == (EXPECTED_TOTAL_NODES, EXPECTED_SKU_COUNT)
    assert np.allclose(S[0, :].sum(), float(EXPECTED_SKU_COUNT))
    assert np.allclose(S[1 : 1 + n_brands, :].sum(axis=0), 1.0), "Every SKU must belong to exactly one brand row"
    assert np.allclose(S[1 + n_brands :, :], np.eye(n_skus)), "Bottom block must be identity matrix"

    logger.info("Constructed summing matrix S of shape (%d, %d).", S.shape[0], S.shape[1])
    return HierarchySpec(
        summing_matrix=S,
        node_ids=node_ids,
        node_levels=node_levels,
        node_parent_ids=node_parent_ids,
        bottom_sku_order=skus,
    )


def reconcile_bottom_up(
    bottom_forecasts: np.ndarray,
    summing_matrix: np.ndarray,
) -> np.ndarray:
    """Perform Bottom-Up reconciliation via summing matrix multiplication.

    Computes:
        Y_hat = B @ S.T
    where B has shape (T, 118) and S has shape (123, 118),
    yielding reconciled forecasts Y_hat of shape (T, 123).

    Parameters
    ----------
    bottom_forecasts : np.ndarray
        Array of shape (T, 118) containing bottom-level SKU forecasts.
    summing_matrix : np.ndarray
        Summing matrix S of shape (123, 118).

    Returns
    -------
    np.ndarray
        Reconciled hierarchy forecast array of shape (T, 123).
    """
    B = np.asarray(bottom_forecasts, dtype=float)
    S = np.asarray(summing_matrix, dtype=float)

    if B.ndim == 1:
        B = B.reshape(1, -1)

    if B.shape[1] != S.shape[1]:
        raise ValueError(f"Dimension mismatch: bottom forecasts have {B.shape[1]} cols, S expects {S.shape[1]}")

    Y_hat = B @ S.T
    return Y_hat


def build_hierarchy_predictions(
    df_sku_preds: pd.DataFrame,
    spec: HierarchySpec,
    is_rolling: bool = False,
    model_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Construct deterministic hierarchically reconciled forecasts and actuals.

    Applies the summing matrix S to bottom-level SKU predictions and actuals:
        Y_hat = B @ S.T
    preserving bottom-level SKU forecasts identically, and constructing exactly
    coherent Brand and Total forecasts.

    Parameters
    ----------
    df_sku_preds : pd.DataFrame
        Bottom-level SKU predictions table containing:
        ['date', 'brand_id', 'sku_id', 'actual'] + model_cols (and 'fold' if is_rolling).
    spec : HierarchySpec
        Validated hierarchy specification with summing matrix S.
    is_rolling : bool, default False
        Whether the predictions are from rolling validation (requires 'fold' column).
    model_cols : Optional[List[str]], default None
        List of model forecast column names to reconcile. Defaults to the four Stage 8 ML models.

    Returns
    -------
    pd.DataFrame
        Complete hierarchy predictions table ordered deterministically by date and node hierarchy.
    """
    if model_cols is None:
        model_cols = [
            "xgboost_no_target_promo",
            "xgboost_with_target_promo",
            "lightgbm_no_target_promo",
            "lightgbm_with_target_promo",
        ]

    # Guard against 2018 contamination
    dates_raw = pd.to_datetime(df_sku_preds["date"])
    if (dates_raw >= pd.Timestamp("2018-01-01")).any():
        raise ValueError("2018 holdout is strictly sealed! Dates >= 2018-01-01 encountered in hierarchy input.")

    required = ["date", "brand_id", "sku_id", "actual"] + model_cols
    if is_rolling:
        required = ["fold"] + required

    missing = set(required) - set(df_sku_preds.columns)
    if missing:
        raise ValueError(f"Missing required columns in prediction input: {missing}")

    dates = sorted(df_sku_preds["date"].unique())
    n_dates = len(dates)
    n_nodes = len(spec.node_ids)

    # Check that each date has all 118 SKUs
    skus_per_date = df_sku_preds.groupby("date")["sku_id"].nunique()
    if not (skus_per_date == len(spec.bottom_sku_order)).all():
        bad_dates = skus_per_date[skus_per_date != len(spec.bottom_sku_order)].index.tolist()
        raise ValueError(f"Dates with incomplete SKU coverage: {bad_dates[:5]}")

    all_target_cols = ["actual"] + model_cols
    reconciled_arrays: Dict[str, np.ndarray] = {}

    for col in all_target_cols:
        piv = df_sku_preds.pivot(index="date", columns="sku_id", values=col)[spec.bottom_sku_order]
        B = piv.loc[dates].values
        Y = reconcile_bottom_up(B, spec.summing_matrix)
        reconciled_arrays[col] = Y

    date_idx = np.repeat(np.arange(n_dates), n_nodes)
    node_idx = np.tile(np.arange(n_nodes), n_dates)

    data_dict: Dict[str, Any] = {}
    if is_rolling:
        date_fold_map = df_sku_preds[["date", "fold"]].drop_duplicates().set_index("date")["fold"].to_dict()
        data_dict["fold"] = [date_fold_map[dates[d]] for d in date_idx]

    data_dict["date"] = [dates[d] for d in date_idx]
    data_dict["hierarchy_level"] = [spec.node_levels[n] for n in node_idx]
    data_dict["node_id"] = [spec.node_ids[n] for n in node_idx]
    data_dict["node_parent_id"] = [spec.node_parent_ids[n] for n in node_idx]

    for col in all_target_cols:
        data_dict[col] = reconciled_arrays[col].ravel()

    df_out = pd.DataFrame(data_dict)
    logger.info(
        "Built hierarchy predictions: %d total rows (%d dates x %d nodes).",
        len(df_out),
        n_dates,
        n_nodes,
    )
    return df_out


def verify_forecast_coherence(
    df_hierarchy: pd.DataFrame,
    model_cols: List[str],
    tolerance: float = 1e-10,
    dataset_name: str = "validation",
    df_sku_source: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Verify exact mathematical coherence of reconciled hierarchy predictions.

    For every date and model, verifies:
    1. Brand == sum(its member SKUs)
    2. Total == sum(4 Brands)
    3. Total == sum(118 SKUs)
    4. SKU forecast preservation (bottom SKU forecasts identical before/after reconciliation)

    Parameters
    ----------
    df_hierarchy : pd.DataFrame
        Complete hierarchy predictions table with columns:
        ['date', 'hierarchy_level', 'node_id', 'node_parent_id', 'actual'] + model_cols.
    model_cols : List[str]
        List of forecast model column names to check.
    tolerance : float, default 1e-10
        Maximum allowed absolute floating-point residual.
    dataset_name : str, default "validation"
        Dataset identifier for reporting.
    df_sku_source : Optional[pd.DataFrame], default None
        Original unreconciled SKU prediction table to verify exact preservation.

    Returns
    -------
    List[Dict[str, Any]]
        List of coherence check records matching hierarchy_coherence_checks schema.
    """
    checks = []

    # Partition by level
    df_sku = df_hierarchy[df_hierarchy["hierarchy_level"] == "sku"]
    df_brand = df_hierarchy[df_hierarchy["hierarchy_level"] == "brand"]
    df_total = df_hierarchy[df_hierarchy["hierarchy_level"] == "total"]

    for model in model_cols:
        # Check 1: Brand equals sum of member SKUs
        sku_by_brand = (
            df_sku.groupby(["date", "node_parent_id"])[model]
            .sum()
            .reset_index()
            .rename(columns={"node_parent_id": "node_id", model: "sku_sum"})
        )
        brand_df = df_brand[["date", "node_id", model]].rename(columns={model: "brand_val"})
        merged_brand = pd.merge(brand_df, sku_by_brand, on=["date", "node_id"], how="inner")
        brand_res = np.abs(merged_brand["brand_val"] - merged_brand["sku_sum"])

        max_brand_res = float(brand_res.max())
        mean_brand_res = float(brand_res.mean())
        viol_brand = int((brand_res > tolerance).sum())

        checks.append(
            {
                "dataset": dataset_name,
                "model": model,
                "hierarchy_level_check": "brand_equals_sum_of_skus",
                "max_abs_residual": round(max_brand_res, 12),
                "mean_abs_residual": round(mean_brand_res, 12),
                "violating_rows": viol_brand,
                "passed": viol_brand == 0,
            }
        )

        # Check 2: Total equals sum of Brands
        brand_sum = (
            df_brand.groupby("date")[model]
            .sum()
            .reset_index()
            .rename(columns={model: "brand_sum"})
        )
        total_val = df_total[["date", model]].rename(columns={model: "total_val"})
        merged_tot_brand = pd.merge(total_val, brand_sum, on="date", how="inner")
        tot_brand_res = np.abs(merged_tot_brand["total_val"] - merged_tot_brand["brand_sum"])

        max_tot_b_res = float(tot_brand_res.max())
        mean_tot_b_res = float(tot_brand_res.mean())
        viol_tot_b = int((tot_brand_res > tolerance).sum())

        checks.append(
            {
                "dataset": dataset_name,
                "model": model,
                "hierarchy_level_check": "total_equals_sum_of_brands",
                "max_abs_residual": round(max_tot_b_res, 12),
                "mean_abs_residual": round(mean_tot_b_res, 12),
                "violating_rows": viol_tot_b,
                "passed": viol_tot_b == 0,
            }
        )

        # Check 3: Total equals sum of all SKUs
        sku_tot_sum = (
            df_sku.groupby("date")[model]
            .sum()
            .reset_index()
            .rename(columns={model: "sku_tot_sum"})
        )
        merged_tot_sku = pd.merge(total_val, sku_tot_sum, on="date", how="inner")
        tot_sku_res = np.abs(merged_tot_sku["total_val"] - merged_tot_sku["sku_tot_sum"])

        max_tot_s_res = float(tot_sku_res.max())
        mean_tot_s_res = float(tot_sku_res.mean())
        viol_tot_s = int((tot_sku_res > tolerance).sum())

        checks.append(
            {
                "dataset": dataset_name,
                "model": model,
                "hierarchy_level_check": "total_equals_sum_of_skus",
                "max_abs_residual": round(max_tot_s_res, 12),
                "mean_abs_residual": round(mean_tot_s_res, 12),
                "violating_rows": viol_tot_s,
                "passed": viol_tot_s == 0,
            }
        )

        # Check 4: SKU forecast preservation (if source table provided)
        if df_sku_source is not None and model in df_sku_source.columns:
            merged_pres = pd.merge(
                df_sku[["date", "node_id", model]],
                df_sku_source[["date", "sku_id", model]],
                left_on=["date", "node_id"],
                right_on=["date", "sku_id"],
                suffixes=("_reconciled", "_source"),
            )
            sku_diff = np.abs(merged_pres[f"{model}_reconciled"] - merged_pres[f"{model}_source"])
            max_sku_diff = float(sku_diff.max())
            mean_sku_diff = float(sku_diff.mean())
            viol_sku = int((sku_diff > tolerance).sum())

            checks.append(
                {
                    "dataset": dataset_name,
                    "model": model,
                    "hierarchy_level_check": "sku_forecast_preservation",
                    "max_abs_residual": round(max_sku_diff, 12),
                    "mean_abs_residual": round(mean_sku_diff, 12),
                    "violating_rows": viol_sku,
                    "passed": viol_sku == 0,
                }
            )

    logger.info(
        "Verified coherence for dataset '%s' across %d models (tolerance=%.1e).",
        dataset_name,
        len(model_cols),
        tolerance,
    )
    return checks

