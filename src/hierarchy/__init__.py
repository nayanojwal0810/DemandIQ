"""Hierarchical forecasting and reconciliation package for DemandIQ."""

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

__all__ = [
    "EXPECTED_BRANDS",
    "EXPECTED_SKU_COUNT",
    "EXPECTED_TOTAL_NODES",
    "TOTAL_NODE_ID",
    "HierarchySpec",
    "load_sku_hierarchy_mapping",
    "build_summing_matrix",
    "reconcile_bottom_up",
    "build_hierarchy_predictions",
    "verify_forecast_coherence",
    "calculate_hierarchy_metrics",
    "calculate_hierarchy_summary",
    "calculate_hierarchy_promotion_comparison",
    "plot_hierarchy_wape_comparison",
    "plot_hierarchy_promotion_delta",
    "run_hierarchy_pipeline",
]


def __getattr__(name: str):
    if name in {
        "calculate_hierarchy_metrics",
        "calculate_hierarchy_summary",
        "calculate_hierarchy_promotion_comparison",
        "plot_hierarchy_wape_comparison",
        "plot_hierarchy_promotion_delta",
        "run_hierarchy_pipeline",
    }:
        from src.hierarchy import evaluate_hierarchy

        return getattr(evaluate_hierarchy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
