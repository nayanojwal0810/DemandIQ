"""Business and demand analytics package for DemandIQ."""

from src.business.promotion_sparse_analysis import (
    SPARSITY_BUCKETS,
    assign_sparsity_bucket,
    compute_promotion_condition_metrics,
    compute_promotion_model_comparison,
    compute_sparse_demand_metrics,
    run_promotion_sparse_analysis_pipeline,
)

__all__ = [
    "SPARSITY_BUCKETS",
    "assign_sparsity_bucket",
    "compute_promotion_condition_metrics",
    "compute_promotion_model_comparison",
    "compute_sparse_demand_metrics",
    "run_promotion_sparse_analysis_pipeline",
]
