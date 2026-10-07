"""Dataset validation script for DemandIQ.

Loads canonical and raw datasets from disk and executes the full data contract
validation suite with standard logging.
"""

from pathlib import Path
import logging
import sys

import pandas as pd

from src.validation.data_contract import validate_canonical_dataset

logger = logging.getLogger(__name__)


def run_dataset_validation(
    canonical_path: Path = Path("data/processed/sku_demand_daily.csv"),
    raw_path: Path = Path("data/raw/hierarchical_sales_data.csv"),
) -> dict:
    """Load canonical and raw data and run complete contract validation.

    Parameters
    ----------
    canonical_path : Path
        Path to processed canonical SKU demand CSV.
    raw_path : Path
        Path to raw wide source CSV.

    Returns
    -------
    dict
        Validation results dictionary.
    """
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical dataset not found at: {canonical_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    logger.info("Loading canonical dataset from %s", canonical_path)
    canonical_df = pd.read_csv(canonical_path)

    logger.info("Loading raw dataset from %s", raw_path)
    source_df = pd.read_csv(raw_path)

    logger.info("Executing comprehensive data contract checks...")
    results = validate_canonical_dataset(canonical_df, source_df)
    logger.info("All data contract checks PASSED successfully.")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        run_dataset_validation()
    except Exception as exc:
        logger.error("Data contract validation FAILED: %s", exc)
        sys.exit(1)
