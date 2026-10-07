"""Ingestion module for raw UCI Hierarchical Sales Data.

Transforms the wide-format source CSV into a canonical long-format daily demand
table and a minimal SKU dimension table.
"""

from pathlib import Path
import logging
import re
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def identify_source_columns(columns: list[str]) -> Tuple[str, list[str], list[str]]:
    """Identify date, quantity, and promotion columns in source schema.

    Parameters
    ----------
    columns : list[str]
        List of column names from the raw dataset.

    Returns
    -------
    Tuple[str, list[str], list[str]]
        Tuple containing (date_col, qty_cols, promo_cols).

    Raises
    ------
    ValueError
        If date column is missing or quantity/promo columns mismatch.
    """
    date_cols = [c for c in columns if c.strip().upper() == "DATE"]
    if not date_cols:
        raise ValueError("Source dataset does not contain required 'DATE' column.")
    date_col = date_cols[0]

    qty_cols = [c for c in columns if c.startswith("QTY_")]
    promo_cols = [c for c in columns if c.startswith("PROMO_")]

    if not qty_cols:
        raise ValueError("No quantity columns (prefixed with 'QTY_') found in source.")
    if not promo_cols:
        raise ValueError("No promotion columns (prefixed with 'PROMO_') found in source.")

    # Validate 1-to-1 matching between quantity and promotion series
    qty_suffixes = {c.replace("QTY_", "") for c in qty_cols}
    promo_suffixes = {c.replace("PROMO_", "") for c in promo_cols}

    if qty_suffixes != promo_suffixes:
        missing_promos = qty_suffixes - promo_suffixes
        missing_qtys = promo_suffixes - qty_suffixes
        raise ValueError(
            f"Mismatch between QTY and PROMO series. "
            f"Missing PROMO: {sorted(missing_promos)}, Missing QTY: {sorted(missing_qtys)}"
        )

    return date_col, sorted(qty_cols), sorted(promo_cols)


def parse_sku_identity(sku_col_name: str) -> Tuple[str, str]:
    """Parse brand_id and sku_id from a source column name.

    Example: 'QTY_B1_1' -> ('B1', 'B1_1')

    Parameters
    ----------
    sku_col_name : str
        Source column name (e.g., 'QTY_B1_1' or 'PROMO_B1_1').

    Returns
    -------
    Tuple[str, str]
        (brand_id, sku_id)
    """
    cleaned = re.sub(r"^(QTY_|PROMO_)", "", sku_col_name)
    parts = cleaned.split("_")
    brand_id = parts[0]
    sku_id = cleaned
    return brand_id, sku_id


def transform_wide_to_canonical(
    source_df: pd.DataFrame,
    date_col: str,
    qty_cols: list[str],
    promo_cols: list[str],
) -> pd.DataFrame:
    """Transform wide sales data into canonical long format.

    Parameters
    ----------
    source_df : pd.DataFrame
        Raw dataset in wide format.
    date_col : str
        Name of the date column.
    qty_cols : list[str]
        List of quantity column names.
    promo_cols : list[str]
        List of promotion column names.

    Returns
    -------
    pd.DataFrame
        Canonical DataFrame with columns: [date, brand_id, sku_id, quantity, promotion],
        deterministically sorted by [date, brand_id, sku_id].
    """
    # 1. Melt quantity columns
    qty_melted = pd.melt(
        source_df,
        id_vars=[date_col],
        value_vars=qty_cols,
        var_name="qty_col",
        value_name="quantity",
    )
    qty_melted["sku_id"] = qty_melted["qty_col"].str.replace("QTY_", "", regex=False)
    qty_melted["brand_id"] = qty_melted["sku_id"].apply(lambda s: s.split("_")[0])

    # 2. Melt promotion columns
    promo_melted = pd.melt(
        source_df,
        id_vars=[date_col],
        value_vars=promo_cols,
        var_name="promo_col",
        value_name="promotion",
    )
    promo_melted["sku_id"] = promo_melted["promo_col"].str.replace("PROMO_", "", regex=False)

    # 3. Merge quantity and promotion tables on date and sku_id
    canonical_df = pd.merge(
        qty_melted[[date_col, "brand_id", "sku_id", "quantity"]],
        promo_melted[[date_col, "sku_id", "promotion"]],
        on=[date_col, "sku_id"],
        how="inner",
    )

    canonical_df.rename(columns={date_col: "date"}, inplace=True)

    # 4. Standardize types
    canonical_df["date"] = canonical_df["date"].astype(str)
    canonical_df["brand_id"] = canonical_df["brand_id"].astype(str)
    canonical_df["sku_id"] = canonical_df["sku_id"].astype(str)
    canonical_df["quantity"] = canonical_df["quantity"].astype(int)
    canonical_df["promotion"] = canonical_df["promotion"].astype(int)

    # 5. Deterministic column order and sorting
    target_columns = ["date", "brand_id", "sku_id", "quantity", "promotion"]
    canonical_df = canonical_df[target_columns]
    canonical_df.sort_values(
        by=["date", "brand_id", "sku_id"],
        ascending=[True, True, True],
        inplace=True,
    )
    canonical_df.reset_index(drop=True, inplace=True)

    return canonical_df


def extract_sku_dimension(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Extract minimal unique SKU dimension table from canonical data.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical SKU demand DataFrame.

    Returns
    -------
    pd.DataFrame
        SKU dimension DataFrame with columns [brand_id, sku_id],
        sorted deterministically by [brand_id, sku_id].
    """
    dim_df = (
        canonical_df[["brand_id", "sku_id"]]
        .drop_duplicates()
        .sort_values(by=["brand_id", "sku_id"])
        .reset_index(drop=True)
    )
    return dim_df


def ingest_sales_data(
    raw_csv_path: Path = Path("data/raw/hierarchical_sales_data.csv"),
    processed_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute raw data ingestion and save canonical artifacts.

    Parameters
    ----------
    raw_csv_path : Path
        Path to raw CSV file.
    processed_dir : Path
        Directory where canonical artifacts will be stored.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (canonical_df, sku_dimension_df)
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw source file not found at: {raw_csv_path}")

    logger.info("Loading source dataset from %s", raw_csv_path)
    source_df = pd.read_csv(raw_csv_path)
    logger.info("Source dataset loaded. Shape: %s", source_df.shape)

    date_col, qty_cols, promo_cols = identify_source_columns(list(source_df.columns))
    logger.info(
        "Identified 1 date column, %d quantity columns, %d promotion columns.",
        len(qty_cols),
        len(promo_cols),
    )

    canonical_df = transform_wide_to_canonical(source_df, date_col, qty_cols, promo_cols)
    logger.info("Canonical SKU-day dataset created. Rows: %d", len(canonical_df))

    sku_dim_df = extract_sku_dimension(canonical_df)
    logger.info("SKU dimension extracted. Total unique SKUs: %d", len(sku_dim_df))

    processed_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = processed_dir / "sku_demand_daily.csv"
    sku_dim_path = processed_dir / "sku_dimension.csv"

    canonical_df.to_csv(canonical_path, index=False)
    sku_dim_df.to_csv(sku_dim_path, index=False)
    logger.info("Canonical datasets saved to %s and %s", canonical_path, sku_dim_path)

    return canonical_df, sku_dim_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ingest_sales_data()
