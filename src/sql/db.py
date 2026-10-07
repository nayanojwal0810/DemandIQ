"""Database connection and execution utilities for PostgreSQL."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import psycopg

logger = logging.getLogger(__name__)

ENV_DATABASE_URL = "DEMANDIQ_DATABASE_URL"


def get_connection_url() -> str:
    """Retrieve the PostgreSQL connection URL from the environment.

    Raises:
        ValueError: If DEMANDIQ_DATABASE_URL is not set.
    """
    url = os.environ.get(ENV_DATABASE_URL)
    if not url:
        raise ValueError(
            f"Environment variable '{ENV_DATABASE_URL}' is not set. "
            "Please configure DEMANDIQ_DATABASE_URL (e.g. postgresql://user:pass@localhost:5432/demandiq)."
        )
    return url


def get_connection(connection_url: Optional[str] = None) -> psycopg.Connection:
    """Create and return an active PostgreSQL connection.

    Args:
        connection_url: Optional explicit URL; if omitted, retrieved from environment.

    Returns:
        psycopg.Connection instance with autocommit=False.
    """
    url = connection_url or get_connection_url()
    conn = psycopg.connect(url)
    return conn


def execute_sql_file(conn: psycopg.Connection, sql_file_path: Union[str, Path]) -> None:
    """Execute a SQL script from a file path within a transaction.

    Args:
        conn: Active psycopg connection.
        sql_file_path: Path to the .sql script.
    """
    path = Path(sql_file_path)
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")

    sql_content = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql_content)
    conn.commit()
    logger.info("Executed SQL script: %s", path.name)


def load_staging_copy(
    conn: psycopg.Connection,
    csv_file_path: Union[str, Path] = Path("data/processed/sku_demand_daily.csv"),
) -> int:
    """Bulk load canonical daily demand CSV into staging_daily_demand using PostgreSQL COPY.

    Args:
        conn: Active psycopg connection.
        csv_file_path: Path to canonical SKU-day CSV.

    Returns:
        Number of rows staged.
    """
    path = Path(csv_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Canonical CSV file not found: {path}")

    copy_sql = (
        "COPY staging_daily_demand (date, brand_id, sku_id, quantity, promotion) "
        "FROM STDIN WITH (FORMAT csv, HEADER true)"
    )

    with conn.cursor() as cur:
        # Clear staging table before bulk loading
        cur.execute("TRUNCATE TABLE staging_daily_demand;")
        with cur.copy(copy_sql) as copy:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    copy.write(chunk)

        cur.execute("SELECT COUNT(*) FROM staging_daily_demand;")
        count = cur.fetchone()[0]

    conn.commit()
    logger.info("Loaded %d rows into staging_daily_demand via bulk COPY.", count)
    return count
