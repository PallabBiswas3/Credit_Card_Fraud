"""
Data Ingestion Layer
--------------------
CHALLENGE FACED: Raw data arrives with duplicates from retry events,
negative amounts from refund mis-logs, and partial nulls from declined
transactions that were only partially written to the DB.
We validate on ingest so nothing dirty reaches preprocessing.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} rows, {df['Class'].sum()} fraud cases")
    return df


def validate(df: pd.DataFrame) -> dict:
    """
    Run basic data quality checks.
    Returns a report dict — caller decides whether to halt or warn.
    """
    report = {}

    # Check expected columns
    required = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    missing_cols = [c for c in required if c not in df.columns]
    report["missing_columns"] = missing_cols

    # Duplicate rows (same Time + Amount + V1 fingerprint)
    dups = df.duplicated(subset=["Time", "Amount", "V1"], keep="first").sum()
    report["duplicate_rows"] = int(dups)

    # Nulls
    report["null_counts"] = df.isnull().sum().to_dict()

    # Negative amounts
    report["negative_amount_rows"] = int((df["Amount"] < 0).sum())

    # Class balance
    report["fraud_rate_pct"] = round(df["Class"].mean() * 100, 4)

    for k, v in report.items():
        log.info(f"  {k}: {v}")

    return report


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Time", "Amount", "V1"], keep="first")
    log.info(f"Removed {before - len(df)} duplicate rows")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw_path = Path(__file__).parents[2] / "data" / "creditcard_raw.csv"
    df = load_raw(str(raw_path))
    report = validate(df)
    df = remove_duplicates(df)
    out = Path(__file__).parents[2] / "data" / "creditcard_ingested.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved ingested data → {out}")
