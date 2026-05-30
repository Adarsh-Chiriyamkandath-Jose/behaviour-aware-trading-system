"""Clean and validate raw OHLCV data before feature engineering."""
from __future__ import annotations

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy OHLCV frame: sorted, de-duplicated, and gap-repaired.

    Trading data has legitimate calendar gaps (weekends, holidays). We only
    repair missing values within the trading days we were given; we never
    fabricate new dates.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # Carry the last known price across the occasional missing day; a missing
    # volume becomes zero (no trading recorded).
    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df[price_cols].ffill()
    df["Volume"] = df["Volume"].fillna(0)

    before = len(df)
    df = df.dropna(subset=price_cols)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with unrecoverable missing prices", dropped)

    # A zero or negative price is bad data, not a real quote.
    df = df[(df[price_cols] > 0).all(axis=1)]
    return df
