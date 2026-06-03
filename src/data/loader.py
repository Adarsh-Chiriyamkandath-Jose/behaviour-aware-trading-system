"""Download historical OHLCV data from Yahoo Finance and cache it locally.

Raw downloads are saved as CSV under data/raw/ so repeated runs don't hit the
network and the whole team works off the same snapshot.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    raw_dir: str | Path = "data/raw",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for one ticker, indexed by date.

    Pass refresh=True to bypass the cache and re-download.
    """
    cache_path = Path(raw_dir) / f"{ticker}.csv"

    if cache_path.exists() and not refresh:
        logger.info("Loading %s from cache: %s", ticker, cache_path)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    logger.info("Downloading %s from Yahoo Finance (%s to %s)", ticker, start, end)
    # auto_adjust folds dividends and splits into the prices, which is what we
    # want so that returns are continuous across those events.
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol and dates.")

    # yfinance can hand back a (field, ticker) MultiIndex even for one symbol;
    # flatten it to plain field names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[OHLCV_COLUMNS]
    df.index.name = "Date"

    ensure_dir(raw_dir)
    df.to_csv(cache_path)
    return df


def load_many(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: str | Path = "data/raw",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load several tickers, returning {ticker: DataFrame}."""
    return {t: load_ohlcv(t, start, end, raw_dir, refresh) for t in tickers}
