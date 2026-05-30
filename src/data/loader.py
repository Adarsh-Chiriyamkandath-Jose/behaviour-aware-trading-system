"""Download historical OHLCV data (Yahoo Finance via yfinance)."""
from __future__ import annotations

import pandas as pd


def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return OHLCV DataFrame indexed by date for a single ticker.

    TODO: implement with yfinance; cache to data/raw/.
    """
    raise NotImplementedError
