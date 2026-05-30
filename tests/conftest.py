"""Shared test fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """A small, deterministic OHLCV frame for fast unit tests."""
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # A gently trending random walk so prices stay positive and realistic.
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    close = np.maximum(close, 1.0)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + rng.uniform(0, 1, n),
            "Low": close - rng.uniform(0, 1, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.Index(dates, name="Date"),
    )
    return df
