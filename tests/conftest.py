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


@pytest.fixture
def large_ohlcv():
    """A longer OHLCV frame (enough rows to survive warmup + walk-forward folds).

    Built with shifting drift and volatility so the direction label carries both
    classes and the behavioral indicators actually vary.
    """
    rng = np.random.default_rng(7)
    n = 900
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    # Alternating calm-up / volatile-down regimes give the model something to learn.
    drift = np.where((np.arange(n) // 120) % 2 == 0, 0.06, -0.04)
    vol = np.where((np.arange(n) // 120) % 2 == 0, 0.8, 1.8)
    close = 100 + np.cumsum(rng.normal(drift, vol))
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


@pytest.fixture
def sample_config():
    """A small but complete config for fast end-to-end experiment tests."""
    return {
        "data": {"tickers": ["TEST"], "start_date": "2018-01-01", "end_date": "2021-01-01", "raw_dir": "data/raw"},
        "labels": {"horizon": 1, "threshold": 0.0},
        "features": {
            "sma_windows": [10, 20],
            "ema_windows": [12],
            "rsi_period": 14,
            "momentum_period": 10,
            "volatility_window": 20,
            "rolling_windows": [5, 20],
        },
        "behavioral": {
            "fear": {"volatility_window": 20},
            "greed": {"momentum_window": 10, "volume_window": 20},
            "herd": {"volume_spike_z": 2.0, "price_move_z": 2.0},
        },
        "split": {"scheme": "walk_forward", "n_splits": 3, "test_size": 0.2, "gap": 0},
        "models": {
            "active": ["logistic", "random_forest"],
            "random_state": 42,
            "logistic": {"C": 1.0, "max_iter": 1000},
            "random_forest": {"n_estimators": 50, "max_depth": 5},
        },
        "backtest": {"initial_capital": 100_000, "transaction_cost": 0.0005, "allow_short": False},
        "seed": 42,
    }
