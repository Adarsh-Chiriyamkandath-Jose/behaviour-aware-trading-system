"""Tests for behavioral indicators."""
import numpy as np
import pandas as pd

from src.behavioral import compute_fear, compute_greed, compute_herd


def test_fear_higher_in_a_crash_than_a_calm_market():
    calm = pd.DataFrame({"Close": np.linspace(100, 110, 100)})
    crash = calm.copy()
    crash.loc[80:, "Close"] = np.linspace(110, 60, 20)  # sharp sustained drop
    fear_calm = compute_fear(calm).iloc[-1]
    fear_crash = compute_fear(crash).iloc[-1]
    assert fear_crash > fear_calm


def test_greed_is_non_negative():
    df = pd.DataFrame({
        "Close": np.linspace(100, 130, 100),
        "Volume": np.random.default_rng(0).integers(1e6, 5e6, 100).astype(float),
    })
    greed = compute_greed(df).dropna()
    assert (greed >= 0).all()


def test_herd_is_zero_without_extremes(sample_ohlcv):
    # A smooth random walk shouldn't trip both volume- and price-extreme gates.
    herd = compute_herd(sample_ohlcv).dropna()
    assert (herd == 0).mean() > 0.5
