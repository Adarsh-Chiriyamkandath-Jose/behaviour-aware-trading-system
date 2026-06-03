"""Quantitative feature engineering.

`build_features` is the convenience entry point: it applies every technical and
rolling indicator described in config.yaml in one call.
"""
from src.features.technical import (
    add_returns,
    add_moving_averages,
    add_rsi,
    add_momentum,
    add_volatility,
)
from src.features.rolling_stats import add_rolling_stats


def build_features(df, config):
    cfg = config["features"]
    df = add_returns(df)
    df = add_moving_averages(df, cfg["sma_windows"], cfg["ema_windows"])
    df = add_rsi(df, cfg["rsi_period"])
    df = add_momentum(df, cfg["momentum_period"])
    df = add_volatility(df, cfg["volatility_window"])
    df = add_rolling_stats(df, cfg["rolling_windows"])
    return df


__all__ = [
    "build_features",
    "add_returns",
    "add_moving_averages",
    "add_rsi",
    "add_momentum",
    "add_volatility",
    "add_rolling_stats",
]
