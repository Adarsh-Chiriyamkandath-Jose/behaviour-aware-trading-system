"""Quantitative technical indicators.

Every indicator here is computed from past-and-present prices only, so the
resulting columns are safe to feed a model without leaking the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()
    return df


def add_moving_averages(
    df: pd.DataFrame,
    sma_windows: list[int],
    ema_windows: list[int],
) -> pd.DataFrame:
    df = df.copy()
    for w in sma_windows:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
        # How far price sits from its average is more informative to a model
        # than the raw average level.
        df[f"close_to_sma_{w}"] = df["Close"] / df[f"sma_{w}"] - 1.0
    for w in ema_windows:
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index using Wilder's smoothing."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[f"rsi_{period}"] = 100 - 100 / (1 + rs)
    return df


def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df[f"momentum_{period}"] = df["Close"].pct_change(period)
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    if "return" not in df.columns:
        df["return"] = df["Close"].pct_change()
    df[f"volatility_{window}"] = df["return"].rolling(window).std()
    return df
