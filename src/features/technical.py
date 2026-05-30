"""Quantitative technical indicators."""
from __future__ import annotations

import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame: raise NotImplementedError
def add_moving_averages(df: pd.DataFrame, sma_windows, ema_windows) -> pd.DataFrame: raise NotImplementedError
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame: raise NotImplementedError
def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame: raise NotImplementedError
def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame: raise NotImplementedError
