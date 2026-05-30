"""Backtesting engine: turn predictions into an equity curve."""
from __future__ import annotations

import pandas as pd


def run_backtest(prices: pd.Series, signals: pd.Series, initial_capital: float = 100_000,
                 transaction_cost: float = 0.0005) -> pd.DataFrame:
    """Simulate trades from signals and return positions + equity curve.

    TODO: implement (apply costs, track cash/position, build equity curve).
    """
    raise NotImplementedError
