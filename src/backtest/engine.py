"""Backtesting engine: turn position signals into an equity curve.

This is the *realistic* counterpart to the quick long/flat proxy in
`src/evaluation/financial.py`. It tracks the held position day by day, charges a
transaction cost whenever that position changes, and compounds the result into
an equity curve — all without look-ahead: the position you choose at the close of
day *t* only earns the market's move on day *t+1*.
"""
from __future__ import annotations

import pandas as pd


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    """Simulate trading `signals` against `prices` and return a per-day ledger.

    Parameters
    ----------
    prices : the asset price series (e.g. adjusted Close), indexed by date.
    signals : target position per day in [-1, 1] (see `src.backtest.strategies`),
        aligned to `prices`. Interpreted as the position *decided* that day.
    transaction_cost : fraction charged on turnover (|Δposition|) each time the
        position changes.

    Returns a DataFrame indexed like `prices` with columns:
    price, asset_return, signal, position, turnover, cost, strategy_return, equity,
    drawdown — so downstream plots and metrics can read whatever they need.
    """
    prices = prices.astype(float)
    signals = signals.reindex(prices.index).ffill().fillna(0.0)

    asset_return = prices.pct_change().fillna(0.0)
    # The position actually *held* today was chosen at yesterday's close, so it
    # earns today's return. Shifting is what keeps the backtest honest.
    position = signals.shift(1).fillna(0.0)

    # Turnover is how much the held position moved (opening from flat counts).
    turnover = position.diff().abs().fillna(position.abs())
    cost = transaction_cost * turnover

    strategy_return = position * asset_return - cost
    equity = (1.0 + strategy_return).cumprod() * initial_capital
    drawdown = equity / equity.cummax() - 1.0

    return pd.DataFrame(
        {
            "price": prices,
            "asset_return": asset_return,
            "signal": signals,
            "position": position,
            "turnover": turnover,
            "cost": cost,
            "strategy_return": strategy_return,
            "equity": equity,
            "drawdown": drawdown,
        }
    )


def buy_and_hold(prices: pd.Series, initial_capital: float = 100_000) -> pd.Series:
    """Benchmark equity curve for being fully invested the whole time."""
    prices = prices.astype(float)
    equity = prices / prices.iloc[0] * initial_capital
    equity.name = "buy_and_hold"
    return equity
