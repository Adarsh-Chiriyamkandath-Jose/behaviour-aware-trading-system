"""Financial metrics: cumulative return, P&L, Sharpe, max drawdown.

This module answers "would acting on the predictions have made money?" It also
ships a *minimal* long/flat strategy (`strategy_returns`) so the ablation can
report financial performance before Member C's full backtest engine lands. The
two should agree to first order; `src/backtest/` is the realistic version with
richer position logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def strategy_returns(
    predictions,
    forward_returns: pd.Series,
    transaction_cost: float = 0.0005,
    allow_short: bool = False,
) -> pd.Series:
    """Per-period net returns of acting on `predictions`.

    Position is taken from the model's call for the period whose realised result
    is `forward_returns` (1 = up -> go long; 0 = down -> flat, or short if
    `allow_short`). Costs are charged on every change of position, so flip-flopping
    is penalised the way real trading punishes it.

    `forward_returns` must already be the return earned *after* the prediction is
    made (e.g. Close.shift(-h)/Close - 1), so there is no look-ahead here.
    """
    preds = pd.Series(np.asarray(predictions), index=forward_returns.index).astype(float)
    # 1 -> long. A "down" call means hold cash (flat), or short if allowed.
    position = preds.where(preds > 0, -1.0 if allow_short else 0.0)

    gross = position * forward_returns
    turnover = position.diff().abs().fillna(position.abs())  # opening cost on day one
    net = gross - transaction_cost * turnover
    net.name = "strategy_return"
    return net


def equity_curve(returns: pd.Series, initial_capital: float = 100_000.0) -> pd.Series:
    """Compound a series of per-period returns into an equity curve."""
    curve = (1.0 + returns.fillna(0.0)).cumprod() * initial_capital
    curve.name = "equity"
    return curve


def financial_metrics(
    equity_curve: pd.Series,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict:
    """Performance summary derived from an equity curve.

    Returns total/annualised return, annualised volatility, Sharpe ratio,
    maximum drawdown and per-period win rate. Sharpe uses the per-period excess
    return scaled by sqrt(periods_per_year); a flat curve yields Sharpe 0 rather
    than a divide-by-zero NaN.
    """
    equity = equity_curve.dropna()
    if len(equity) < 2:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }

    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    n_periods = len(rets)

    annual_return = (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0
    annual_vol = rets.std(ddof=1) * np.sqrt(periods_per_year)

    excess = rets - risk_free / periods_per_year
    sharpe = 0.0 if rets.std(ddof=1) == 0 else excess.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year)

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": (rets > 0).mean(),
    }
