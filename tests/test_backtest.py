"""Tests for the backtesting engine and signal strategies."""
import numpy as np
import pandas as pd

from src.backtest import (
    buy_and_hold,
    run_backtest,
    signal_from_predictions,
    signal_from_proba,
)


def _prices(n=50, start=100.0, step=1.0):
    return pd.Series(start + np.arange(n) * step, index=pd.date_range("2021-01-01", periods=n, freq="B"))


def test_signal_from_predictions_long_flat():
    sig = signal_from_predictions([1, 0, 1], allow_short=False)
    assert list(sig) == [1.0, 0.0, 1.0]


def test_signal_from_predictions_can_short():
    sig = signal_from_predictions([1, 0], allow_short=True)
    assert list(sig) == [1.0, -1.0]


def test_signal_from_proba_neutral_band():
    proba = [0.9, 0.5, 0.1]
    sig = signal_from_proba(proba, upper=0.55, lower=0.45, allow_short=True)
    assert list(sig) == [1.0, 0.0, -1.0]  # confident up / uncertain / confident down


def test_backtest_no_lookahead_first_day_flat():
    """Day-one position is flat: the signal acts only on the *next* day's move."""
    prices = _prices()
    signals = pd.Series(1.0, index=prices.index)  # always-long signal
    bt = run_backtest(prices, signals, transaction_cost=0.0)
    assert bt["position"].iloc[0] == 0.0
    assert bt["strategy_return"].iloc[0] == 0.0


def test_backtest_long_matches_asset_on_rising_prices():
    prices = _prices()
    signals = pd.Series(1.0, index=prices.index)
    bt = run_backtest(prices, signals, transaction_cost=0.0)
    # With no costs and a permanent long, strategy return == asset return (shifted in).
    expected = prices.pct_change().fillna(0.0).shift(0)
    np.testing.assert_allclose(bt["strategy_return"].iloc[1:], (prices.pct_change().fillna(0.0)).iloc[1:])
    assert bt["equity"].iloc[-1] > bt["equity"].iloc[0]


def test_backtest_costs_reduce_equity():
    prices = _prices()
    # Flip-flop every day to force turnover.
    signals = pd.Series([1.0, 0.0] * (len(prices) // 2), index=prices.index)
    free = run_backtest(prices, signals, transaction_cost=0.0)["equity"].iloc[-1]
    costed = run_backtest(prices, signals, transaction_cost=0.01)["equity"].iloc[-1]
    assert costed < free


def test_backtest_flat_signal_preserves_capital():
    prices = _prices()
    signals = pd.Series(0.0, index=prices.index)
    bt = run_backtest(prices, signals, initial_capital=100_000, transaction_cost=0.001)
    assert np.isclose(bt["equity"].iloc[-1], 100_000)


def test_buy_and_hold_tracks_price():
    prices = _prices()
    bh = buy_and_hold(prices, initial_capital=100_000)
    assert np.isclose(bh.iloc[0], 100_000)
    assert np.isclose(bh.iloc[-1] / bh.iloc[0], prices.iloc[-1] / prices.iloc[0])
