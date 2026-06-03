"""Backtest package: map predictions -> signals -> a realistic equity curve."""
from src.backtest.engine import buy_and_hold, run_backtest
from src.backtest.strategies import signal_from_predictions, signal_from_proba

__all__ = [
    "run_backtest",
    "buy_and_hold",
    "signal_from_predictions",
    "signal_from_proba",
]
