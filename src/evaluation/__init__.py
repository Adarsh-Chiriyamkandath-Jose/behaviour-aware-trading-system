"""Evaluation package: classification quality, financial performance, and the
statistical test that decides whether the ablation's gap is real.
"""
from src.evaluation.classification import classification_metrics
from src.evaluation.financial import (
    equity_curve,
    financial_metrics,
    strategy_returns,
)
from src.evaluation.statistical import paired_significance

__all__ = [
    "classification_metrics",
    "strategy_returns",
    "equity_curve",
    "financial_metrics",
    "paired_significance",
]
