"""Ablation study — THE core research question.

Compare model performance on `quant_only` vs `quant_plus_behavioral` feature
sets, across all models, using time-aware splits. Report both classification
and financial metrics, with statistical significance.
"""
from __future__ import annotations

from typing import Any


def run_ablation(config: dict[str, Any]) -> "Any":
    """Run the with/without-behavioral comparison and return a results table.

    TODO: implement once features/models/backtest are ready.
    """
    raise NotImplementedError
