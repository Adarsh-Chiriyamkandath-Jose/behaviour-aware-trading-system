"""Statistical significance testing for the ablation comparison.

E.g. paired test across walk-forward folds to check whether behavioral
features give a statistically significant improvement.
"""
from __future__ import annotations


def paired_significance(scores_a, scores_b) -> dict:
    """Compare two models' per-fold scores. TODO: implement."""
    raise NotImplementedError
