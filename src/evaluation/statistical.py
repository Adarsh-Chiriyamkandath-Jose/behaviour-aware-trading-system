"""Statistical significance testing for the ablation comparison.

The ablation gives us a per-fold score for the quant-only model and a per-fold
score for the quant+behavioral model. The question this module answers: is the
gap real, or just noise across folds? We run a paired test (each fold is one
matched pair) so we're comparing like with like.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_significance(scores_a, scores_b) -> dict:
    """Compare two models' per-fold scores with a paired test.

    `scores_b` is the candidate (quant+behavioral), `scores_a` the baseline
    (quant-only); a positive `mean_diff` means behavioral features helped. We
    report a paired t-test (parametric) alongside the Wilcoxon signed-rank test
    (distribution-free) because with only a handful of folds neither alone is
    conclusive — agreement between them is the reassuring signal.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Score arrays must align fold-for-fold: {a.shape} vs {b.shape}")

    diff = b - a
    n = len(diff)
    mean_diff = float(diff.mean())

    result = {
        "n_folds": n,
        "mean_diff": mean_diff,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "t_stat": np.nan,
        "t_pvalue": np.nan,
        "wilcoxon_pvalue": np.nan,
    }

    # Both tests need at least two folds and some variation in the differences.
    if n >= 2 and np.any(diff != 0):
        t_stat, t_p = stats.ttest_rel(b, a)
        result["t_stat"] = float(t_stat)
        result["t_pvalue"] = float(t_p)
        try:
            _, w_p = stats.wilcoxon(b, a)
            result["wilcoxon_pvalue"] = float(w_p)
        except ValueError:
            # Wilcoxon rejects all-zero differences / too-few samples.
            pass

    return result
