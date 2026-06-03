"""SHAP-based interpretability (STRETCH GOAL).

Explain which features drive predictions — and crucially, whether the behavioral
proxies (fear / greed / herd) carry real weight or are just along for the ride.
This is the "insights into market psychology" deliverable.

`shap` is an optional dependency; importing this module is cheap, but calling its
functions without `shap` installed raises a clear, actionable error rather than a
bare ImportError.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir

BEHAVIORAL = ("fear", "greed", "herd")


def _require_shap():
    try:
        import shap  # noqa: F401
        return shap
    except ImportError as exc:  # pragma: no cover - exercised only without shap
        raise ImportError(
            "SHAP analysis needs the 'shap' package. Install it with "
            "`pip install shap` (it's already in requirements.txt)."
        ) from exc


def _sample(X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return X if len(X) <= n else X.sample(n, random_state=seed)


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 200, seed: int = 42):
    """Return ``(shap_values, X_sample)`` for the positive class.

    Tree models (random forest, xgboost) use the exact, fast TreeExplainer; other
    models fall back to a model-agnostic explainer over a background sample. The
    returned array has shape (n_sample, n_features), aligned to `X_sample`.
    """
    shap = _require_shap()
    Xs = _sample(X, max_samples, seed)

    if getattr(model, "name", "") in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model.model)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):  # older shap: [class0, class1]
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # newer shap: (n, features, classes)
            sv = sv[:, :, 1]
    else:
        background = _sample(X, min(100, len(X)), seed)
        explainer = shap.Explainer(model.predict_proba, shap.maskers.Independent(background))
        sv = np.asarray(explainer(Xs).values)

    return sv, Xs


def behavioral_shap_ranking(model, X: pd.DataFrame, max_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted, flagged for behavioral features.

    The quick answer to "did fear/greed/herd matter?": their rows near the top
    mean yes. Returns columns ``mean_abs_shap`` and ``is_behavioral``.
    """
    sv, Xs = compute_shap_values(model, X, max_samples, seed)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
    return pd.DataFrame(
        {
            "mean_abs_shap": importance,
            "is_behavioral": [name in BEHAVIORAL for name in importance.index],
        }
    )


def shap_summary(model, X: pd.DataFrame, save_path: str | None = None, max_samples: int = 200, seed: int = 42):
    """Beeswarm summary plot of SHAP values; returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    shap = _require_shap()
    sv, Xs = compute_shap_values(model, X, max_samples, seed)

    shap.summary_plot(sv, Xs, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP feature impact (behavioral features included)", y=1.02)
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig
