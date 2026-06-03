"""Tests for plotting helpers and SHAP analysis.

We force the non-interactive Agg backend so figures render headless in CI.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.visualize import (
    plot_ablation_comparison,
    plot_confusion,
    plot_equity_curve,
    plot_feature_importance,
)


def _equity(n=30):
    return pd.Series(100_000 * (1 + np.linspace(0, 0.2, n)), index=pd.date_range("2021-01-01", periods=n, freq="B"))


def test_plot_equity_curve_returns_figure():
    fig = plot_equity_curve(_equity(), benchmark=_equity() * 0.9)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_equity_curve_saves_file(tmp_path):
    out = tmp_path / "figs" / "equity.png"
    plot_equity_curve(_equity(), save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_ablation_comparison_orders_feature_sets():
    summary = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "model": ["logistic", "logistic", "rf", "rf"],
            "feature_set": ["quant_only", "quant_plus_behavioral"] * 2,
            "accuracy": [0.51, 0.55, 0.52, 0.57],
        }
    )
    fig = plot_ablation_comparison(summary, metric="accuracy")
    assert isinstance(fig, Figure)


def test_plot_confusion_returns_figure():
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 1, 1, 0, 0])
    fig = plot_confusion(y_true, y_pred)
    assert isinstance(fig, Figure)


def test_plot_feature_importance_highlights_behavioral():
    imp = pd.Series({"rsi_14": 0.3, "fear": 0.25, "momentum_10": 0.2, "greed": 0.1})
    fig = plot_feature_importance(imp, top_n=4)
    assert isinstance(fig, Figure)


def test_behavioral_shap_ranking_covers_all_features():
    shap = pytest.importorskip("shap")
    from src.models import RandomForestModel
    from src.visualize import behavioral_shap_ranking

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120), "fear": rng.normal(size=120), "greed": rng.normal(size=120)})
    y = (X["a"] + X["fear"] > 0).astype(int)
    model = RandomForestModel(n_estimators=30, max_depth=4).fit(X, y)

    ranking = behavioral_shap_ranking(model, X, max_samples=60)
    assert set(ranking.index) == set(X.columns)
    assert ranking.loc["fear", "is_behavioral"]
    assert not ranking.loc["a", "is_behavioral"]
