"""Plotting helpers: equity curves, the ablation comparison, confusion matrices,
and feature importances.

Every function returns the matplotlib ``Figure`` (and optionally saves it), so
notebooks can display it and `main.py` can write it to `results/figures/`. We
deliberately keep behavioral features (fear/greed/herd) visually distinct in the
charts where they appear — they're the point of the whole project.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.utils import ensure_dir

BEHAVIORAL = ("fear", "greed", "herd")


def _save(fig, save_path: str | None) -> None:
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)


def plot_equity_curve(equity: pd.Series, benchmark: pd.Series | None = None, save_path: str | None = None):
    """Strategy equity over time, optionally against a buy-and-hold benchmark."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, label="Strategy", lw=1.6)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label="Buy & hold", lw=1.2, ls="--", c="grey")
    ax.set_title("Equity curve")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_drawdown(equity: pd.Series, save_path: str | None = None):
    """Underwater (drawdown) plot — how far below the prior peak we ever sat."""
    drawdown = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_ablation_comparison(summary: pd.DataFrame, metric: str = "accuracy", save_path: str | None = None):
    """Grouped bars: quant-only vs quant+behavioral for `metric`, per model.

    `summary` is the ablation summary table (columns include model, feature_set,
    and the metric). Tickers are averaged so the story reads at a glance.
    """
    agg = summary.groupby(["model", "feature_set"])[metric].mean().unstack("feature_set")
    # Stable, meaningful column order if both feature sets are present.
    cols = [c for c in ["quant_only", "quant_plus_behavioral"] if c in agg.columns]
    agg = agg[cols] if cols else agg

    fig, ax = plt.subplots(figsize=(8, 5))
    agg.plot.bar(ax=ax, color={"quant_only": "#9aa5b1", "quant_plus_behavioral": "#2f9e6f"})
    if metric == "accuracy":
        ax.axhline(0.5, ls="--", c="grey", lw=1, label="coin flip")
    ax.set_title(f"Ablation: {metric} by model")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion(y_true, y_pred, save_path: str | None = None):
    """Confusion matrix for the up/down classifier."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], labels=["down", "up"])
    ax.set_yticks([0, 1], labels=["down", "up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_feature_importance(importances: pd.Series, top_n: int = 15, save_path: str | None = None):
    """Horizontal bar chart of the top features, behavioral ones highlighted."""
    top = importances.head(top_n)[::-1]
    colors = ["crimson" if name in BEHAVIORAL else "#4a6fa5" for name in top.index]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(top))))
    ax.barh(top.index, top.values, color=colors)
    ax.set_title(f"Top {len(top)} feature importances (behavioral in red)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
