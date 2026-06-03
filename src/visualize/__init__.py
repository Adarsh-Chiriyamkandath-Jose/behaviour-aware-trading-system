"""Visualization package: report-ready figures and SHAP interpretability."""
from src.visualize.plots import (
    plot_ablation_comparison,
    plot_confusion,
    plot_drawdown,
    plot_equity_curve,
    plot_feature_importance,
)
from src.visualize.shap_analysis import (
    behavioral_shap_ranking,
    compute_shap_values,
    shap_summary,
)

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_ablation_comparison",
    "plot_confusion",
    "plot_feature_importance",
    "compute_shap_values",
    "behavioral_shap_ranking",
    "shap_summary",
]
