"""Classification metrics: accuracy, precision, recall, F1, ROC-AUC.

These answer "is the model right about direction?" — separate from whether
acting on it makes money (that's financial.py).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Return accuracy / precision / recall / F1 (and ROC-AUC if probabilities
    are supplied), all with respect to the positive class (1 = up).

    `zero_division=0` keeps precision/recall finite when a model predicts a
    single class on a fold — a degenerate but real outcome we'd rather see as
    0.0 than as a crash.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        # ROC-AUC is undefined when only one class is present in y_true.
        if len(np.unique(y_true)) < 2:
            metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics
