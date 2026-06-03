"""Tests for evaluation metrics."""
import numpy as np
import pandas as pd

from src.evaluation import (
    classification_metrics,
    equity_curve,
    financial_metrics,
    paired_significance,
    strategy_returns,
)


def test_classification_perfect_prediction():
    y = pd.Series([0, 1, 0, 1, 1])
    m = classification_metrics(y, y, y.astype(float))
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0


def test_classification_handles_single_predicted_class():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 0, 0, 0])  # model never says "up"
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == 0.0  # zero_division guard, not a crash
    assert "roc_auc" not in m  # not requested


def test_roc_auc_nan_when_one_class_present():
    y_true = pd.Series([1, 1, 1])
    m = classification_metrics(y_true, y_true, y_true.astype(float))
    assert np.isnan(m["roc_auc"])


def test_strategy_goes_flat_on_down_calls():
    fwd = pd.Series([0.02, -0.03, 0.01], index=pd.RangeIndex(3))
    preds = [1, 0, 1]  # long, flat, long
    rets = strategy_returns(preds, fwd, transaction_cost=0.0)
    # The flat day earns nothing regardless of the market's -3%.
    assert rets.iloc[1] == 0.0
    assert rets.iloc[0] == 0.02


def test_transaction_cost_penalises_turnover():
    fwd = pd.Series([0.0, 0.0, 0.0], index=pd.RangeIndex(3))
    free = strategy_returns([1, 0, 1], fwd, transaction_cost=0.0)
    costed = strategy_returns([1, 0, 1], fwd, transaction_cost=0.01)
    assert costed.sum() < free.sum()


def test_financial_metrics_on_known_curve():
    # Steadily rising equity: positive return, zero drawdown.
    curve = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    m = financial_metrics(curve)
    assert m["total_return"] > 0
    assert m["max_drawdown"] == 0.0
    assert m["sharpe"] > 0


def test_financial_metrics_drawdown_is_negative_after_a_fall():
    curve = pd.Series([100.0, 120.0, 90.0, 95.0])
    m = financial_metrics(curve)
    assert m["max_drawdown"] < 0


def test_paired_significance_detects_consistent_improvement():
    base = [0.50, 0.51, 0.49, 0.50]
    cand = [0.55, 0.56, 0.54, 0.55]  # consistently ~5pp better
    res = paired_significance(base, cand)
    assert res["mean_diff"] > 0
    assert res["t_pvalue"] < 0.05


def test_paired_significance_handles_no_difference():
    res = paired_significance([0.5, 0.5], [0.5, 0.5])
    assert res["mean_diff"] == 0.0
    assert np.isnan(res["t_pvalue"])  # no variation -> undefined, not a crash
