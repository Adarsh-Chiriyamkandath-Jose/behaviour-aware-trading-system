"""Tests for the ablation experiment — the project's core deliverable."""
import pandas as pd

from src.experiments import build_model_frame, feature_columns, run_ablation
from src.experiments.regime_analysis import regime_breakdown, run_regime_analysis


def test_feature_sets_differ_only_by_behavioral(large_ohlcv, sample_config):
    frame = build_model_frame(large_ohlcv, sample_config)
    quant = feature_columns(frame, "quant_only")
    both = feature_columns(frame, "quant_plus_behavioral")

    assert set(quant).isdisjoint({"fear", "greed", "herd"})
    assert set(both) - set(quant) == {"fear", "greed", "herd"}
    # Raw levels and the regime/target columns are never model inputs.
    for leaked in ["Close", "Volume", "regime", "target", "fwd_return", "sma_10"]:
        assert leaked not in both


def test_model_frame_has_no_missing_values(large_ohlcv, sample_config):
    frame = build_model_frame(large_ohlcv, sample_config)
    cols = feature_columns(frame, "quant_plus_behavioral") + ["target", "fwd_return"]
    assert not frame[cols].isna().any().any()
    assert len(frame) > 100  # plenty left after warmup trimming


def test_run_ablation_offline_produces_tables(large_ohlcv, sample_config):
    results = run_ablation(sample_config, raw_data={"TEST": large_ohlcv})

    per_fold = results["per_fold"]
    # 2 models x 2 feature sets x 3 folds = 12 rows for the one ticker.
    assert len(per_fold) == 12
    assert set(per_fold["feature_set"].unique()) == {"quant_only", "quant_plus_behavioral"}
    for col in ["accuracy", "sharpe", "max_drawdown"]:
        assert col in per_fold.columns

    # Summary collapses folds; significance compares the two feature sets.
    assert len(results["summary"]) == 4  # 2 models x 2 feature sets
    sig = results["significance"]
    assert set(sig["metric"].unique()) == {"accuracy", "sharpe"}
    assert (sig["n_folds"] == 3).all()


def test_regime_breakdown_covers_each_regime_plus_all():
    idx = pd.RangeIndex(6)
    y_true = pd.Series([1, 0, 1, 0, 1, 0], index=idx)
    y_pred = [1, 0, 0, 0, 1, 1]
    fwd = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01], index=idx)
    regimes = pd.Series(["bull_calm", "bull_calm", "bear_volatile", "bear_volatile", "bull_calm", "bear_volatile"], index=idx)

    table = regime_breakdown(y_true, y_pred, fwd, regimes)
    assert "ALL" in table["regime"].values
    assert {"bull_calm", "bear_volatile"}.issubset(set(table["regime"]))
    assert table[table["regime"] == "ALL"]["n"].iloc[0] == 6


def test_run_regime_analysis_offline(large_ohlcv, sample_config):
    out = run_regime_analysis(sample_config, raw_data={"TEST": large_ohlcv}, model_name="random_forest")
    assert "TEST" in out
    assert "regime" in out["TEST"].columns
