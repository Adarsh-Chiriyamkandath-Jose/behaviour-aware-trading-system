"""Tests for quantitative features."""
from src.features import build_features
from src.features.technical import add_returns, add_rsi


def _config():
    return {
        "features": {
            "sma_windows": [10, 20],
            "ema_windows": [12],
            "rsi_period": 14,
            "momentum_period": 10,
            "volatility_window": 20,
            "rolling_windows": [5, 20],
        }
    }


def test_rsi_stays_within_bounds(sample_ohlcv):
    out = add_rsi(sample_ohlcv, period=14)
    rsi = out["rsi_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_returns_match_pct_change(sample_ohlcv):
    out = add_returns(sample_ohlcv)
    assert out["return"].iloc[1] == sample_ohlcv["Close"].pct_change().iloc[1]


def test_build_features_adds_expected_columns(sample_ohlcv):
    out = build_features(sample_ohlcv, _config())
    for col in ["return", "sma_10", "ema_12", "rsi_14", "momentum_10", "volatility_20", "roll_std_20"]:
        assert col in out.columns


def test_features_do_not_use_the_future(sample_ohlcv):
    """Truncating the tail must not change earlier feature values (no lookahead)."""
    full = build_features(sample_ohlcv, _config())
    partial = build_features(sample_ohlcv.iloc[:200], _config())
    common = full.loc[partial.index, "rsi_14"].dropna()
    assert (common == partial["rsi_14"].dropna()).all()
