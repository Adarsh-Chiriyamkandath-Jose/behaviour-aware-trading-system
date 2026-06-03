"""Tests for the prediction target."""
import pandas as pd

from src.data.labels import make_direction_label


def test_label_is_one_when_price_rises():
    df = pd.DataFrame({"Close": [10.0, 11.0, 10.5]})
    label = make_direction_label(df, horizon=1, threshold=0.0)
    assert label.iloc[0] == 1.0   # 10 -> 11 is up
    assert label.iloc[1] == 0.0   # 11 -> 10.5 is down


def test_last_rows_have_no_label(sample_ohlcv):
    label = make_direction_label(sample_ohlcv, horizon=3)
    assert label.iloc[-3:].isna().all()
