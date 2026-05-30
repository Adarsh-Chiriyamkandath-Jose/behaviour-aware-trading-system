"""Tests for time-aware splitting - the guard against future leakage."""
from src.data.splitter import walk_forward_splits, single_split


def test_train_always_before_test(sample_ohlcv):
    for train_idx, test_idx in walk_forward_splits(sample_ohlcv, n_splits=4):
        assert train_idx.max() < test_idx.min()


def test_gap_creates_embargo(sample_ohlcv):
    train_idx, test_idx = single_split(sample_ohlcv, test_size=0.2, gap=5)
    # The 5-row embargo means the last training date is well before the test start.
    assert train_idx.max() < test_idx.min()
    assert len(train_idx) == int(len(sample_ohlcv) * 0.8) - 5
