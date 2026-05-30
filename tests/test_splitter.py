"""Tests for time-aware splitting — guards against future leakage."""
import pytest


@pytest.mark.skip(reason="TODO: implement once splitter is built")
def test_train_always_before_test():
    """Every train index must be chronologically before its test index."""
    ...
