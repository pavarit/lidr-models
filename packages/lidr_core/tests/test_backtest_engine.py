"""Regression tests for the expanding-window backtest engine.

Guards the fix for the duplicate-boundary-date bug: a split's test_end is the
next split's test_start (because the loop sets train_end = test_end), so the
boundary date used to appear in both splits' predictions. Splits are now
half-open on the right and the engine asserts the stitched output index is
unique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lidr_core.backtest.engine import expanding_window_backtest


class _DummyModel:
    """Predicts class 1 with fixed probability — no training, no surprises."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


def _toy_xy(years: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.bdate_range("2010-01-04", periods=252 * years)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.integers(0, 2, size=len(idx)), index=idx)
    return X, y


def test_predictions_index_is_unique() -> None:
    X, y = _toy_xy(years=10)
    result = expanding_window_backtest(
        X, y, model_factory=_DummyModel, initial_train_years=5, test_period_months=12
    )
    assert result.predictions.index.is_unique


def test_consecutive_splits_do_not_overlap() -> None:
    # The bug was: split N's test_end == split N+1's test_start, and both
    # endpoints were inclusive, so the boundary date was predicted twice.
    # Verify no consecutive pair of splits now overlaps.
    X, y = _toy_xy(years=10)
    result = expanding_window_backtest(
        X, y, model_factory=_DummyModel, initial_train_years=5, test_period_months=12
    )

    assert len(result.splits) >= 2, "toy data should produce multiple splits"
    for (_, _, _, end_a), (_, _, start_b, _) in zip(
        result.splits, result.splits[1:], strict=False
    ):
        assert end_a < start_b, f"split test ranges overlap at boundary: {end_a} >= {start_b}"
