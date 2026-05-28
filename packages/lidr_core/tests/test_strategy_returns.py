"""Regression tests for the equity-curve calculation in add_strategy_returns.

These guard the fix for the overlapping-returns bug: each period's return must
be compounded exactly once, and days the strategy is in cash must earn nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lidr_core.backtest.engine import add_strategy_returns


def _preds(y_pred: list[int], idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"y_true": 1, "y_pred": y_pred, "y_proba_1": 0.6},
        index=idx,
    )


def test_equity_compounds_each_return_once() -> None:
    # Always invested, no costs: equity must be the simple compounding of the
    # per-day returns — each counted exactly once (no overlap).
    idx = pd.bdate_range("2020-01-01", periods=3)
    preds = _preds([1, 1, 1], idx)
    fwd = pd.Series([0.01, 0.01, 0.01], index=idx)

    out = add_strategy_returns(preds, fwd, transaction_cost_bps=0.0)

    expected = np.cumprod([1.01, 1.01, 1.01])
    assert np.allclose(out["strategy_equity"].values, expected)
    assert np.allclose(out["buy_hold_equity"].values, expected)


def test_cash_days_earn_nothing() -> None:
    # Position 1, 0, 1: the middle (cash) day must not move the equity curve.
    idx = pd.bdate_range("2020-01-01", periods=3)
    preds = _preds([1, 0, 1], idx)
    fwd = pd.Series([0.02, 0.02, 0.02], index=idx)

    out = add_strategy_returns(preds, fwd, transaction_cost_bps=0.0)

    expected = np.array([1.02, 1.02, 1.02 * 1.02])
    assert np.allclose(out["strategy_equity"].values, expected)


def test_transaction_costs_charged_on_position_change() -> None:
    # Entering a position on day 0 (from flat) incurs one round of cost.
    idx = pd.bdate_range("2020-01-01", periods=2)
    preds = _preds([1, 1], idx)
    fwd = pd.Series([0.0, 0.0], index=idx)

    out = add_strategy_returns(preds, fwd, transaction_cost_bps=10.0)

    # Day 0: flat -> long is a change, so charged 10bps. Day 1: no change.
    assert np.isclose(out["strategy_return"].iloc[0], -0.0010)
    assert np.isclose(out["strategy_return"].iloc[1], 0.0)
