"""Accuracy tests for lidr_ml signals.

Validates that each registered signal produces mathematically correct values via
two independent checks:

  1. Element-wise comparison against a simple inline reference formula (obvious
     by inspection — catches wrong min_periods, wrong normalization, wrong formula).

  2. Spot checks against hand-derived expected values on a known synthetic price
     series, where the values can be verified without running any code.

When porting a new signal, add one ACCURACY_CASES entry with both a reference_fn
and at least two spot_checks before merging.  For complex smoothing algorithms
(RSI, MACD), the reference_fn should be a structurally-different reimplementation
of the same algorithm so a shared bug between signal and reference is unlikely;
spot checks are then the independent ground truth.

The ``synthetic_prices`` fixture used by ``test_signal_matches_reference`` is
defined in ``tests/conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lidr_ml.signals import get_signal

# Importing the package triggers signal registration via signals/__init__.py.


# ---------------------------------------------------------------------------
# Helpers — price fixtures used for hand-derived spot checks.
# ---------------------------------------------------------------------------


def arithmetic_prices(n: int = 100) -> pd.DataFrame:
    """Return a price DataFrame with close = [100, 101, ..., 100+n-1].

    The values are exact integers (cast to float), so every SMA is the
    arithmetic mean of a consecutive integer range and can be computed as
    ``(first + last) / 2`` without a calculator.
    """
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame({"close": close})


def zigzag_prices(n: int = 30) -> pd.DataFrame:
    """Return a price series with deltas alternating +2, -1, +2, -1, ...

    Starting at close[0] = 100, this gives close = [100, 102, 101, 103, 102,
    104, 103, ...]. For RSI with period=14, the seed window covers exactly
    7 gains of 2 and 7 losses of 1 (avg_gain=1.0, avg_loss=0.5, RS=2.0),
    making the seeded value derivable by hand.
    """
    deltas = np.tile([2.0, -1.0], (n + 1) // 2)[: n - 1]
    close = np.concatenate([[100.0], 100.0 + np.cumsum(deltas)])
    return pd.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# Reference implementations for non-trivial signals.
# ---------------------------------------------------------------------------


def _rsi_reference(prices: pd.DataFrame, params: dict) -> pd.Series:
    """Reference RSI implementation — structurally different from the signal.

    Operates on ``np.diff(close)`` (length n-1, no prepended NaN) instead of
    ``close.diff()``. The recursion math is identical to Wilder's smoothing;
    a shared bug between this and ``signals/rsi.py`` is unlikely.
    """
    period = params["period"]
    close = prices["close"].to_numpy()
    n = len(close)
    diffs = np.diff(close)
    gains = np.maximum(diffs, 0.0)
    losses = -np.minimum(diffs, 0.0)

    rsi = np.full(n, np.nan)
    if n > period:
        # Output index k corresponds to diffs[k-1]. Seed at output index = period.
        ag = gains[:period].mean()
        al = losses[:period].mean()
        rsi[period] = 100.0 if al == 0.0 else 100.0 - 100.0 / (1.0 + ag / al)
        for k in range(period + 1, n):
            ag = (ag * (period - 1) + gains[k - 1]) / period
            al = (al * (period - 1) + losses[k - 1]) / period
            rsi[k] = 100.0 if al == 0.0 else 100.0 - 100.0 / (1.0 + ag / al)

    return pd.Series(rsi, index=prices.index)


# ---------------------------------------------------------------------------
# ACCURACY_CASES: one tuple per registered signal.
#
# Schema: (name, params, reference_fn, prices_factory, spot_checks)
#
#   name           — signal key in the registry
#   params         — parameter dict passed to the signal
#   reference_fn   — callable(prices, params) -> pd.Series, computed
#                    independently of the signal implementation
#   prices_factory — callable() -> pd.DataFrame, the price fixture used for
#                    the hand-derived spot checks (chosen so the expected
#                    values fall out without arithmetic)
#   spot_checks    — list of (positional_index, expected_value) pairs derived
#                    by hand from prices_factory(); use float("nan") to assert
#                    a NaN at that index
# ---------------------------------------------------------------------------

ACCURACY_CASES = [
    (
        "sma_crossover",
        {"fast": 5, "slow": 10},
        # Reference: direct transcription of the formula in sma_crossover.py.
        # "How far above/below the slow MA the fast MA sits, as a fraction of
        # the slow MA."  Deliberately written without importing the signal module
        # so that an accidental breakage in the signal module is caught here.
        lambda prices, p: (
            prices["close"].rolling(p["fast"], min_periods=p["fast"]).mean()
            - prices["close"].rolling(p["slow"], min_periods=p["slow"]).mean()
        )
        / prices["close"].rolling(p["slow"], min_periods=p["slow"]).mean(),
        arithmetic_prices,
        # Spot checks on arithmetic_prices (close[i] = 100 + i, step=1).
        #
        # For any arithmetic series with step s=1:
        #   fast_sma[i]  = i - (fast-1)/2   (e.g. 107.0 at i=9, fast=5)
        #   slow_sma[i]  = i - (slow-1)/2   (e.g. 104.5 at i=9, slow=10)
        #   fast - slow  = (slow - fast) / 2 = 2.5  (constant for these params)
        #   signal       = 2.5 / slow_sma[i]
        #
        # All values below are derivable with pencil and paper.
        [
            (8,  float("nan")),   # slow window not yet full (need 10 rows, only 9 available)
            (9,  2.5 / 104.5),    # first valid: slow=mean(100..109)=104.5, fast=mean(105..109)=107.0
            (10, 2.5 / 105.5),    # slow=mean(101..110)=105.5, fast=mean(106..110)=108.0
            (50, 2.5 / 145.5),    # slow=mean(141..150)=145.5, fast=mean(146..150)=148.0
        ],
    ),
    (
        "rsi",
        {"period": 14},
        _rsi_reference,
        zigzag_prices,
        # Spot checks on zigzag_prices (deltas alternating +2, -1, +2, -1, ...).
        # diff[i] = +2 for odd i, -1 for even i.
        #
        # Seed at index 14 covers diff[1..14] = 7 gains of 2 + 7 losses of 1.
        #   avg_gain[14] = 14/14 = 1.0
        #   avg_loss[14] =  7/14 = 0.5
        #   rs[14]       = 2.0
        #   rsi[14]      = 100 - 100/(1+2) = 200/3
        #
        # At index 15, diff[15] = +2 (gain), loss = 0:
        #   avg_gain[15] = (1.0 * 13 + 2)   / 14 = 15/14
        #   avg_loss[15] = (0.5 * 13 + 0)   / 14 = 13/28
        #   rs[15]       = (15/14) / (13/28) = 30/13
        #   rsi[15]      = 100 - 100/(1 + 30/13) = 3000/43
        [
            (13, float("nan")),   # last index before seed (period=14 → seed at idx 14)
            (14, 200.0 / 3.0),    # seed: avg_gain=1, avg_loss=0.5, rs=2, rsi=66.6667
            (15, 3000.0 / 43.0),  # first recursion step ≈ 69.7674
        ],
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,ref_fn,prices_factory,spot_checks", ACCURACY_CASES)
def test_signal_matches_reference(
    name: str,
    params: dict,
    ref_fn,
    prices_factory,
    spot_checks: list,
    synthetic_prices: pd.DataFrame,
) -> None:
    """Layer 1: signal output must agree element-wise with the inline reference.

    Uses the shared ``synthetic_prices`` fixture (600-day log-normal series)
    so that numerical edge-cases on real-world-shaped data are also covered.
    """
    fn = get_signal(name)
    actual = fn(synthetic_prices, params)
    expected = ref_fn(synthetic_prices, params)

    # NaN positions must agree exactly.
    assert (actual.isna() == expected.isna()).all(), (
        f"{name}: NaN mask mismatch between signal and reference"
    )

    # Non-NaN values must match to machine precision.
    mask = ~actual.isna()
    np.testing.assert_allclose(
        actual[mask].to_numpy(),
        expected[mask].to_numpy(),
        rtol=1e-12,
        err_msg=f"{name}: signal diverges from reference formula",
    )


@pytest.mark.parametrize("name,params,ref_fn,prices_factory,spot_checks", ACCURACY_CASES)
def test_signal_spot_checks(
    name: str,
    params: dict,
    ref_fn,
    prices_factory,
    spot_checks: list,
) -> None:
    """Layer 2: signal must hit hand-derived expected values at specific indices.

    Each case carries its own ``prices_factory`` so the spot checks can use a
    price series tuned for hand-derivation — no library or external tool required.
    """
    prices = prices_factory()
    fn = get_signal(name)
    result = fn(prices, params)

    for idx, expected_val in spot_checks:
        actual_val = result.iloc[idx]
        if isinstance(expected_val, float) and np.isnan(expected_val):
            assert pd.isna(actual_val), (
                f"{name}: expected NaN at index {idx}, got {actual_val}"
            )
        else:
            np.testing.assert_allclose(
                actual_val,
                expected_val,
                rtol=1e-12,
                err_msg=f"{name}: wrong value at index {idx}",
            )
