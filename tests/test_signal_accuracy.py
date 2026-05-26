"""Accuracy tests for lidr_ml signals.

Validates that each registered signal produces mathematically correct values via
two independent checks:

  1. Element-wise comparison against a simple inline reference formula (obvious
     by inspection — catches wrong min_periods, wrong normalization, wrong formula).

  2. Spot checks against hand-derived expected values on a known arithmetic price
     series (close = [100, 101, ...]), where SMA values reduce to sums of
     consecutive integers and can be verified without running any code.

When porting a new signal, add one ACCURACY_CASES entry with both a reference_fn
and at least two spot_checks before merging.  For complex smoothing algorithms
(RSI, MACD) use pandas-ta as the reference instead of an inline formula.

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
# ACCURACY_CASES: one tuple per registered signal.
#
# Schema: (name, params, reference_fn, spot_checks)
#
#   name         — signal key in the registry
#   params       — parameter dict passed to the signal
#   reference_fn — lambda(prices, params) -> pd.Series  (simple inline formula,
#                  readable as a direct transcription of the definition)
#   spot_checks  — list of (positional_index, expected_value) pairs derived by
#                  hand from the arithmetic_prices fixture; use float("nan") to
#                  assert a NaN at that index
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
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def arithmetic_prices(n: int = 100) -> pd.DataFrame:
    """Return a price DataFrame with close = [100, 101, ..., 100+n-1].

    The values are exact integers (cast to float), so every SMA is the
    arithmetic mean of a consecutive integer range and can be computed as
    ``(first + last) / 2`` without a calculator.
    """
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,ref_fn,spot_checks", ACCURACY_CASES)
def test_signal_matches_reference(
    name: str,
    params: dict,
    ref_fn,
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


@pytest.mark.parametrize("name,params,ref_fn,spot_checks", ACCURACY_CASES)
def test_signal_spot_checks(
    name: str,
    params: dict,
    ref_fn,
    spot_checks: list,
) -> None:
    """Layer 2: signal must hit hand-derived expected values at specific indices.

    Uses arithmetic_prices (close = [100, 101, ...]) so the expected values
    are derivable by inspection — no library or external tool required.
    """
    prices = arithmetic_prices()
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
