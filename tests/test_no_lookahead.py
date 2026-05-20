"""Assert that every registered signal is lookahead-safe.

For each signal, we compute its value over the full price series, then
recompute it over a series truncated at time t, and assert the value at
time t matches. If a signal accidentally peeks at data after t, the two
will differ and this test will fail.

When adding a new signal, add a (name, params) entry to SIGNAL_CASES below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lidr_ml.signals import get_signal
# Importing the package triggers signal registration via signals/__init__.py.

# Add one entry per signal as you port them.
SIGNAL_CASES = [
    ("sma_crossover", {"fast": 20, "slow": 50}),
]


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    dates = pd.bdate_range("2010-01-01", periods=n)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


@pytest.mark.parametrize("name,params", SIGNAL_CASES)
def test_signal_has_no_lookahead(name: str, params: dict, synthetic_prices: pd.DataFrame) -> None:
    fn = get_signal(name)
    full = fn(synthetic_prices, params)

    # Check several points across the series.
    check_dates = synthetic_prices.index[[200, 300, 400, 500]]
    for t in check_dates:
        truncated = synthetic_prices.loc[:t]
        partial = fn(truncated, params)
        if pd.isna(full.loc[t]):
            assert pd.isna(partial.loc[t]), f"{name}: full is NaN but truncated is not at {t}"
        else:
            assert np.isclose(full.loc[t], partial.loc[t], equal_nan=False), (
                f"{name} leaks future info at {t}: full={full.loc[t]}, truncated={partial.loc[t]}"
            )
