"""Shared pytest fixtures for lidr_ml tests.

The ``synthetic_prices`` fixture provides a deterministic 600-day OHLCV
DataFrame (log-normal random walk, seed=0) used by both the lookahead-safety
tests and the signal accuracy tests.  Keeping it here avoids duplication and
ensures both test modules see the same data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """600 business-day log-normal price series, fully deterministic (seed=0)."""
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
