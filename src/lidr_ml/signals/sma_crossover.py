"""SMA crossover signal.

Ported from lidr's `lib/signals/sma.ts`. The value at time t is:

    (sma_fast_t - sma_slow_t) / sma_slow_t

i.e. how far above (positive) or below (negative) the slow MA the fast MA sits,
expressed as a fraction of the slow MA. This is a continuous version of the
binary "golden cross / death cross" idea — it carries the *magnitude* of the
crossover, which is more useful as a model feature than a binary flag.

Lookahead-safety: pandas `.rolling(window).mean()` is left-anchored (uses only
data through time t), so this signal is naturally lookahead-safe.
"""

from __future__ import annotations

import pandas as pd

from lidr_ml.signals.registry import register


@register("sma_crossover")
def sma_crossover(prices: pd.DataFrame, params: dict) -> pd.Series:
    fast = int(params["fast"])
    slow = int(params["slow"])
    if fast >= slow:
        raise ValueError(f"sma_crossover requires fast < slow, got fast={fast}, slow={slow}")

    close = prices["close"]
    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()

    feature = (sma_fast - sma_slow) / sma_slow
    feature.name = f"sma_crossover_{fast}_{slow}"
    return feature
