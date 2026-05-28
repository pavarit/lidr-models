"""Breakout signal — position within the N-day price range.

Ported from lidr's `lib/signals/breakout.ts`. The TS variant emits a BUY/SELL
action based on whether price is within 2% of the N-day high or low. For ML
we emit the **continuous position** of price inside its N-day range:

    feature = (close - low_N) / (high_N - low_N)

This is 0 when price is at the N-day low, 1 when at the N-day high, 0.5 at
the midpoint. The model can learn its own thresholds — there's no reason to
hard-code the 2% "near the extreme" cutoff into the feature itself.

This is essentially the unscaled Williams %R / Stochastic indicator on
closes only. Bounded in [0, 1].

NaN convention: the first (period − 1) rows are NaN (window not yet full).
If the entire window has zero range (price perfectly flat across all N
days) the position is undefined and we emit NaN.

Lookahead-safety: rolling .max() and .min() are left-anchored on the input
index; both depend only on prices at or before time t.
"""

from __future__ import annotations

import pandas as pd

from ta_ensemble.signals.registry import register


@register("breakout")
def breakout(prices: pd.DataFrame, params: dict) -> pd.Series:
    period = int(params["period"])
    if period < 2:
        raise ValueError(f"breakout requires period >= 2, got period={period}")

    close = prices["close"]
    high = close.rolling(period, min_periods=period).max()
    low = close.rolling(period, min_periods=period).min()

    feature = (close - low) / (high - low)
    feature.name = f"breakout_{period}"
    return feature
