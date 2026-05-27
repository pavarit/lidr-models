"""Bollinger Bands signal.

Ported from lidr's `lib/signals/bollinger.ts`. Bollinger Bands wrap price with
an N-day moving average plus/minus K standard deviations. When price punches
above the upper band it's statistically far above its recent norm
(overextended); below the lower band it's similarly stretched downward.

For ML we emit the raw **z-score** of price relative to the recent mean:

    feature = (close - sma_N) / std_N

This is the continuous, signed, unclipped version of what lidr's TS confidence
formula computes. The model can learn its own thresholds — there's no reason
to hard-code the conventional ±2σ "overextended" cutoff into the feature.

Standard deviation matches lidr's TS exactly: **population std** (divides by
N, not N−1). pandas defaults to sample std (ddof=1), so we explicitly pass
ddof=0.

NaN convention: the first (period − 1) rows are NaN (window not yet full).
If a window has zero variance (perfectly flat) the z-score is undefined and
we emit NaN there too; the model treats NaN as missing.

Lookahead-safety: rolling mean and rolling std are left-anchored on the
input index; both depend only on prices at or before time t.
"""

from __future__ import annotations

import pandas as pd

from lidr_ml.signals.registry import register


@register("bollinger")
def bollinger(prices: pd.DataFrame, params: dict) -> pd.Series:
    period = int(params["period"])
    if period < 2:
        raise ValueError(f"bollinger requires period >= 2, got period={period}")

    close = prices["close"]
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)

    feature = (close - sma) / std
    feature.name = f"bollinger_z_{period}"
    return feature
