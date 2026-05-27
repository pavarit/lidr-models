"""MACD (Moving Average Convergence Divergence) signal.

Ported from lidr's `lib/signals/macd.ts`. MACD tracks the gap between a fast
and a slow exponential moving average of price; a "signal line" smooths the
gap; the histogram (macd - signal) captures momentum-of-momentum.

We emit the histogram normalized by the slow EMA:

    feature = (macd_line - signal_line) / slow_ema

This is the same normalization lidr's TS uses for its confidence score, just
signed instead of absolute (so the model can learn both directions). The
normalization makes the feature comparable across tickers and price levels.

EMA matches lidr's TS exactly:
    k          = 2 / (period + 1)
    ema[0]     = closes[0]
    ema[i]     = closes[i] * k + ema[i-1] * (1 - k)

This is bit-equivalent to `pd.Series.ewm(span=period, adjust=False).mean()`.

NaN convention: lidr's TS refuses to compute MACD until the input has at least
`slow + signal + 5` observations (its `minLen` check). We mirror that — the
first `slow + signal + 5 - 1` rows are NaN, valid values follow. The +5 buffer
gives the EMAs time to decay past the seed bias before the model sees them.

Lookahead-safety: each EMA at index i depends only on closes[0..i]; MACD and
signal lines inherit that property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lidr_ml.signals.registry import register


@register("macd")
def macd(prices: pd.DataFrame, params: dict) -> pd.Series:
    fast = int(params["fast"])
    slow = int(params["slow"])
    signal_p = int(params["signal"])
    if fast >= slow:
        raise ValueError(f"macd requires fast < slow, got fast={fast}, slow={slow}")
    if signal_p < 1:
        raise ValueError(f"macd requires signal >= 1, got signal={signal_p}")

    close = prices["close"]

    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
    histogram = macd_line - signal_line

    feature = histogram / slow_ema

    warmup = slow + signal_p + 5
    if len(feature) >= warmup:
        feature.iloc[: warmup - 1] = np.nan
    else:
        feature[:] = np.nan

    feature.name = f"macd_{fast}_{slow}_{signal_p}"
    return feature
