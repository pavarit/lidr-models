"""RSI (Relative Strength Index) signal.

Ported from lidr's `lib/signals/rsi.ts`. RSI is a momentum oscillator that
measures the magnitude of recent up-day moves vs. down-day moves on a 0-100
scale. Conventional thresholds: >70 overbought (potential pullback), <30
oversold (potential mean reversion).

We emit the raw RSI value, not a thresholded action or confidence. The model
is free to learn whatever nonlinear bins it wants over the continuous 0-100
range.

Smoothing follows the classic Wilder method:

    avg_gain[period] = mean(gain[1..period])
    avg_loss[period] = mean(loss[1..period])
    avg_gain[i]      = (avg_gain[i-1] * (period - 1) + gain[i]) / period
    avg_loss[i]      = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    rs[i]            = avg_gain[i] / avg_loss[i]
    rsi[i]           = 100 - 100 / (1 + rs[i])

If avg_loss is zero, RSI is defined as 100 to mirror lidr's TS implementation
(avoids divide-by-zero; treats "no losses in window" as max momentum).

Lookahead-safety: the seeded recursion at each index depends only on close
prices at or before that index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta_ensemble.signals.registry import register


@register("rsi")
def rsi(prices: pd.DataFrame, params: dict) -> pd.Series:
    period = int(params["period"])
    if period < 2:
        raise ValueError(f"rsi requires period >= 2, got period={period}")

    close = prices["close"]
    n = len(close)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).to_numpy()
    loss = (-delta).where(delta < 0, 0.0).to_numpy()

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    if n > period:
        # Seed at index `period`: SMA over the first `period` valid deltas
        # (delta[0] is NaN; delta[1..period] are the first `period` real moves).
        avg_gain[period] = gain[1 : period + 1].mean()
        avg_loss[period] = loss[1 : period + 1].mean()
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = avg_gain / avg_loss
    rsi_val = 100.0 - 100.0 / (1.0 + rs)
    zero_loss = (avg_loss == 0.0) & ~np.isnan(avg_gain)
    rsi_val = np.where(zero_loss, 100.0, rsi_val)

    return pd.Series(rsi_val, index=close.index, name=f"rsi_{period}")
