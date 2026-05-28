"""Volume-ratio signal — today's volume relative to its N-day average.

Adapted from lidr's `lib/signals/volume.ts`. The TS variant is a *composite*
signal: it fires BUY/SELL only when price is within 2% of a breakout high/low
AND volume is heavy (≥ volumeMultiplier × average). For ML we strip the
composite structure and emit just the raw underlying measurement:

    feature = today_volume / N-day-average-volume

This is the same `volMultiple` quantity the TS computes internally (the
"how unusual is today's volume?" signal). The breakout context is already
exposed as a separate feature by the breakout signal — there's no reason
to fuse them inside the feature, the model can learn the interaction.

Values above 1.0 mean today's volume is above its N-day average; above ~1.5
is the textbook "unusual volume" threshold. Below 1.0 means thin volume.

Average convention matches lidr's TS: the N-day window **includes** today
(volumes.slice(-N) is N values ending at today, inclusive). pandas'
default `rolling(N).mean()` matches this exactly.

NaN convention: first (period − 1) rows are NaN (window not yet full). If
the window has zero average (all volumes in the window are 0) the ratio is
0/0 = NaN — natural pandas behavior, no special handling needed.

Lookahead-safety: rolling mean is left-anchored; depends only on volumes
at or before time t.
"""

from __future__ import annotations

import pandas as pd

from ta_ensemble.signals.registry import register


@register("volume")
def volume(prices: pd.DataFrame, params: dict) -> pd.Series:
    period = int(params["period"])
    if period < 2:
        raise ValueError(f"volume requires period >= 2, got period={period}")

    vol = prices["volume"]
    avg_vol = vol.rolling(period, min_periods=period).mean()
    feature = vol / avg_vol
    feature.name = f"volume_ratio_{period}"
    return feature
