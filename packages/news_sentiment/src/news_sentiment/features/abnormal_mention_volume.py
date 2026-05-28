"""abnormal_mention_volume — z-score of item count vs trailing baseline.

What it watches: attention spikes. A high positive value means **way more**
items than usual were published about this ticker recently. Per the plan,
"the spike is the signal" — attention going abnormal is the prior on
something real being underway.

Computed as ``(count_t - rolling_mean_count) / rolling_std_count`` over a
trailing baseline window. NaNs (early/insufficient data) and zero-std
windows are returned as 0.0 so the column stays usable in the feature
matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from news_sentiment.features._common import align_to_trading_days, items_to_daily
from news_sentiment.features.registry import register
from news_sentiment.types import ScoredItem


@register("abnormal_mention_volume")
def abnormal_mention_volume(
    items: list[ScoredItem],
    price_index: pd.DatetimeIndex,
    params: dict,
) -> pd.Series:
    baseline_days = int(params.get("baseline_days", 30))
    daily = items_to_daily(items)
    if daily.empty:
        return pd.Series(0.0, index=price_index, name="abnormal_mention_volume")
    aligned = align_to_trading_days(
        daily["count"].astype(float).rename("abnormal_mention_volume"), price_index
    )
    rolling = aligned.rolling(window=baseline_days, min_periods=5)
    mean = rolling.mean()
    std = rolling.std()
    z = (aligned - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0).rename("abnormal_mention_volume")
