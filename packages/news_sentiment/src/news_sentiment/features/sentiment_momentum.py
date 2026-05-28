"""sentiment_momentum — fast-minus-slow trailing sentiment.

What it watches: change in sentiment direction. A positive value means
recent items are more positive than the longer trailing average — a
sentiment up-turn — and vice versa. The shape is analogous to MACD on a
price series.
"""

from __future__ import annotations

import pandas as pd

from news_sentiment.features._common import align_to_trading_days, items_to_daily
from news_sentiment.features.registry import register
from news_sentiment.types import ScoredItem


@register("sentiment_momentum")
def sentiment_momentum(
    items: list[ScoredItem],
    price_index: pd.DatetimeIndex,
    params: dict,
) -> pd.Series:
    fast = int(params.get("fast", 3))
    slow = int(params.get("slow", 10))
    if fast >= slow:
        raise ValueError(f"sentiment_momentum needs fast < slow, got fast={fast}, slow={slow}")
    daily = items_to_daily(items)
    if daily.empty:
        return pd.Series(0.0, index=price_index, name="sentiment_momentum")
    aligned = align_to_trading_days(
        daily["mean_sentiment"].rename("sentiment_momentum"), price_index
    )
    fast_ma = aligned.rolling(window=fast, min_periods=1).mean()
    slow_ma = aligned.rolling(window=slow, min_periods=1).mean()
    return (fast_ma - slow_ma).rename("sentiment_momentum")
