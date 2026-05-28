"""sentiment_level — trailing-window mean of per-item sentiment.

What it watches: the average sentiment of items about ``ticker`` over a
rolling lookback. A persistently positive level means a steady stream of
favorable coverage; near zero means mixed/no opinion.

Lookahead-safe by construction: items are first daily-aggregated, then
shifted forward by one trading day at alignment, then a rolling window is
applied on the trading-day series.
"""

from __future__ import annotations

import pandas as pd

from news_sentiment.features._common import align_to_trading_days, items_to_daily
from news_sentiment.features.registry import register
from news_sentiment.types import ScoredItem


@register("sentiment_level")
def sentiment_level(
    items: list[ScoredItem],
    price_index: pd.DatetimeIndex,
    params: dict,
) -> pd.Series:
    lookback = int(params.get("lookback_days", 5))
    daily = items_to_daily(items)
    if daily.empty:
        return pd.Series(0.0, index=price_index, name="sentiment_level")
    aligned = align_to_trading_days(daily["mean_sentiment"].rename("sentiment_level"), price_index)
    return aligned.rolling(window=lookback, min_periods=1).mean().rename("sentiment_level")
