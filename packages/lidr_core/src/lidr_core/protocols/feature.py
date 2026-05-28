"""Feature protocol — Task 2 placeholder.

A Feature is a callable that takes whatever inputs it needs (a DataFrame of
prices, a stream of scored news items, regime indicators) and returns a Series
of feature values aligned to a common time index. Signals already conform to a
narrower flavor of this (prices-only); the broader ``Feature`` lets news_sentiment
plug news/sentiment streams into the same feature-matrix shape without forcing
them through the ``SignalFn(prices, params)`` signature.

The concrete shape will firm up in Task 2 (news-sentiment model build); leaving
the interface intentionally permissive avoids over-fitting it to today's
assumptions before the news ingestion code exists.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class Feature(Protocol):
    """A pluggable feature producer.

    Implementations accept whatever named inputs they need (declared per-model
    in config) and return a Series aligned to the model's target index. Each
    Feature must be lookahead-safe in the same sense as a SignalFn: the value
    at index ``t`` may depend only on data observable at or before ``t``.
    """

    def __call__(self, **inputs: Any) -> pd.Series: ...
