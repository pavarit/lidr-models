"""Feature registry — ``name → feature callable`` lookup.

A news feature is a callable ``(scored_items, price_index, params) -> pd.Series``
where the returned Series is aligned to ``price_index`` (a ``DatetimeIndex``
of trading days) and is **lookahead-safe**: the value at index ``t`` may
depend only on items with ``published_at < t``.

The price-index parameter lets every feature return a Series the pipeline
can concat into a single feature matrix without any extra alignment work.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from news_sentiment.types import ScoredItem

NewsFeatureFn = Callable[[list[ScoredItem], pd.DatetimeIndex, dict], pd.Series]

REGISTRY: dict[str, NewsFeatureFn] = {}


def register(name: str) -> Callable[[NewsFeatureFn], NewsFeatureFn]:
    """Decorator: register ``fn`` under ``name`` in the global feature registry."""

    def _wrap(fn: NewsFeatureFn) -> NewsFeatureFn:
        if name in REGISTRY:
            raise ValueError(f"News feature {name!r} already registered.")
        REGISTRY[name] = fn
        return fn

    return _wrap


def get_feature(name: str) -> NewsFeatureFn:
    if name not in REGISTRY:
        raise KeyError(f"Unknown news feature {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


# Side-effect imports so @register decorators run at startup.
from news_sentiment.features import (  # noqa: E402, F401
    abnormal_mention_volume,
    sentiment_level,
    sentiment_momentum,
)
