"""Data adapters. Each conforms to ``lidr_core.protocols.datasource.DataSource``
and returns ``list[NewsItem]`` for a (ticker, start, end) window.

A new source is one new module + one line in ``REGISTRY`` — that is the entire
"swappable data source" contract from ADR 0001. Configs select which sources
are active.
"""

from __future__ import annotations

from collections.abc import Callable

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.datasources.edgar import EdgarSource
from news_sentiment.datasources.gdelt import GdeltSource
from news_sentiment.datasources.google_trends import GoogleTrendsSource
from news_sentiment.datasources.reddit import RedditSource
from news_sentiment.datasources.synthetic import SyntheticSource
from news_sentiment.datasources.tiingo import TiingoSource

REGISTRY: dict[str, Callable[..., BaseNewsSource]] = {
    "synthetic": SyntheticSource,
    "edgar": EdgarSource,
    "gdelt": GdeltSource,
    "reddit": RedditSource,
    "google_trends": GoogleTrendsSource,
    "tiingo": TiingoSource,
}


def build_source(name: str, **params) -> BaseNewsSource:
    if name not in REGISTRY:
        raise KeyError(f"Unknown data source {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
