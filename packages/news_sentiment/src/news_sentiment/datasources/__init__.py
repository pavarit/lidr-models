"""Data adapters. Each conforms to ``lidr_core.protocols.datasource.DataSource``
and returns ``list[NewsItem]`` for a (ticker, start, end) window.

A new source is one new module + one line in ``REGISTRY`` — that is the entire
"swappable data source" contract from ADR 0001. Configs select which sources
are active.

Active real adapters: ``synthetic`` (offline dev), ``edgar`` + ``gdelt``
(free, PR-A), ``finnhub`` + ``apewisdom`` (free, PR-B), ``eodhd`` (paid, PR-B),
and the optional ``hn`` (free, tech-skewed). ``reddit`` and ``google_trends``
are kept as **permanent stubs** that raise on use (the data-source rewire,
2026-05-28) — they remain registered only so a stale config gets the real
reason rather than a bare ``KeyError``. See ``docs/research/data-sources.md``.
"""

from __future__ import annotations

from collections.abc import Callable

from news_sentiment.datasources.apewisdom import ApewisdomSource
from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.datasources.edgar import EdgarSource
from news_sentiment.datasources.eodhd import EodhdSource
from news_sentiment.datasources.finnhub import FinnhubSource
from news_sentiment.datasources.gdelt import GdeltSource
from news_sentiment.datasources.google_trends import GoogleTrendsSource
from news_sentiment.datasources.hn import HackerNewsSource
from news_sentiment.datasources.reddit import RedditSource
from news_sentiment.datasources.synthetic import SyntheticSource

REGISTRY: dict[str, Callable[..., BaseNewsSource]] = {
    "synthetic": SyntheticSource,
    "edgar": EdgarSource,
    "gdelt": GdeltSource,
    "finnhub": FinnhubSource,
    "apewisdom": ApewisdomSource,
    "eodhd": EodhdSource,
    "hn": HackerNewsSource,
    # Permanent stubs (raise on use) — kept registered so a stale config gets
    # the real reason, not a bare KeyError. See docs/research/data-sources.md.
    "reddit": RedditSource,
    "google_trends": GoogleTrendsSource,
}


def build_source(name: str, **params) -> BaseNewsSource:
    if name not in REGISTRY:
        raise KeyError(f"Unknown data source {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
