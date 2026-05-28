"""DataSource protocol — Task 2 placeholder.

A DataSource is an external ingestion adapter: yfinance, Tiingo News, EDGAR,
GDELT, Reddit, etc. Each one knows how to pull a slice of its native shape
(prices / headlines / filings / posts) for a (ticker, start, end) window and
return it as a pandas object. The pipeline assembles features from a
configured *set* of DataSources, so swapping or adding a source is a config
edit + one adapter file — see ADR 0001 ("Designed for change") for the
contract this enforces.

The shape will firm up in Task 2 (news-sentiment data path); leaving it
permissive avoids prematurely standardizing the return type before two
concrete sources exist.
"""

from __future__ import annotations

from typing import Any, Protocol


class DataSource(Protocol):
    """A pluggable external data adapter.

    Implementations return whatever native shape the source produces (DataFrame
    of OHLCV for yfinance, list/DataFrame of scored headlines for a news
    source). Higher-level Feature implementations bridge that native shape into
    a feature Series aligned to the model's index.
    """

    def fetch(self, ticker: str, start: str, end: str, **kwargs: Any) -> Any: ...
