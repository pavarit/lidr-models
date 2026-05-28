"""Base class for every news adapter.

Implements ``lidr_core.protocols.datasource.DataSource`` with the news-shaped
return type. Subclasses override ``fetch_raw`` to do the real I/O; the base
class handles uniform conversion to ``NewsItem`` and the timestamp invariants
the collector relies on.

Point-in-time discipline lives at this layer: ``published_at`` MUST be the
true publish timestamp the source reported, never the time we ingested it.
The collector relies on this; the lookahead test relies on this. If a source
can only give us a (sometimes-revised) "as-of" timestamp, document that
caveat in the adapter and treat it accordingly downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from news_sentiment.types import NewsItem


class BaseNewsSource(ABC):
    """A pluggable news data source.

    Subclasses implement ``fetch_raw(ticker, start, end)`` to return a list of
    ``NewsItem``. ``fetch`` is the public entry point and may apply uniform
    post-processing (timestamp clamping, sorting) once we firm it up.
    """

    name: str = "base"

    @abstractmethod
    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        """Return items strictly in ``[start, end)`` for ``ticker``.

        Implementations MUST set ``published_at`` to the true publish
        timestamp from the source. Items outside the window may be returned —
        ``fetch`` will filter — but must still carry a real timestamp.
        """

    def fetch(self, ticker: str, start: str, end: str, **_: object) -> list[NewsItem]:
        s = _parse_dt(start)
        e = _parse_dt(end)
        raw = self.fetch_raw(ticker, s, e)
        clean = [it for it in raw if s <= it.published_at < e]
        clean.sort(key=lambda it: it.published_at)
        return clean


def _parse_dt(s: str) -> datetime:
    """Permissive date/datetime parser. Accepts 'YYYY-MM-DD' or ISO 8601."""
    return datetime.fromisoformat(s) if "T" in s else datetime.strptime(s, "%Y-%m-%d")
