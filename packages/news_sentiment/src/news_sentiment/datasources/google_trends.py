"""Google Trends (pytrends) adapter — **permanent stub.**

Google Trends is **out** of the data stack. The ``pytrends`` client was
**archived 2025-04-17** and is no longer maintained; it returns 429s at modest
volume and the community ``pytrends-modern`` fork has the same unofficial-API
fragility. We drop search-interest signals rather than replace them — Apewisdom
covers retail-attention more directly than a generic search proxy.

If a non-investor attention proxy ever matters again, SerpAPI's Trends endpoint
(~$50+/mo) or Glimpse are the paid options now that pytrends is dead.

Full reasoning: ``docs/research/data-sources.md``. The class is kept (rather
than deleted) so the registry can name it and surface the reason if a stale
config references it; it raises immediately on use.
"""

from __future__ import annotations

from datetime import datetime

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_REASON = (
    "pytrends was archived 2025-04-17; see docs/research/data-sources.md. "
    "'apewisdom' covers retail-attention more directly."
)


class GoogleTrendsSource(BaseNewsSource):
    name = "google_trends"

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        raise NotImplementedError(_REASON)
