"""Hacker News (Algolia) adapter — optional, free, tech-skewed supplement.

Algolia's HN Search API (https://hn.algolia.com/api) is free, no key, full
history. Coverage skews heavily toward tech, so this is a *supplement* for
mega-cap tech names (AAPL, NVDA, MSFT, …), not a general source. Included
because it's free and the marginal cost of wiring it is one small module.

Endpoint: ``GET /api/v1/search?query=...&tags=story&numericFilters=created_at_i>=...,created_at_i<...``
``created_at_i`` is a Unix timestamp (seconds) — the true publish time.

**Caveat — single page, no pagination.** This adapter issues one request with
``hitsPerPage`` (default 100), so a window with more than ``hitsPerPage``
matching stories is **silently truncated** to the most-relevant page. Fine for
its intended role (an occasional tech-name supplement); if HN ever becomes a
primary source, add Algolia ``page=`` paging or switch to ``search_by_date``.

HTTP-only — uses ``requests`` (a base dep), no optional extra.
"""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_SEARCH = "https://hn.algolia.com/api/v1/search"


class HackerNewsSource(BaseNewsSource):
    name = "hn"

    def __init__(self, hits_per_page: int = 100) -> None:
        self.hits_per_page = int(hits_per_page)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        start_i = int(start.replace(tzinfo=timezone.utc).timestamp())
        end_i = int(end.replace(tzinfo=timezone.utc).timestamp())
        params = {
            "query": ticker,
            "tags": "story",
            "numericFilters": f"created_at_i>={start_i},created_at_i<{end_i}",
            "hitsPerPage": self.hits_per_page,
        }
        resp = requests.get(_SEARCH, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json() or {}
        items: list[NewsItem] = []
        for hit in payload.get("hits", []) or []:
            created = hit.get("created_at_i")
            if not created:
                continue
            try:
                published = datetime.fromtimestamp(int(created), tz=timezone.utc).replace(
                    tzinfo=None
                )
            except (ValueError, OverflowError, OSError):
                continue
            title = str(hit.get("title") or hit.get("story_title") or "")
            if not title:
                continue
            items.append(
                NewsItem(
                    ticker=ticker,
                    published_at=published,
                    source=self.name,
                    title=title,
                    body=str(hit.get("story_text", "") or ""),
                    url=str(hit.get("url", "") or ""),
                    meta={
                        "objectID": hit.get("objectID"),
                        "points": hit.get("points"),
                        "num_comments": hit.get("num_comments"),
                        "author": hit.get("author"),
                    },
                )
            )
        return items
