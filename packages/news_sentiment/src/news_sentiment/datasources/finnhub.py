"""Finnhub company-news adapter.

Free personal-use tier: 60 calls/min, ~1 year of US company news plus
real-time updates (research doc, 2026-05-28). The free backbone for live US
single-name news. Premium is only needed for international coverage or history
beyond 1 year — neither is required for a starting backtest.

Endpoint: ``GET /api/v1/company-news?symbol=AAPL&from=YYYY-MM-DD&to=YYYY-MM-DD``
authenticated with ``token=$FINNHUB_API_KEY``. Returns a flat JSON array of
articles; each carries a Unix ``datetime`` (seconds) which is the true publish
time we map onto ``NewsItem.published_at``.

HTTP-only — uses ``requests`` (a base dep), no optional extra. The API key is
read from the ``FINNHUB_API_KEY`` env var so a missing key is reported with a
clear message rather than firing a silent 401.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"


class FinnhubSource(BaseNewsSource):
    name = "finnhub"

    def __init__(self, request_delay_s: float = 1.1, api_key: str | None = None) -> None:
        # Free tier is 60 calls/min; a ~1.1s spacing keeps a single-ticker
        # backtest comfortably under the limit without external rate limiting.
        self.request_delay_s = float(request_delay_s)
        self._api_key = api_key

    def _key(self) -> str:
        key = self._api_key or os.environ.get("FINNHUB_API_KEY")
        if not key:
            raise RuntimeError(
                "FinnhubSource needs FINNHUB_API_KEY in env. "
                "Get a free key at https://finnhub.io/dashboard."
            )
        return key

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        params = {
            "symbol": ticker,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "token": self._key(),
        }
        resp = requests.get(_COMPANY_NEWS, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        # Finnhub returns a flat array of articles; an {"error": ...} dict (with
        # HTTP 200) signals an auth/plan problem. Guard so the loop doesn't
        # iterate the dict's keys and crash with an opaque AttributeError.
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(
                f"Finnhub company-news API error: {payload['error']!r}. "
                "Check that FINNHUB_API_KEY is correct and the plan grants "
                "company-news access."
            )
        items: list[NewsItem] = []
        for art in payload or []:
            ts = art.get("datetime")
            if not ts:
                continue
            try:
                published = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)
            except (ValueError, OverflowError, OSError):
                continue
            headline = str(art.get("headline", "") or "")
            if not headline:
                continue
            items.append(
                NewsItem(
                    ticker=ticker,
                    published_at=published,
                    source=self.name,
                    title=headline,
                    body=str(art.get("summary", "") or ""),
                    url=str(art.get("url", "") or ""),
                    meta={
                        "id": art.get("id"),
                        "category": art.get("category"),
                        "finnhub_source": art.get("source"),
                    },
                )
            )
        time.sleep(self.request_delay_s)
        return items
