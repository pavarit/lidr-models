"""GDELT 2.0 DOC API adapter.

Free, no key, deep history. The research doc calls GDELT "the backtest
workhorse for news volume/tone." We query the ``artlist`` mode (article-level
URLs + tone) over the requested window.

GDELT returns ISO-format ``seendate`` timestamps; we use those as
``published_at`` after parsing. GDELT itself indexes within 15 minutes of
publication, so this is effectively point-in-time for daily features.
"""

from __future__ import annotations

from datetime import datetime

import requests

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltSource(BaseNewsSource):
    name = "gdelt"

    def __init__(self, max_records: int = 250) -> None:
        self.max_records = int(max_records)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        params = {
            "query": f'"{ticker}"',
            "mode": "artlist",
            "format": "json",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
            "maxrecords": str(self.max_records),
            "sort": "datedesc",
        }
        resp = requests.get(_GDELT_DOC_API, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        items: list[NewsItem] = []
        for art in payload.get("articles", []) or []:
            seendate = art.get("seendate") or ""
            try:
                published = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
            except ValueError:
                continue
            items.append(
                NewsItem(
                    ticker=ticker,
                    published_at=published,
                    source=self.name,
                    title=str(art.get("title", "") or ""),
                    body="",
                    url=str(art.get("url", "") or ""),
                    meta={
                        "domain": art.get("domain"),
                        "language": art.get("language"),
                        "tone": art.get("tone"),
                    },
                )
            )
        return items
