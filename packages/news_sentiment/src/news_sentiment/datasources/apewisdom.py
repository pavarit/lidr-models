"""Apewisdom retail-attention adapter.

Free, no auth. Replaces Reddit's *live* role: Apewisdom aggregates mention
counts and sentiment across WSB / r/stocks / r/investing / r/options /
r/SPACs / 4chan-biz and exposes them through a simple ranking API.

**Live-snapshot only.** The public API returns the *current* ranking (plus a
24h-ago comparison), not a queryable history. So this adapter cannot serve a
backtest window directly — instead it emits one synthetic ``NewsItem`` per
ticker for the *current* day, carrying the mention count + sentiment in
``meta``. The collector forward-collects these day by day, building our own
historical store over time. That is why Apewisdom features are excluded from
``news_v0.yaml``'s first backtest (no history yet at PR-C run-time); they enter
a later iteration once months of snapshots have accumulated.

Endpoint: ``GET /api/v1.0/filter/{filter}/page/{n}`` — ``filter`` defaults to
``all-stocks`` (surfaced as a config field); 100 results/page. We page until we
find the requested ticker or exhaust ``max_pages``.

HTTP-only — uses ``requests`` (a base dep), no optional extra.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import requests

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_API = "https://apewisdom.io/api/v1.0/filter"


class ApewisdomSource(BaseNewsSource):
    name = "apewisdom"

    def __init__(self, filter: str = "all-stocks", max_pages: int = 5) -> None:
        self.filter = str(filter)
        self.max_pages = int(max_pages)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        # Live snapshot: stamp "now" (UTC). The base class window filter keeps
        # this item only if it falls inside [start, end) — so a historical
        # backtest window returns nothing, which is the honest behaviour
        # (Apewisdom has no history; we forward-collect).
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        target = ticker.upper()

        row = self._find_ticker(target)
        if row is None:
            return []

        mentions = _as_int(row.get("mentions"))
        published = datetime.combine(now.date(), time(23, 59, 59))
        title = f"{target} retail attention: {mentions} mentions"
        return [
            NewsItem(
                ticker=ticker,
                published_at=published,
                source=self.name,
                title=title,
                body="",
                url=f"https://apewisdom.io/stocks/{target}/",
                meta={
                    "mentions": mentions,
                    "mentions_24h_ago": _as_int(row.get("mentions_24h_ago")),
                    "rank": _as_int(row.get("rank")),
                    "upvotes": _as_int(row.get("upvotes")),
                    "apewisdom_sentiment": row.get("sentiment"),
                    "filter": self.filter,
                    "snapshot": True,
                },
            )
        ]

    def _find_ticker(self, target: str) -> dict | None:
        for page in range(1, self.max_pages + 1):
            resp = requests.get(f"{_API}/{self.filter}/page/{page}", timeout=30)
            resp.raise_for_status()
            payload = resp.json() or {}
            results = payload.get("results", []) or []
            for row in results:
                if str(row.get("ticker", "")).upper() == target:
                    return row
            pages = _as_int(payload.get("pages"))
            if pages and page >= pages:
                break
        return None


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
