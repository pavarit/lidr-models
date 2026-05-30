"""Apewisdom retail-attention adapter.

Free, no auth. Replaces Reddit's *live* role: Apewisdom aggregates mention
counts and sentiment across WSB / r/stocks / r/investing / r/options /
r/SPACs / 4chan-biz and exposes them through a simple ranking API.

**Live-snapshot only.** The public API returns the *current* ranking (plus a
24h-ago comparison), not a queryable history. So this adapter cannot serve a
backtest window directly — it emits a single ``NewsItem`` representing "retail
attention as of now". The collector forward-collects these snapshots day by
day, building our own historical store over time. That is why Apewisdom
features are excluded from ``news_v0.yaml``'s first backtest (no history yet at
PR-C run-time); they enter a later iteration once months have accumulated.

**Timestamp semantics — honest now-stamp, never backdated.** The snapshot is
stamped at the true ``now`` and is emitted **only when ``now`` falls inside the
half-open window ``[start, end)``**. We deliberately do *not* clamp/backdate the
stamp to squeeze it into a window that already ended: backdating a signal is the
dangerous lookahead direction this whole project guards against (it would claim
retail attention was known earlier than it was).

Consequence for callers: **forward-collection must pass an ``end`` strictly
after now** (e.g. ``end`` = tomorrow). Both ``base.fetch`` and
``collector.collect`` filter on ``published_at < end`` (exclusive), so a
now-stamped item only survives when ``now < end``. A window ending at or before
now — e.g. ``collect(ticker, start, <today-midnight>)`` run during the day, or a
purely historical window — correctly returns nothing (Apewisdom has no history
to fill it). The pipeline's live forward-collection sets ``end_date`` to the
present-or-future accordingly.

Endpoint: ``GET /api/v1.0/filter/{filter}/page/{n}`` — ``filter`` defaults to
``all-stocks`` (surfaced as a config field); 100 results/page. We page until we
find the requested ticker or exhaust ``max_pages``.

HTTP-only — uses ``requests`` (a base dep), no optional extra.
"""

from __future__ import annotations

from datetime import datetime, timezone

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
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # The snapshot is "attention as of now". Emit it only when now is inside
        # the half-open window [start, end), and stamp it at the true now — never
        # backdate to fit a window that already closed (that would be lookahead).
        # So a window ending at/before now, or a purely historical/future window,
        # returns nothing. Forward-collection must use end > now (e.g. tomorrow).
        if not (start <= now < end):
            return []

        target = ticker.upper()
        row = self._find_ticker(target)
        if row is None:
            return []

        mentions = _as_int(row.get("mentions"))
        title = f"{target} retail attention: {mentions} mentions"
        return [
            NewsItem(
                ticker=ticker,
                published_at=now,
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
