"""EODHD News adapter — the paid historical-news backbone ($19.99/mo).

EODHD News + Calendar (https://eodhd.com/financial-apis/stock-market-financial-news-api)
gives multi-year, date-filterable, ticker-tagged news with a per-article
sentiment block (polarity / neg / neu / pos). This is the source that makes a
*historical* backtest possible — the free sources (Finnhub 1yr, Apewisdom
live-only) can't reach far enough back.

Endpoint: ``GET /api/news?s={ticker}.US&from=YYYY-MM-DD&to=YYYY-MM-DD&api_token=...``

**Rate-limit math (respect the daily quota):** EODHD bills the News endpoint at
**5 API calls per request** (not 1). A backtest that pages through years of a
single ticker can exhaust a day's quota fast. The adapter therefore (a) uses a
large ``limit`` (default 1000, the max) to minimise page count, (b) caps total
pages at ``max_pages``, and (c) **never runs live in CI** — the integration
test uses a recorded fixture. ``offset``-based paging is used to walk windows
larger than one page.

**Point-in-time caveat (the #1 way sentiment backtests lie).** The cheap tier
does not *guarantee* strict point-in-time correctness — a timestamp could in
principle be backfilled/revised. We map the article ``date`` to
``published_at`` and treat EODHD's own ``sentiment`` block as a *validation
baseline only* (recorded in ``meta``), never as model ground truth — we still
score every item ourselves. PR-B records a manual timestamp spot-check in the
PR description; ``align_to_trading_days``' one-day shift is the enforcement.

HTTP-only — uses ``requests`` (a base dep), no optional extra.
"""

from __future__ import annotations

import os
from datetime import datetime

import requests

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_NEWS = "https://eodhd.com/api/news"


class EodhdSource(BaseNewsSource):
    name = "eodhd"

    def __init__(
        self,
        exchange_suffix: str = "US",
        limit: int = 1000,
        max_pages: int = 10,
        api_token: str | None = None,
    ) -> None:
        self.exchange_suffix = str(exchange_suffix)
        self.limit = int(limit)
        self.max_pages = int(max_pages)
        self._api_token = api_token

    def _token(self) -> str:
        token = self._api_token or os.environ.get("EODHD_API_TOKEN")
        if not token:
            raise RuntimeError(
                "EodhdSource needs EODHD_API_TOKEN in env. Subscribe to the "
                "Calendar & News add-on ($19.99/mo) at https://eodhd.com and "
                "copy the token from Settings → Secret API Token."
            )
        return token

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        symbol = f"{ticker}.{self.exchange_suffix}"
        token = self._token()
        items: list[NewsItem] = []
        for page in range(self.max_pages):
            params = {
                "s": symbol,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "limit": self.limit,
                "offset": page * self.limit,
                "api_token": token,
                "fmt": "json",
            }
            resp = requests.get(_NEWS, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            # EODHD signals auth / subscription failures with HTTP 200 and an
            # {"error": ...} body (raise_for_status won't catch it), so check
            # explicitly — otherwise the loop below would iterate the dict's
            # keys and crash with an opaque AttributeError.
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(
                    f"EODHD news API error: {payload['error']!r}. Check that "
                    "EODHD_API_TOKEN is correct and the Calendar & News add-on "
                    "($19.99/mo) is active on the account."
                )
            batch = payload or []
            if not batch:
                break
            for art in batch:
                published = _parse_eodhd_date(art.get("date"))
                if published is None:
                    continue
                title = str(art.get("title", "") or "")
                if not title:
                    continue
                items.append(
                    NewsItem(
                        ticker=ticker,
                        published_at=published,
                        source=self.name,
                        title=title,
                        body=str(art.get("content", "") or ""),
                        url=str(art.get("link", "") or ""),
                        meta={
                            # EODHD's own sentiment is a validation baseline,
                            # NOT model ground truth — we score items ourselves.
                            "eodhd_sentiment": art.get("sentiment"),
                            "symbols": art.get("symbols"),
                            "tags": art.get("tags"),
                        },
                    )
                )
            if len(batch) < self.limit:
                break
        return items


def _parse_eodhd_date(raw: object) -> datetime | None:
    """EODHD ``date`` is ISO 8601, often with a timezone offset.

    Normalise to a UTC-naive datetime to satisfy the point-in-time contract
    (``NewsItem.published_at`` is UTC-naive across all adapters).
    """
    if not raw:
        return None
    s = str(raw)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(tz=None).replace(tzinfo=None)
    return dt
