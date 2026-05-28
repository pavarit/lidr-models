"""Tiingo News API adapter — **stub, lands in PR-B.**

Tiingo News add-on (~$10/mo, "Power" tier) is the paid source that unlocks
~10 years of ticker-tagged history per the research doc — without it, the
real backtest has to wait months for the live collectors to accumulate data.

The class is registered now so configs can name it; calling ``fetch_raw``
without the PR-B implementation raises ``NotImplementedError`` with the
exact thing the next PR needs to land. **Point-in-time discipline is
mandatory** when this is wired: validate that Tiingo's ``publishedDate`` is
the true publish time and not a "as-of" backfill.
"""

from __future__ import annotations

from datetime import datetime

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem


class TiingoSource(BaseNewsSource):
    name = "tiingo"

    def __init__(self, tickers_alias: str | None = None, page_size: int = 1000) -> None:
        self.tickers_alias = tickers_alias
        self.page_size = int(page_size)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        raise NotImplementedError(
            "TiingoSource lands in PR-B. Required next: "
            "(1) read TIINGO_API_KEY from env; "
            "(2) GET https://api.tiingo.com/tiingo/news with tickers, startDate, endDate, limit; "
            "(3) parse publishedDate (validate it is true publish time, not as-of); "
            "(4) emit one NewsItem per article."
        )
