"""Google Trends (pytrends) attention-proxy adapter.

Google Trends gives a relative search interest number per day for a query.
Unlike the other adapters, the unit here is not "an article was published";
it's "this much attention was paid on this day." We emit one synthetic
``NewsItem`` per day with the interest value in ``meta['interest']`` and an
empty body, so downstream features can pick it up uniformly.

pytrends is an unofficial client and rate-limits; lazy-imported.
"""

from __future__ import annotations

from datetime import datetime, time

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem


class GoogleTrendsSource(BaseNewsSource):
    name = "google_trends"

    def __init__(self, geo: str = "US", timeframe_hint: str = "all") -> None:
        self.geo = str(geo)
        self.timeframe_hint = str(timeframe_hint)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        try:
            from pytrends.request import TrendReq
        except ImportError as exc:
            raise ImportError(
                "GoogleTrendsSource requires the optional 'trends' extra. "
                "Install with: pip install -e ./packages/news_sentiment[trends]"
            ) from exc

        timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload([ticker], timeframe=timeframe, geo=self.geo)
        df = pytrends.interest_over_time()
        items: list[NewsItem] = []
        if df is None or df.empty:
            return items
        for ts, row in df.iterrows():
            interest = float(row.get(ticker, 0) or 0)
            published = datetime.combine(ts.date(), time(23, 59, 59))
            items.append(
                NewsItem(
                    ticker=ticker,
                    published_at=published,
                    source=self.name,
                    title=f"{ticker} search interest {interest:.0f}",
                    body="",
                    url="",
                    meta={"interest": interest, "geo": self.geo},
                )
            )
        return items
