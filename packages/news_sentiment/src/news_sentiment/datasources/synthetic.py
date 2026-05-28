"""Synthetic news source — deterministic offline-safe items.

Generates a configurable number of items per business day, with publish
timestamps spread across the day and sentiment-bearing text drawn from a tiny
fixed lexicon. Used by ``dev.yaml`` so the full pipeline (collector → scorer
→ features → backtest → artifact) can be exercised without internet.

Important: the timestamps emitted here are real Python datetimes that satisfy
the point-in-time contract — features at date ``t`` will only see items with
``published_at < t``. The lookahead test runs against the synthetic stream
for that reason.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_POSITIVE = [
    "beats expectations",
    "raises guidance",
    "strong quarter",
    "record revenue",
    "wins contract",
    "upgraded to buy",
]
_NEGATIVE = [
    "misses expectations",
    "cuts guidance",
    "weak quarter",
    "loss widens",
    "downgraded to sell",
    "subpoena disclosed",
]
_NEUTRAL = [
    "files quarterly report",
    "appoints board member",
    "announces conference",
    "completes refinancing",
]


class SyntheticSource(BaseNewsSource):
    name = "synthetic"

    def __init__(
        self,
        items_per_day_mean: float = 2.5,
        positive_share: float = 0.4,
        negative_share: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.items_per_day_mean = float(items_per_day_mean)
        self.positive_share = float(positive_share)
        self.negative_share = float(negative_share)
        self.seed = int(seed)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        # Deterministic per (ticker, window) so repeated calls return the same items.
        seed_material = (ticker, start.toordinal(), end.toordinal(), self.seed)
        rng = np.random.default_rng(abs(hash(seed_material)) % (2**32))

        items: list[NewsItem] = []
        day = start.date()
        end_date = end.date()
        idx = 0
        while day < end_date:
            # Only generate items on weekdays; that's where real news clusters too.
            if day.weekday() < 5:
                n = int(rng.poisson(self.items_per_day_mean))
                for _ in range(n):
                    hour = int(rng.integers(0, 24))
                    minute = int(rng.integers(0, 60))
                    second = int(rng.integers(0, 60))
                    ts = datetime.combine(day, time(hour, minute, second))
                    bucket = rng.random()
                    if bucket < self.positive_share:
                        phrase = _POSITIVE[int(rng.integers(0, len(_POSITIVE)))]
                    elif bucket < self.positive_share + self.negative_share:
                        phrase = _NEGATIVE[int(rng.integers(0, len(_NEGATIVE)))]
                    else:
                        phrase = _NEUTRAL[int(rng.integers(0, len(_NEUTRAL)))]
                    title = f"{ticker} {phrase}"
                    items.append(
                        NewsItem(
                            ticker=ticker,
                            published_at=ts,
                            source=self.name,
                            title=title,
                            body="",
                            url=f"synthetic://{ticker}/{idx}",
                            meta={"synthetic": True},
                        )
                    )
                    idx += 1
            day = day + timedelta(days=1)
        return items
