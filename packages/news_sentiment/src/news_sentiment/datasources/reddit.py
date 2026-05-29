"""Reddit (PRAW) adapter — **permanent stub.**

Reddit is **out** of the data stack. Reddit's Responsible Builder Policy
(updated 2026-05-18) requires explicit approval for API use: moderator access
is "solely for performing moderation actions", commercial use needs written
approval, and research access is gated to "academic researchers affiliated
with an accredited university". None of these fit a personal-research
market-signal collector, so PRAW is not a usable source for us.

What replaces it:
- **Live retail-attention** → ``apewisdom`` (free, no auth).
- **Historical WSB mentions** → Quiver Quantitative Hobbyist ($30/mo,
  licensed to redistribute, optional later).

Full reasoning: ``docs/research/data-sources.md``. The class is kept (rather
than deleted) so the registry can name it and surface the reason if a stale
config references it; it raises immediately on use.
"""

from __future__ import annotations

from datetime import datetime

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_REASON = (
    "Reddit is blocked by the Responsible Builder Policy; see "
    "docs/research/data-sources.md. Use 'apewisdom' for live retail-attention; "
    "Quiver Quant for historical WSB."
)


class RedditSource(BaseNewsSource):
    name = "reddit"

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        raise NotImplementedError(_REASON)
