"""Reddit (PRAW) adapter.

Live-only — per the research doc, Pushshift historical access went away in
2023, so this adapter starts the clock at install time. Every day of delay
is lost training data; that's why Phase 0 is "start collecting now."

Requires credentials in env (``REDDIT_CLIENT_ID``, ``REDDIT_CLIENT_SECRET``,
``REDDIT_USER_AGENT``). Without them the adapter raises a helpful error;
PRA-only smoke tests use the synthetic source instead.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


class RedditSource(BaseNewsSource):
    name = "reddit"

    def __init__(
        self,
        subreddits: tuple[str, ...] = _DEFAULT_SUBREDDITS,
        per_subreddit_limit: int = 200,
    ) -> None:
        self.subreddits = tuple(subreddits)
        self.per_subreddit_limit = int(per_subreddit_limit)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        try:
            import praw  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RedditSource requires the optional 'reddit' extra. "
                "Install with: pip install -e ./packages/news_sentiment[reddit]"
            ) from exc

        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        user_agent = os.environ.get("REDDIT_USER_AGENT", "lidr-models/0.1 research")
        if not client_id or not client_secret:
            raise RuntimeError(
                "RedditSource needs REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET "
                "in env. Register an app at https://www.reddit.com/prefs/apps."
            )

        import praw as _praw

        reddit = _praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        items: list[NewsItem] = []
        query = f"${ticker} OR {ticker}"
        for sub in self.subreddits:
            for post in reddit.subreddit(sub).search(
                query, sort="new", time_filter="all", limit=self.per_subreddit_limit
            ):
                published = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).replace(
                    tzinfo=None
                )
                if not (start <= published < end):
                    continue
                items.append(
                    NewsItem(
                        ticker=ticker,
                        published_at=published,
                        source=self.name,
                        title=str(post.title or ""),
                        body=str(getattr(post, "selftext", "") or ""),
                        url=f"https://reddit.com{post.permalink}",
                        meta={
                            "subreddit": sub,
                            "score": int(getattr(post, "score", 0) or 0),
                            "num_comments": int(getattr(post, "num_comments", 0) or 0),
                        },
                    )
                )
        return items
