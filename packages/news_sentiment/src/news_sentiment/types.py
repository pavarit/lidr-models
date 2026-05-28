"""Shared item / scored-item dataclasses.

Every data adapter returns ``list[NewsItem]``. Scorers consume ``NewsItem``s
and return ``list[ScoredItem]``. Features consume scored-item streams and
return daily ``pd.Series`` aligned to the price index.

``published_at`` is the **true publish timestamp**, UTC-naive. This is the
single most important field for point-in-time discipline: features at date
``t`` may only use items with ``published_at < t`` (where ``t`` is taken at
the open of the trading day). The lookahead test mechanises this rule.

``content_hash`` is deterministic over ``(source, url, title, body)`` so the
collector can dedup across overlapping source pulls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class NewsItem:
    """A single piece of content (headline, filing excerpt, post, search-trend bucket).

    Lookahead-critical fields are ``published_at`` (true publish time) and
    ``ticker`` (the entity the item is attributed to). All other fields are
    advisory.
    """

    ticker: str
    published_at: datetime
    source: str
    title: str
    body: str = ""
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        h = hashlib.sha1()
        h.update(self.source.encode("utf-8"))
        h.update(b"|")
        h.update(self.url.encode("utf-8"))
        h.update(b"|")
        h.update(self.title.encode("utf-8"))
        h.update(b"|")
        h.update(self.body.encode("utf-8"))
        return h.hexdigest()


@dataclass(frozen=True)
class ScoredItem:
    """A NewsItem plus scorer output.

    ``sentiment`` is in ``[-1, +1]`` (negative=bearish, positive=bullish).
    ``relevance`` is in ``[0, 1]`` (how much this item is actually about the
    ticker — important once LLM-side entity linking lands). ``confidence`` is
    in ``[0, 1]`` and reports how sure the scorer was (low confidence routes
    to the LLM in the hybrid scorer; PR-B).
    """

    item: NewsItem
    sentiment: float
    relevance: float = 1.0
    confidence: float = 1.0
    scorer: str = ""

    @property
    def published_at(self) -> datetime:
        return self.item.published_at

    @property
    def ticker(self) -> str:
        return self.item.ticker
