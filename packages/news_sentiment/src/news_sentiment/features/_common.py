"""Shared helpers for news features.

The single most important thing this module exists for is ``items_to_daily``:
convert a list of ``ScoredItem`` into a daily DataFrame indexed by date,
with one row per calendar day. Every feature uses this to roll item streams
up to daily granularity before aligning to the trading-day index.

**Point-in-time rule (mechanised here, asserted by the lookahead test).**
The feature value at trading day ``t`` may only see items with
``published_at < t.normalize()`` — i.e. items published strictly before the
start of trading day ``t``. We implement this by computing daily aggregates
keyed by ``floor("D")`` and then **shifting the aligned Series forward by one
trading day** at the alignment step. That way items from "today" inform
*tomorrow's* prediction, never today's.
"""

from __future__ import annotations

import pandas as pd

from news_sentiment.types import ScoredItem


def items_to_daily(items: list[ScoredItem]) -> pd.DataFrame:
    """Aggregate scored items into a daily DataFrame.

    Columns: ``count`` (n items that day), ``mean_sentiment`` (weighted by
    relevance), ``mean_relevance``. Indexed by the floor-of-day of
    ``published_at``. Days with no items are absent — callers reindex.
    """
    if not items:
        return pd.DataFrame(columns=["count", "mean_sentiment", "mean_relevance"])
    rows = [
        {
            "day": pd.Timestamp(it.published_at).normalize(),
            "sentiment": float(it.sentiment),
            "relevance": float(it.relevance),
        }
        for it in items
    ]
    df = pd.DataFrame(rows)
    # Sentiment is weighted by relevance — an item the scorer says is barely
    # about this ticker counts less. With lexicon-only scoring everything has
    # relevance 1.0 so this is a no-op; PR-B's LLM relevance makes it bite.
    df["w_sentiment"] = df["sentiment"] * df["relevance"]
    grp = df.groupby("day", sort=True)
    out = pd.DataFrame(
        {
            "count": grp.size(),
            "mean_sentiment": grp["w_sentiment"].sum() / grp["relevance"].sum().replace(0.0, 1.0),
            "mean_relevance": grp["relevance"].mean(),
        }
    )
    return out


def align_to_trading_days(daily: pd.Series, price_index: pd.DatetimeIndex) -> pd.Series:
    """Project a daily Series onto trading days with strict point-in-time discipline.

    Steps:
    1. Reindex ``daily`` to *every calendar day* in the relevant span
       (forward-fill not used — we want zeros, not last value).
    2. Reindex onto ``price_index`` (trading days only).
    3. **Shift forward by one trading day** so a value computed from items on
       day D is only visible at the next trading day's prediction time. This
       is what enforces "the value at index t depends only on items strictly
       before t."
    """
    if daily.empty:
        return pd.Series(0.0, index=price_index, name=daily.name)
    full = pd.date_range(daily.index.min(), price_index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0.0)
    aligned = daily.reindex(price_index, fill_value=0.0)
    # shift(1) creates one NaN at position 0; treat "nothing visible yet" as
    # 0.0 (the same value a no-news day produces) so the feature is usable
    # from the first trading row.
    return aligned.shift(1).fillna(0.0)
