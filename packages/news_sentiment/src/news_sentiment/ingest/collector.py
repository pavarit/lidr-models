"""Collector — fans out to every configured DataSource, dedups, persists.

The collector is what makes the multi-source "Phase 0: start the clock now"
posture work. It pulls items per (ticker, source) window, dedups by
content_hash, and writes raw items to disk so the data clock advances
day-by-day independent of any future model code change.

Cache layout (one file per ticker, jsonl, append-only-ish):

    <cache_dir>/<ticker>__<source>.jsonl

Each line is a serialized NewsItem with the **true publish timestamp**, not
the ingest time. The collector treats the cache as the ground truth: on
re-run, items already in the cache are kept; items returned by the source
that aren't in the cache get appended. This is what gives the system its
point-in-time discipline at the ingestion layer — once an item is recorded
with a publish time, future re-runs can't quietly revise it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem


def collect(
    sources: list[BaseNewsSource],
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
) -> list[NewsItem]:
    """Pull items for ``ticker`` over ``[start, end)`` from every source.

    Dedups across sources by ``content_hash``. If ``cache_dir`` is provided,
    union the fresh pull with whatever was already cached and persist the
    result. Items are returned sorted by ``published_at``.
    """
    seen: dict[str, NewsItem] = {}

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            for it in _read_cache(cache_dir, ticker, src.name):
                seen[it.content_hash] = it

    for src in sources:
        fresh = src.fetch(ticker, start, end)
        new_items: list[NewsItem] = []
        for it in fresh:
            if it.content_hash in seen:
                continue
            seen[it.content_hash] = it
            new_items.append(it)
        if cache_dir is not None and new_items:
            _append_cache(cache_dir, ticker, src.name, new_items)

    s = _parse_dt(start)
    e = _parse_dt(end)
    items = [it for it in seen.values() if s <= it.published_at < e]
    items.sort(key=lambda it: it.published_at)
    return items


# --------------------------------------------------------------------------- #
# Disk cache (JSONL — one item per line)                                      #
# --------------------------------------------------------------------------- #

def _cache_path(cache_dir: Path, ticker: str, source: str) -> Path:
    return cache_dir / f"{ticker}__{source}.jsonl"


def _read_cache(cache_dir: Path, ticker: str, source: str) -> list[NewsItem]:
    path = _cache_path(cache_dir, ticker, source)
    if not path.exists():
        return []
    out: list[NewsItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        d["published_at"] = datetime.fromisoformat(d["published_at"])
        out.append(NewsItem(**d))
    return out


def _append_cache(cache_dir: Path, ticker: str, source: str, items: list[NewsItem]) -> None:
    path = _cache_path(cache_dir, ticker, source)
    with path.open("a", encoding="utf-8") as fh:
        for it in items:
            d = asdict(it)
            d["published_at"] = it.published_at.isoformat()
            fh.write(json.dumps(d, ensure_ascii=False))
            fh.write("\n")


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s) if "T" in s else datetime.strptime(s, "%Y-%m-%d")
