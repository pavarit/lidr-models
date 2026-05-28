"""Collector tests — dedup, cache round-trip, point-in-time persistence."""

from __future__ import annotations

from pathlib import Path

from news_sentiment.datasources.synthetic import SyntheticSource
from news_sentiment.ingest.collector import collect


def test_collector_dedupes_across_runs(tmp_path: Path) -> None:
    """A second collect over the same window must not duplicate items."""
    src = SyntheticSource(items_per_day_mean=2.0, seed=1)
    first = collect([src], "X", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    second = collect([src], "X", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    assert {it.content_hash for it in first} == {it.content_hash for it in second}
    assert len(first) == len(second)


def test_collector_persists_publish_timestamps(tmp_path: Path) -> None:
    """Items round-trip through the cache with their true publish timestamps."""
    src = SyntheticSource(items_per_day_mean=1.0, seed=2)
    first = collect([src], "X", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    cache_file = tmp_path / "X__synthetic.jsonl"
    assert cache_file.exists()
    # Re-read via a new collector run — timestamps must be preserved exactly.
    second = collect([src], "X", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    by_hash_first = {it.content_hash: it for it in first}
    for it in second:
        assert it.published_at == by_hash_first[it.content_hash].published_at
