"""Tests for the data-source layer.

The synthetic source is exercised end-to-end here; the live HTTP adapters get
recorded-fixture integration tests in ``test_datasources_integration.py`` (no
live quota burned). This file covers the registry shape and the two permanent
stubs (``reddit``, ``google_trends``) that must raise with a real reason.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from news_sentiment.datasources import REGISTRY, build_source
from news_sentiment.datasources.google_trends import GoogleTrendsSource
from news_sentiment.datasources.reddit import RedditSource
from news_sentiment.datasources.synthetic import SyntheticSource


def test_registry_lists_expected_sources() -> None:
    names = set(REGISTRY)
    expected = {
        "synthetic",
        "edgar",
        "gdelt",
        "finnhub",
        "apewisdom",
        "eodhd",
        "hn",
        "reddit",
        "google_trends",
    }
    assert expected <= names


def test_tiingo_is_unregistered() -> None:
    # Tiingo was deleted in the 2026-05-28 data-source rewire.
    assert "tiingo" not in REGISTRY
    with pytest.raises(KeyError):
        build_source("tiingo")


def test_synthetic_source_is_deterministic_and_windowed() -> None:
    a = SyntheticSource(items_per_day_mean=1.0, seed=7)
    b = SyntheticSource(items_per_day_mean=1.0, seed=7)
    win_a = a.fetch("X", "2024-01-01", "2024-02-01")
    win_b = b.fetch("X", "2024-01-01", "2024-02-01")
    assert [it.content_hash for it in win_a] == [it.content_hash for it in win_b]
    for it in win_a:
        assert datetime(2024, 1, 1) <= it.published_at < datetime(2024, 2, 1)


def test_synthetic_source_emits_items_only_on_weekdays() -> None:
    src = SyntheticSource(items_per_day_mean=5.0, seed=0)
    items = src.fetch("X", "2024-01-01", "2024-01-31")
    assert items, "expected non-empty"
    for it in items:
        assert it.published_at.weekday() < 5


def test_build_source_unknown_raises() -> None:
    with pytest.raises(KeyError):
        build_source("not_a_real_source")


def test_reddit_is_permanent_stub() -> None:
    with pytest.raises(NotImplementedError, match="Responsible Builder Policy"):
        RedditSource().fetch("AAPL", "2024-01-01", "2024-01-15")


def test_google_trends_is_permanent_stub() -> None:
    with pytest.raises(NotImplementedError, match="pytrends"):
        GoogleTrendsSource().fetch("AAPL", "2024-01-01", "2024-01-15")
