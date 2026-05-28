"""Tests for the data-source layer.

PR-A only exercises the synthetic source against the network. The free
adapters are constructed (to catch import-time errors) and their fetch path
is *not* exercised — they need internet and (for Reddit) credentials. PR-B
will add adapter-specific integration tests with recorded fixtures.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from news_sentiment.datasources import REGISTRY, build_source
from news_sentiment.datasources.synthetic import SyntheticSource
from news_sentiment.datasources.tiingo import TiingoSource


def test_registry_lists_expected_sources() -> None:
    names = set(REGISTRY)
    assert {"synthetic", "edgar", "gdelt", "reddit", "google_trends", "tiingo"} <= names


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


def test_tiingo_stub_raises_until_pr_b() -> None:
    src = TiingoSource()
    with pytest.raises(NotImplementedError, match="PR-B"):
        src.fetch("AAPL", "2024-01-01", "2024-01-15")
