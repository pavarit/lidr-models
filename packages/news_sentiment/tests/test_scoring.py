"""Tests for the lexicon scorer + the LLM scorer's cost-control scaffolding.

The lexicon scorer ships in PR-A and must be deterministic on a fixed input.
The LLM scorer's actual call lands in PR-B, but the cache + budget cap +
spend log scaffolding is real PR-A code and must hold its invariants today.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from news_sentiment.scoring.lexicon import LexiconScorer
from news_sentiment.scoring.llm import LlmBudgetExceeded, LlmScorer
from news_sentiment.types import NewsItem


def _item(title: str, body: str = "") -> NewsItem:
    return NewsItem(
        ticker="X",
        published_at=datetime(2024, 1, 1, 12, 0, 0),
        source="t",
        title=title,
        body=body,
    )


def test_lexicon_positive_words_score_positive() -> None:
    scored = LexiconScorer().score([_item("X beats expectations, raises guidance")])
    assert scored[0].sentiment > 0
    assert scored[0].confidence > 0


def test_lexicon_negative_words_score_negative() -> None:
    scored = LexiconScorer().score([_item("X misses expectations, downgraded")])
    assert scored[0].sentiment < 0


def test_lexicon_zero_match_zero_confidence() -> None:
    scored = LexiconScorer().score([_item("X holds annual meeting next Tuesday")])
    assert scored[0].sentiment == 0.0
    assert scored[0].confidence == 0.0


def test_llm_budget_cap_triggers(tmp_path: Path) -> None:
    scorer = LlmScorer(
        model="m",
        max_calls=0,
        max_usd=0.0,
        cache_path=tmp_path / "cache.jsonl",
        spend_log_path=tmp_path / "spend.csv",
        run_id="t",
    )
    with pytest.raises(LlmBudgetExceeded):
        scorer._check_budget(est_usd=0.001)


def test_llm_cache_hit_skips_budget(tmp_path: Path) -> None:
    """If a content_hash is already cached, no budget should be consumed."""
    cache = tmp_path / "cache.jsonl"
    scorer = LlmScorer(
        model="m",
        max_calls=10,
        max_usd=1.0,
        cache_path=cache,
        spend_log_path=tmp_path / "spend.csv",
        run_id="t",
    )
    item = _item("X beats")
    key = scorer.cache_key(item, "m")
    scorer._append_cache(key, {"sentiment": 0.5, "relevance": 1.0, "confidence": 0.9})
    out = scorer.score([item])
    assert len(out) == 1 and out[0].sentiment == 0.5
    assert scorer.budget_remaining()["calls_remaining"] == 10
