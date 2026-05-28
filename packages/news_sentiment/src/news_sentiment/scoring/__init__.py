"""Scoring registry — name → scorer factory.

A scorer takes ``list[NewsItem]`` and returns ``list[ScoredItem]``. PR-A
ships the deterministic ``lexicon`` scorer; ``finbert`` and ``llm`` are stubs
with the cache + budget-cap scaffolding wired through, and they raise at
call time so a config typo here doesn't silently fall back.
"""

from __future__ import annotations

from collections.abc import Callable

from news_sentiment.scoring.finbert import FinBertScorer
from news_sentiment.scoring.lexicon import LexiconScorer
from news_sentiment.scoring.llm import LlmScorer

REGISTRY: dict[str, Callable[..., object]] = {
    "lexicon": LexiconScorer,
    "finbert": FinBertScorer,
    "llm": LlmScorer,
}


def build_scorer(name: str, **params) -> object:
    if name not in REGISTRY:
        raise KeyError(f"Unknown scorer {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
