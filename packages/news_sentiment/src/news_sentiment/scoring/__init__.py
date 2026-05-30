"""Scoring registry — name → scorer factory.

A scorer takes ``list[NewsItem]`` and returns ``list[ScoredItem]``. Four are
registered: ``lexicon`` (deterministic, dependency-free, the offline fallback),
``finbert`` (local model, ``[scoring]`` extra), ``llm`` (live Anthropic call
inside the cost-control harness, ``[llm]`` extra), and ``hybrid`` (FinBERT bulk
pass + LLM on the low-confidence tail). Heavy deps are lazy-imported, so naming
``finbert``/``llm``/``hybrid`` in a config only pulls the extra at call time.
"""

from __future__ import annotations

from collections.abc import Callable

from news_sentiment.scoring.finbert import FinBertScorer
from news_sentiment.scoring.hybrid import HybridScorer
from news_sentiment.scoring.lexicon import LexiconScorer
from news_sentiment.scoring.llm import LlmScorer

REGISTRY: dict[str, Callable[..., object]] = {
    "lexicon": LexiconScorer,
    "finbert": FinBertScorer,
    "llm": LlmScorer,
    "hybrid": HybridScorer,
}


def build_scorer(name: str, **params) -> object:
    if name not in REGISTRY:
        raise KeyError(f"Unknown scorer {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
