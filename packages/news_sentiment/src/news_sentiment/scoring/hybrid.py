"""Hybrid scorer — FinBERT bulk pass + LLM on the low-confidence tail.

This is the scorer the plan's "FinBERT + LLM hybrid, with cost control" calls
for. The routing is the whole point: FinBERT scores every item locally and for
free; only the items FinBERT is *unsure* about (confidence below
``llm_confidence_threshold``, or relevance the LLM should adjudicate) escalate
to the paid LLM. Combined with the LLM scorer's own cache + budget cap + spend
log, this keeps LLM spend to the genuinely-ambiguous minority of items.

Config shape (selected by ``scorer.name: hybrid``)::

    scorer:
      name: hybrid
      llm_confidence_threshold: 0.6   # FinBERT below this → escalate to LLM
      finbert:                        # forwarded to FinBertScorer
        confidence_threshold: 0.6
      llm:                            # forwarded to LlmScorer
        model: claude-haiku-4-5
        max_calls: 200
        max_usd: 5.0

The pipeline injects the LLM cache / spend-log / run-id paths (the same ones it
injects for ``scorer.name: llm``); ``HybridScorer`` forwards them into its LLM
sub-scorer so the budget harness behaves identically whether the LLM is used
directly or via the hybrid.
"""

from __future__ import annotations

from news_sentiment.scoring.finbert import FinBertScorer
from news_sentiment.scoring.llm import LlmScorer
from news_sentiment.types import NewsItem, ScoredItem


class HybridScorer:
    name = "hybrid"

    def __init__(
        self,
        llm_confidence_threshold: float = 0.6,
        finbert: dict | None = None,
        llm: dict | None = None,
        # The pipeline injects these for budget plumbing; forward to the LLM.
        cache_path=None,
        spend_log_path=None,
        run_id: str = "unknown",
    ) -> None:
        self.llm_confidence_threshold = float(llm_confidence_threshold)
        self._finbert = FinBertScorer(**(finbert or {}))
        llm_kwargs = dict(llm or {})
        llm_kwargs.setdefault("cache_path", cache_path)
        llm_kwargs.setdefault("spend_log_path", spend_log_path)
        llm_kwargs.setdefault("run_id", run_id)
        self._llm = LlmScorer(**llm_kwargs)

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        if not items:
            return []
        base = self._finbert.score(items)
        # Items FinBERT is unsure about escalate to the LLM. Preserve order.
        escalate_idx = [
            i for i, s in enumerate(base) if s.confidence < self.llm_confidence_threshold
        ]
        if not escalate_idx:
            return base
        escalated = self._llm.score([base[i].item for i in escalate_idx])
        out = list(base)
        for slot, scored in zip(escalate_idx, escalated, strict=True):
            out[slot] = scored
        return out
