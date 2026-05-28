"""FinBERT scorer — **stub, lands in PR-B.**

FinBERT (ProsusAI/finbert) is the local, free, bulk scorer in the hybrid
design. It will score the bulk of items; the LLM handles the ambiguous tail.

The interface, cache key, and ``confidence_threshold`` knob are wired now so
the pipeline config and the hybrid scorer can be written against the real
shape. Loading the model itself (~440MB download to ``~/.cache/huggingface``)
lands in PR-B.
"""

from __future__ import annotations

from news_sentiment.types import NewsItem, ScoredItem


class FinBertScorer:
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", confidence_threshold: float = 0.6) -> None:
        self.model_name = str(model_name)
        self.confidence_threshold = float(confidence_threshold)

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        raise NotImplementedError(
            "FinBertScorer lands in PR-B. Required next: "
            "(1) lazy-import transformers + torch; "
            "(2) load ProsusAI/finbert (cache locally); "
            "(3) batch-tokenize title + body, softmax → (pos, neu, neg); "
            "(4) emit sentiment = pos - neg, confidence = max softmax."
        )
