"""Lexicon scorer — Loughran-McDonald-style word counts.

The deterministic fallback scorer. Pure-Python, no deps. Used by ``dev.yaml``
and by the hybrid scorer (PR-B) as the cheap first pass before FinBERT / LLM.

Sentiment is computed per item as ``(pos - neg) / (pos + neg)`` clipped to
``[-1, +1]``, where ``pos`` and ``neg`` are whole-word matches of the
embedded LM word list in ``title + " " + body``. Items with zero matches
score 0 with confidence 0 — useful so the hybrid scorer can route them to a
heavier scorer instead of trusting noise.
"""

from __future__ import annotations

import re

from news_sentiment.scoring._lm_wordlist import NEGATIVE, POSITIVE
from news_sentiment.types import NewsItem, ScoredItem

_WORD_RE = re.compile(r"[a-zA-Z']+")


class LexiconScorer:
    name = "lexicon"

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        return [self._score_one(it) for it in items]

    def _score_one(self, item: NewsItem) -> ScoredItem:
        text = f"{item.title} {item.body}".lower()
        tokens = _WORD_RE.findall(text)
        pos = sum(1 for tok in tokens if tok in POSITIVE)
        neg = sum(1 for tok in tokens if tok in NEGATIVE)
        total = pos + neg
        if total == 0:
            return ScoredItem(item=item, sentiment=0.0, relevance=1.0, confidence=0.0, scorer=self.name)
        sentiment = (pos - neg) / total
        # Confidence rises with sample size (more hits = sturdier signal),
        # asymptoting to 1.0. Calibration is not the point here — PR-B's
        # LLM step is what will get serious about confidence.
        confidence = 1.0 - 1.0 / (1.0 + total)
        return ScoredItem(
            item=item,
            sentiment=float(sentiment),
            relevance=1.0,
            confidence=float(confidence),
            scorer=self.name,
        )
