"""A tiny embedded Loughran-McDonald-style word list.

The real LM dictionary has ~2,300 positive and ~2,300 negative finance-specific
terms. We embed a small subset here so PR-A's dev scorer is zero-dep and
deterministic. PR-B may replace this with a downloaded full dictionary or
the VADER package.

Each word is lowercased and unstemmed. The lexicon scorer counts whole-word
matches in title + body and emits sentiment = (pos - neg) / (pos + neg) when
either count is nonzero, else 0.
"""

from __future__ import annotations

POSITIVE: frozenset[str] = frozenset(
    {
        "beat",
        "beats",
        "exceed",
        "exceeds",
        "exceeded",
        "strong",
        "stronger",
        "record",
        "raises",
        "raised",
        "upgraded",
        "upgrade",
        "buy",
        "wins",
        "won",
        "win",
        "growth",
        "profit",
        "profitable",
        "outperform",
        "outperformed",
        "positive",
        "rally",
        "surge",
        "surged",
        "gains",
        "approved",
        "approval",
    }
)

NEGATIVE: frozenset[str] = frozenset(
    {
        "miss",
        "misses",
        "missed",
        "weak",
        "weaker",
        "loss",
        "losses",
        "cut",
        "cuts",
        "downgrade",
        "downgraded",
        "sell",
        "subpoena",
        "investigation",
        "lawsuit",
        "lawsuits",
        "fraud",
        "decline",
        "declines",
        "declined",
        "negative",
        "plunge",
        "plunged",
        "tumble",
        "tumbled",
        "warning",
        "warned",
        "bankrupt",
        "bankruptcy",
        "recall",
        "recalled",
    }
)
