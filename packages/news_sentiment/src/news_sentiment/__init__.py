"""news_sentiment — a competing model for lidr-models.

Phase-0 scaffolding (PR-A): pluggable free data adapters + collector +
deterministic lexicon scorer + three features + offline dev pipeline.

FinBERT + LLM scoring and the Tiingo News adapter are stubs; they land in PR-B.
A real ``news_v0.yaml`` backtest with comparison evidence lands in PR-C.
"""

from __future__ import annotations

__version__ = "0.1.0"
