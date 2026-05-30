"""news_sentiment — a competing model for lidr-models.

Pluggable data adapters (edgar / gdelt / finnhub / apewisdom / eodhd / hn, plus
the synthetic offline source; reddit + google_trends are permanent stubs) +
collector + scorers (lexicon / finbert / llm / hybrid) + three features +
offline dev pipeline.

As of PR-B, FinBERT + live-LLM scoring are wired and the Tiingo adapter has
been removed in the data-source rewire. A real ``news_v0.yaml`` backtest with
comparison evidence lands in PR-C.
"""

from __future__ import annotations

__version__ = "0.1.0"
