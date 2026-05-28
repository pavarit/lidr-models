"""LLM scorer — **interface + cache/budget scaffolding in PR-A; real call in PR-B.**

The LLM half of the FinBERT + LLM hybrid. Handles the tail of items FinBERT
isn't confident about, and adds relevance + entity-linking + event-type so
the hybrid output is richer than either component alone.

PR-A wires the three cost controls the plan calls out as **required**:

1. **Content-hash cache.** Every result is cached keyed by a content hash so
   the same headline is never paid for twice across runs. The cache is on
   disk (``cache/llm.jsonl``) so it survives process restarts.
2. **Per-run budget cap.** ``max_calls`` and ``max_usd`` are read from the
   pipeline config. When either is exceeded the scorer stops calling the LLM
   for the remainder of the run and falls back to lexicon (or whatever
   ``fallback_scorer`` is configured).
3. **Spend log.** Every LLM call appends a row to ``artifacts/llm_spend.csv``
   with ``run_id, model, prompt_tokens, completion_tokens, est_usd`` so
   cost-vs-quality is measurable across versions.

The actual LLM call (Anthropic SDK / response parsing) is the stubbed part
that PR-B will fill. The scaffolding above is real so PR-B can wire the
call into a known-good cost-control harness.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from news_sentiment.types import NewsItem, ScoredItem


class LlmBudgetExceeded(RuntimeError):
    """Raised when a single LLM call would push the run over its budget."""


class LlmScorer:
    name = "llm"

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_calls: int = 0,
        max_usd: float = 0.0,
        usd_per_1k_input: float = 0.0008,
        usd_per_1k_output: float = 0.004,
        cache_path: Path | None = None,
        spend_log_path: Path | None = None,
        run_id: str = "unknown",
        fallback_scorer_name: str = "lexicon",
    ) -> None:
        self.model = str(model)
        self.max_calls = int(max_calls)
        self.max_usd = float(max_usd)
        self.usd_per_1k_input = float(usd_per_1k_input)
        self.usd_per_1k_output = float(usd_per_1k_output)
        self.cache_path = Path(cache_path) if cache_path else None
        self.spend_log_path = Path(spend_log_path) if spend_log_path else None
        self.run_id = str(run_id)
        self.fallback_scorer_name = str(fallback_scorer_name)
        self._calls = 0
        self._spend_usd = 0.0
        self._cache: dict[str, dict] = self._load_cache()

    # -- cost-control plumbing (real in PR-A) -------------------------------

    @staticmethod
    def cache_key(item: NewsItem, model: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"|")
        h.update(item.content_hash.encode("utf-8"))
        return h.hexdigest()

    def has_cached(self, item: NewsItem) -> bool:
        return self.cache_key(item, self.model) in self._cache

    def budget_remaining(self) -> dict[str, float]:
        return {
            "calls_remaining": max(0, self.max_calls - self._calls),
            "usd_remaining": max(0.0, self.max_usd - self._spend_usd),
        }

    def _check_budget(self, est_usd: float) -> None:
        if self._calls >= self.max_calls:
            raise LlmBudgetExceeded(
                f"LLM call budget exhausted ({self.max_calls} calls). "
                "Increase scoring.llm.max_calls or rely on the fallback."
            )
        if self._spend_usd + est_usd > self.max_usd:
            raise LlmBudgetExceeded(
                f"LLM USD budget exhausted (max ${self.max_usd:.2f}, "
                f"would reach ${self._spend_usd + est_usd:.2f}). "
                "Increase scoring.llm.max_usd or rely on the fallback."
            )

    def _record_spend(self, prompt_tokens: int, completion_tokens: int) -> float:
        est_usd = (
            prompt_tokens * self.usd_per_1k_input / 1000.0
            + completion_tokens * self.usd_per_1k_output / 1000.0
        )
        self._spend_usd += est_usd
        self._calls += 1
        if self.spend_log_path is not None:
            self.spend_log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.spend_log_path.exists()
            with self.spend_log_path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(
                        [
                            "ts",
                            "run_id",
                            "model",
                            "prompt_tokens",
                            "completion_tokens",
                            "est_usd",
                        ]
                    )
                w.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        self.run_id,
                        self.model,
                        prompt_tokens,
                        completion_tokens,
                        f"{est_usd:.6f}",
                    ]
                )
        return est_usd

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        out: dict[str, dict] = {}
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = d.get("cache_key")
            if key:
                out[key] = d
        return out

    def _append_cache(self, key: str, payload: dict) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cache_key": key, **payload}, ensure_ascii=False))
            fh.write("\n")
        self._cache[key] = {"cache_key": key, **payload}

    # -- the actual call (stubbed in PR-A) ---------------------------------

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        out: list[ScoredItem] = []
        for it in items:
            key = self.cache_key(it, self.model)
            cached = self._cache.get(key)
            if cached is not None:
                out.append(
                    ScoredItem(
                        item=it,
                        sentiment=float(cached["sentiment"]),
                        relevance=float(cached.get("relevance", 1.0)),
                        confidence=float(cached.get("confidence", 1.0)),
                        scorer=self.name,
                    )
                )
                continue
            # Live call would happen here; PR-A stops short. Confirm an API
            # key is present so a missing key is reported now rather than
            # silently fired into a 401 in PR-B.
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "LlmScorer needs ANTHROPIC_API_KEY in env. "
                    "Cache-hit items would have been served from disk; "
                    f"this item ({it.title[:80]}) was a miss and needs a live call."
                )
            raise NotImplementedError(
                "LlmScorer live call lands in PR-B. Required next: "
                "(1) anthropic.Anthropic().messages.create with a prompt asking for "
                "    sentiment ∈ [-1,1], relevance ∈ [0,1], event_type; "
                "(2) self._check_budget(est_usd) BEFORE the call; "
                "(3) self._record_spend(prompt_tokens, completion_tokens) AFTER; "
                "(4) self._append_cache(key, {sentiment, relevance, confidence, event_type})."
            )
        return out
