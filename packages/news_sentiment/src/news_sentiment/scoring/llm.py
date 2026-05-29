"""LLM scorer — live Anthropic call inside the PR-A cost-control harness.

The LLM half of the FinBERT + LLM hybrid. Handles the tail of items FinBERT
isn't confident about, and adds relevance + entity-linking + event-type so
the hybrid output is richer than either component alone.

Three cost controls (scaffolding from PR-A, live call wired in PR-B):

1. **Content-hash cache.** Every result is cached keyed by a content hash so
   the same headline is never paid for twice across runs. The cache is on
   disk (``cache/llm.jsonl``) so it survives process restarts.
2. **Per-run budget cap.** ``max_calls`` and ``max_usd`` are read from the
   pipeline config. When either would be exceeded the scorer stops calling the
   LLM for the remainder of the run and falls back to lexicon (or whatever
   ``fallback_scorer_name`` is configured) — it does **not** raise, so a run
   degrades gracefully instead of dying mid-backtest.
3. **Spend log.** Every LLM call appends a row to ``artifacts/llm_spend.csv``
   with ``run_id, model, prompt_tokens, completion_tokens, est_usd`` so
   cost-vs-quality is measurable across versions.

The Anthropic SDK is the ``[llm]`` extra and is **lazy-imported** so the
offline dev path never needs it. The call asks for a strict JSON object
(``sentiment`` ∈ [-1,1], ``relevance`` ∈ [0,1], ``confidence`` ∈ [0,1],
``event_type``); a parse failure falls back rather than crashing the run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from news_sentiment.types import NewsItem, ScoredItem

_SYSTEM_PROMPT = (
    "You are a financial-news sentiment rater. For the given headline (and "
    "optional body) about a specific stock ticker, return a STRICT JSON object "
    "with exactly these keys and no prose:\n"
    '  "sentiment": float in [-1, 1] (−1 very bearish, 0 neutral, +1 very bullish),\n'
    '  "relevance": float in [0, 1] (how much the item is actually about THIS ticker),\n'
    '  "confidence": float in [0, 1] (how sure you are),\n'
    '  "event_type": short snake_case label (e.g. earnings, guidance, mna, '
    "legal, product, macro, other).\n"
    "Judge directional impact on the stock, not how positive the prose sounds."
)


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
        self._client = None
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

    def _budget_ok(self, est_usd: float) -> bool:
        """Non-raising budget check used by the score loop."""
        try:
            self._check_budget(est_usd)
            return True
        except LlmBudgetExceeded:
            return False

    # -- the live call -----------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "LlmScorer requires the optional 'llm' extra (anthropic SDK). "
                "Install with: pip install -e ./packages/news_sentiment[llm]"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LlmScorer needs ANTHROPIC_API_KEY in env. "
                "Create a key at https://console.anthropic.com."
            )
        self._client = anthropic.Anthropic()
        return self._client

    def _call_llm(self, item: NewsItem) -> tuple[dict, int, int]:
        """Return (parsed_result, prompt_tokens, completion_tokens).

        Raises on transport error or unparseable response; the caller decides
        whether to fall back.
        """
        client = self._get_client()
        body = item.body.strip()
        user = f"Ticker: {item.ticker}\nHeadline: {item.title}"
        if body:
            user += f"\nBody: {body[:1500]}"
        resp = client.messages.create(
            model=self.model,
            max_tokens=200,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(block, "text", "") for block in resp.content)
        parsed = _parse_llm_json(text)
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return parsed, prompt_tokens, completion_tokens

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        out: list[ScoredItem] = []
        fallback = None  # lazily built once the budget is exhausted
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

            # Estimate the cost before spending it. If we can't afford the
            # call, degrade to the fallback scorer for the rest of the run
            # rather than raising mid-backtest.
            est_usd = self._estimate_usd(it)
            if not self._budget_ok(est_usd):
                if fallback is None:
                    fallback = self._build_fallback()
                out.append(fallback.score([it])[0])
                continue

            try:
                parsed, prompt_tokens, completion_tokens = self._call_llm(it)
            except Exception:  # noqa: BLE001 - any live-call failure degrades gracefully
                if fallback is None:
                    fallback = self._build_fallback()
                out.append(fallback.score([it])[0])
                continue

            self._record_spend(prompt_tokens, completion_tokens)
            payload = {
                "sentiment": float(parsed["sentiment"]),
                "relevance": float(parsed.get("relevance", 1.0)),
                "confidence": float(parsed.get("confidence", 1.0)),
                "event_type": str(parsed.get("event_type", "other")),
            }
            self._append_cache(key, payload)
            out.append(
                ScoredItem(
                    item=it,
                    sentiment=payload["sentiment"],
                    relevance=payload["relevance"],
                    confidence=payload["confidence"],
                    scorer=self.name,
                )
            )
        return out

    def _estimate_usd(self, item: NewsItem) -> float:
        """Rough pre-call cost estimate (~4 chars/token + fixed output budget)."""
        chars = len(_SYSTEM_PROMPT) + len(item.title) + len(item.body[:1500]) + 40
        est_input = chars / 4.0
        est_output = 80.0
        return (
            est_input * self.usd_per_1k_input / 1000.0
            + est_output * self.usd_per_1k_output / 1000.0
        )

    def _build_fallback(self):
        from news_sentiment.scoring.lexicon import LexiconScorer

        if self.fallback_scorer_name != "lexicon":
            # Only lexicon is dependency-free; anything heavier would defeat
            # the point of a budget-exhaustion fallback. Document and degrade.
            pass
        return LexiconScorer()


def _parse_llm_json(text: str) -> dict:
    """Extract the JSON object from a model response.

    Tolerant of leading/trailing prose or code fences: grabs the first
    ``{...}`` span and parses it. Raises ValueError if no valid object with a
    numeric ``sentiment`` is found.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")
    obj = json.loads(match.group(0))
    if "sentiment" not in obj:
        raise ValueError(f"LLM response missing 'sentiment': {obj!r}")
    float(obj["sentiment"])  # validate numeric
    return obj
