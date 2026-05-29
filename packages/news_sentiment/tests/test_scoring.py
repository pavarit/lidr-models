"""Tests for the scorers.

- ``lexicon`` is deterministic and dependency-free.
- ``llm`` cost-control scaffolding (cache + budget cap + spend log) plus the
  PR-B live-call path, exercised with a mocked Anthropic client so no live
  call is made.
- ``finbert`` field mapping, exercised with a fake torch + injected model so
  CI needs neither transformers nor torch.
- ``hybrid`` routing — only low-confidence items escalate to the LLM.
"""

from __future__ import annotations

import builtins
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest
from news_sentiment.scoring.finbert import FinBertScorer
from news_sentiment.scoring.hybrid import HybridScorer
from news_sentiment.scoring.lexicon import LexiconScorer
from news_sentiment.scoring.llm import LlmBudgetExceeded, LlmScorer, _parse_llm_json
from news_sentiment.types import NewsItem, ScoredItem


def _item(title: str, body: str = "") -> NewsItem:
    return NewsItem(
        ticker="X",
        published_at=datetime(2024, 1, 1, 12, 0, 0),
        source="t",
        title=title,
        body=body,
    )


# --------------------------------------------------------------------------- #
# Lexicon                                                                      #
# --------------------------------------------------------------------------- #


def test_lexicon_positive_words_score_positive() -> None:
    scored = LexiconScorer().score([_item("X beats expectations, raises guidance")])
    assert scored[0].sentiment > 0
    assert scored[0].confidence > 0


def test_lexicon_negative_words_score_negative() -> None:
    scored = LexiconScorer().score([_item("X misses expectations, downgraded")])
    assert scored[0].sentiment < 0


def test_lexicon_zero_match_zero_confidence() -> None:
    scored = LexiconScorer().score([_item("X holds annual meeting next Tuesday")])
    assert scored[0].sentiment == 0.0
    assert scored[0].confidence == 0.0


# --------------------------------------------------------------------------- #
# LLM cost-control scaffolding                                                 #
# --------------------------------------------------------------------------- #


def test_llm_budget_cap_triggers(tmp_path: Path) -> None:
    scorer = LlmScorer(
        model="m",
        max_calls=0,
        max_usd=0.0,
        cache_path=tmp_path / "cache.jsonl",
        spend_log_path=tmp_path / "spend.csv",
        run_id="t",
    )
    with pytest.raises(LlmBudgetExceeded):
        scorer._check_budget(est_usd=0.001)


def test_llm_cache_hit_skips_budget(tmp_path: Path) -> None:
    """If a content_hash is already cached, no budget should be consumed."""
    cache = tmp_path / "cache.jsonl"
    scorer = LlmScorer(
        model="m",
        max_calls=10,
        max_usd=1.0,
        cache_path=cache,
        spend_log_path=tmp_path / "spend.csv",
        run_id="t",
    )
    item = _item("X beats")
    key = scorer.cache_key(item, "m")
    scorer._append_cache(key, {"sentiment": 0.5, "relevance": 1.0, "confidence": 0.9})
    out = scorer.score([item])
    assert len(out) == 1 and out[0].sentiment == 0.5
    assert scorer.budget_remaining()["calls_remaining"] == 10


# --------------------------------------------------------------------------- #
# LLM live call (mocked Anthropic client — no network)                        #
# --------------------------------------------------------------------------- #


class _FakeUsage:
    input_tokens = 120
    output_tokens = 25


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **_: object) -> _FakeResp:
        return _FakeResp(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def _llm(tmp_path: Path, **kw) -> LlmScorer:
    defaults = dict(
        model="m",
        max_calls=10,
        max_usd=1.0,
        cache_path=tmp_path / "cache.jsonl",
        spend_log_path=tmp_path / "spend.csv",
        run_id="t",
    )
    defaults.update(kw)
    return LlmScorer(**defaults)


def test_llm_live_call_parses_records_and_caches(tmp_path: Path, monkeypatch) -> None:
    scorer = _llm(tmp_path)
    monkeypatch.setattr(
        scorer,
        "_get_client",
        lambda: _FakeClient(
            '{"sentiment": 0.7, "relevance": 0.9, "confidence": 0.8, "event_type": "earnings"}'
        ),
    )
    item = _item("X beats expectations")
    out = scorer.score([item])

    assert len(out) == 1
    assert out[0].sentiment == 0.7
    assert out[0].relevance == 0.9
    assert out[0].scorer == "llm"
    # spend recorded (one call consumed)
    assert scorer.budget_remaining()["calls_remaining"] == 9
    # cached for next time, so a second run pays nothing
    assert scorer.has_cached(item)
    assert (tmp_path / "spend.csv").exists()


def test_llm_budget_exhaustion_falls_back_to_lexicon(tmp_path: Path) -> None:
    # Zero budget → the very first miss degrades to the lexicon, no raise.
    scorer = _llm(tmp_path, max_calls=0, max_usd=0.0)
    out = scorer.score([_item("X beats expectations, raises guidance")])
    assert len(out) == 1
    assert out[0].scorer == "lexicon"
    assert out[0].sentiment > 0  # lexicon picked up the positive words


def test_llm_malformed_response_falls_back(tmp_path: Path, monkeypatch) -> None:
    scorer = _llm(tmp_path)
    monkeypatch.setattr(scorer, "_get_client", lambda: _FakeClient("sorry, no JSON here"))
    out = scorer.score([_item("X misses expectations, downgraded")])
    assert out[0].scorer == "lexicon"  # graceful degradation, not a crash


def test_parse_llm_json_tolerates_prose_and_fences() -> None:
    parsed = _parse_llm_json('```json\n{"sentiment": -0.3, "event_type": "legal"}\n```')
    assert parsed["sentiment"] == -0.3
    with pytest.raises(ValueError):
        _parse_llm_json("absolutely no object")


# --------------------------------------------------------------------------- #
# FinBERT (fake torch + injected model — no transformers/torch needed)        #
# --------------------------------------------------------------------------- #


class _FakeRow:
    def __init__(self, vals: list[float]) -> None:
        self._vals = vals

    def __getitem__(self, i: int) -> float:
        return self._vals[i]

    def max(self) -> float:
        return max(self._vals)


class _FakeLogits:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(_FakeRow(r) for r in self.rows)


class _FakeModelOut:
    def __init__(self, rows: list[list[float]]) -> None:
        self.logits = _FakeLogits(rows)


class _FakeModel:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def __call__(self, **_: object) -> _FakeModelOut:
        return _FakeModelOut(self._rows)


class _FakeTokenizer:
    def __call__(self, texts, **_: object) -> dict:
        return {}


def _install_fake_torch(monkeypatch) -> None:
    fake = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake.no_grad = lambda: _NoGrad()
    # Treat the injected rows as already-softmaxed probabilities.
    fake.softmax = lambda x, dim=-1: x
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_finbert_score_maps_softmax_to_sentiment(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    scorer = FinBertScorer()
    # Bypass the 440MB load: inject model + tokenizer + label index directly.
    scorer._model = _FakeModel(rows=[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
    scorer._tokenizer = _FakeTokenizer()
    scorer._label_index = {"positive": 0, "negative": 1, "neutral": 2}

    out = scorer.score([_item("good"), _item("bad")])
    assert out[0].sentiment == pytest.approx(0.7)  # 0.8 - 0.1
    assert out[0].confidence == pytest.approx(0.8)
    assert out[0].scorer == "finbert"
    assert out[1].sentiment == pytest.approx(-0.6)  # 0.1 - 0.7
    assert out[1].confidence == pytest.approx(0.7)


def test_finbert_without_extra_raises_helpful_error(monkeypatch) -> None:
    # Deterministically simulate the [scoring] extra being absent by making
    # `import torch` raise, regardless of whether the dep is installed. (Setting
    # sys.modules["torch"]=None is NOT reliable under pytest's import hook — it
    # binds None instead of raising — so patch __import__ directly.)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.split(".")[0] == "transformers":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="scoring"):
        FinBertScorer().score([_item("anything")])


# --------------------------------------------------------------------------- #
# Hybrid routing                                                              #
# --------------------------------------------------------------------------- #


class _ConfScorer:
    """Returns a fixed confidence per item, tagged with a scorer name."""

    def __init__(self, name: str, confs: list[float]) -> None:
        self.name = name
        self._confs = confs

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        return [
            ScoredItem(item=it, sentiment=0.0, relevance=1.0, confidence=c, scorer=self.name)
            for it, c in zip(items, self._confs, strict=True)
        ]


class _LlmMarker:
    name = "llm"

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        return [
            ScoredItem(item=it, sentiment=1.0, relevance=1.0, confidence=1.0, scorer="llm")
            for it in items
        ]


def test_hybrid_escalates_only_low_confidence_items() -> None:
    h = HybridScorer(llm_confidence_threshold=0.6)
    h._finbert = _ConfScorer("finbert", [0.9, 0.3, 0.95])
    h._llm = _LlmMarker()
    out = h.score([_item("a"), _item("b"), _item("c")])
    assert [s.scorer for s in out] == ["finbert", "llm", "finbert"]


def test_hybrid_no_escalation_when_all_confident() -> None:
    h = HybridScorer(llm_confidence_threshold=0.6)
    h._finbert = _ConfScorer("finbert", [0.9, 0.8])
    h._llm = _LlmMarker()
    out = h.score([_item("a"), _item("b")])
    assert all(s.scorer == "finbert" for s in out)
