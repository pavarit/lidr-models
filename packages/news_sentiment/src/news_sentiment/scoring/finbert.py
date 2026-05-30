"""FinBERT scorer — local, free, bulk sentiment.

FinBERT (``ProsusAI/finbert``) is the local bulk scorer in the hybrid design:
it scores the bulk of items cheaply, and the LLM handles only the ambiguous
tail (low FinBERT confidence). The model is ~440MB and downloads once to
``~/.cache/huggingface`` on first use.

Output mapping per item:
- softmax over the model's 3 classes → ``(positive, negative, neutral)``
- ``sentiment = positive - negative`` ∈ ``[-1, 1]``
- ``confidence = max(softmax)`` ∈ ``[0, 1]`` — how peaked the distribution is;
  this is what the hybrid scorer thresholds on to decide LLM escalation.

``transformers`` + ``torch`` are the ``[scoring]`` extra and are **lazy-imported**
inside ``_ensure_model`` so the offline dev path (and CI) never need the 440MB
download. The model is loaded once per scorer instance and reused across
``score`` calls.
"""

from __future__ import annotations

from news_sentiment.types import NewsItem, ScoredItem


class FinBertScorer:
    name = "finbert"

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        confidence_threshold: float = 0.6,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.model_name = str(model_name)
        self.confidence_threshold = float(confidence_threshold)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._model = None
        self._tokenizer = None
        self._label_index: dict[str, int] | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "FinBertScorer requires the optional 'scoring' extra "
                "(transformers + torch, ~440MB on first model download). "
                "Install with: pip install -e ./packages/news_sentiment[scoring]"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        # ProsusAI/finbert labels are {0: positive, 1: negative, 2: neutral},
        # but read id2label rather than assume, so a relabelled checkpoint
        # still maps correctly.
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self._label_index = {v: k for k, v in id2label.items()}
        for required in ("positive", "negative", "neutral"):
            if required not in self._label_index:
                raise ValueError(
                    f"FinBERT model {self.model_name!r} is missing the "
                    f"{required!r} label; got {sorted(self._label_index)}."
                )

    def score(self, items: list[NewsItem]) -> list[ScoredItem]:
        if not items:
            return []
        self._ensure_model()
        import torch

        assert self._tokenizer is not None and self._model is not None
        assert self._label_index is not None
        pos_i = self._label_index["positive"]
        neg_i = self._label_index["negative"]

        out: list[ScoredItem] = []
        for batch_start in range(0, len(items), self.batch_size):
            batch = items[batch_start : batch_start + self.batch_size]
            texts = [self._text(it) for it in batch]
            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
            for it, row in zip(batch, probs, strict=True):
                pos = float(row[pos_i])
                neg = float(row[neg_i])
                sentiment = max(-1.0, min(1.0, pos - neg))
                confidence = float(row.max())
                out.append(
                    ScoredItem(
                        item=it,
                        sentiment=sentiment,
                        relevance=1.0,
                        confidence=confidence,
                        scorer=self.name,
                    )
                )
        return out

    @staticmethod
    def _text(item: NewsItem) -> str:
        body = item.body.strip()
        return f"{item.title}. {body}" if body else item.title
