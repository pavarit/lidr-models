# news_sentiment

A competing model for `lidr-models`: turn recent news and internet discussion
about high-news individual stocks into a BUY / HOLD / SELL recommendation,
scored by the same `lidr_core` harness as `ta_ensemble` so the two compete
apples-to-apples.

**Hypothesis.** Synthesis / attention edge — recognizing shifts in aggregate
sentiment, attention spikes, and post-event drift. **Not** a latency edge; we
are always late to a single headline by construction.

## Status (PR-B — data-source rewire + FinBERT/LLM scoring)

- Package layout, `DataSource` and `Feature` registries wired against
  `lidr_core` protocols.
- **Data adapters** (all real, HTTP-only via `requests` — no per-adapter extras):
  - `synthetic` — offline deterministic items for the dev smoke path.
  - `edgar`, `gdelt` — free, no key (PR-A).
  - `finnhub` — free personal-use tier, 1yr US company news (`FINNHUB_API_KEY`).
  - `apewisdom` — free, no auth, **live-snapshot** retail-attention; emits a
    per-day snapshot the collector forward-collects into history.
  - `eodhd` — paid ($19.99/mo) historical news + per-article sentiment
    (`EODHD_API_TOKEN`); each request bills **5 API calls** — respect the quota.
  - `hn` — free Hacker News (Algolia), tech-skewed supplement.
  - `reddit`, `google_trends` — **permanent stubs** that raise with the reason
    (Reddit blocked by the Responsible Builder Policy; pytrends archived
    2025-04-17). Kept registered so a stale config gets the reason, not a
    `KeyError`. `tiingo` is **deleted** ($30/mo for only 3mo history).
    Rationale: [`docs/research/data-sources.md`](../../docs/research/data-sources.md).
- Collector with timestamped on-disk cache + dedup-by-content-hash so the data
  clock starts now (live sources like Apewisdom can't be backfilled — every day
  of delay is lost training data).
- **Scoring** (heavy deps lazy-imported; offline path needs none):
  - `lexicon` — deterministic Loughran-McDonald word counts, dependency-free.
  - `finbert` — local `ProsusAI/finbert` (`[scoring]` extra, ~440MB on first
    use); `sentiment = pos − neg`, `confidence = max softmax`.
  - `llm` — live Anthropic call (`[llm]` extra, `ANTHROPIC_API_KEY`) inside the
    cache + per-run budget cap + spend-log harness; degrades to `lexicon` on
    budget exhaustion or a malformed response instead of crashing.
  - `hybrid` — FinBERT bulk pass, escalating only low-confidence items to the
    LLM (the cost-controlled FinBERT + LLM design).
- Three features: `sentiment_level`, `sentiment_momentum`, `abnormal_mention_volume`.
- `configs/dev.yaml` runs the full pipeline on synthetic prices + synthetic
  items end-to-end (no internet, no keys).

PR-B intentionally does **not** ship a real `news_v0.yaml` backtest or a
news-vs-TA comparison — that's PR-C (real backtest + comparison). Apewisdom
features are excluded from PR-C's first config because the forward-collected
store has no history yet.

## CLI

```bash
make install                                                   # installs all three packages editable
python -m news_sentiment list-features                         # discover registered features
python -m news_sentiment list-sources                          # discover registered data adapters
python -m news_sentiment backtest packages/news_sentiment/configs/dev.yaml
```

## What the data looks like

A picture of the pipeline by example, top to bottom. Every shape below is real
PR-A code; the synthetic source produces these exact structures and the
lookahead test asserts against them.

### 1. A raw item out of any DataSource

Every adapter (`synthetic`, `edgar`, `gdelt`, `finnhub`, `apewisdom`, `eodhd`,
`hn`) returns a list of `NewsItem` (`src/news_sentiment/types.py`):

```python
NewsItem(
    ticker="FAKE",
    published_at=datetime(2024, 3, 14, 9, 31, 0),   # TRUE publish time, UTC-naive
    source="synthetic",
    title="FAKE beats expectations, raises guidance",
    body="",
    url="synthetic://FAKE/427",
    meta={"synthetic": True},
)
```

The `published_at` is the single most important field — it carries the
point-in-time contract. The lookahead test re-runs every feature on items
truncated at a check date `t` and asserts the value at `t` matches the
value from the full stream, so this field is what guarantees no future
leak.

### 2. A scored item after the lexicon scorer

```python
ScoredItem(
    item=<the NewsItem above>,
    sentiment=+0.33,    # in [-1, +1] — bullish positive, bearish negative
    relevance=1.0,      # in [0, 1] — how much this item is about the ticker
    confidence=0.67,    # in [0, 1] — how sure the scorer was
    scorer="lexicon",
)
```

### 3. Daily aggregation (inside `features/_common.items_to_daily`)

Scored items collapse into a daily DataFrame before any feature runs:

```
              count  mean_sentiment  mean_relevance
2024-03-13        2          -0.500            1.0
2024-03-14        3          +0.330            1.0
2024-03-15        1           0.000            1.0
```

### 4. A feature row aligned to the trading-day index

`align_to_trading_days` reindexes daily aggregates onto trading days,
**shifts forward by one trading day** (this is the PIT enforcement), then
each feature applies its own rolling math:

```
              sentiment_level  sentiment_momentum  abnormal_mention_volume
2024-03-14            0.000              0.000                     0.00
2024-03-15           -0.500             -0.500                     0.41
2024-03-18           -0.083             -0.250                     0.92
2024-03-19           +0.330             +0.330                    -0.11
```

A feature value at row `2024-03-15` aggregates items published strictly
before `2024-03-15` — that's the shift in action. The feature matrix this
produces is what feeds `expanding_window_backtest` in `lidr_core`.

### 5. The schema-v2 artifact that comes out the other end

After the backtest, predictions are written to
`artifacts/predictions/news_sentiment/<config>-<timestamp>.json`. Same
shape `ta_ensemble` produces — it's the contract `lidr` reads:

```json
{
  "schema_version": 2,
  "model_id": "news_sentiment",
  "model_version": "0.1.0",
  "config_name": "news_dev_synthetic",
  "ticker": "FAKE",
  "generated_at": "20260528-113425",
  "metrics": { "classification": {...}, "strategy": {...}, "benchmark": {...} },
  "predictions": [
    { "date": "2024-03-15", "recommendation": "HOLD", "probability_up": 0.52, "y_pred": 1, "y_true": 0 }
  ]
}
```

The full schema is at
[`packages/lidr_core/src/lidr_core/contract/schema/artifact.schema.json`](../lidr_core/src/lidr_core/contract/schema/artifact.schema.json);
the smoke test validates every produced artifact against it.

## Layout

```
src/news_sentiment/
  cli.py  pipeline.py  types.py
  datasources/    synthetic + edgar + gdelt + finnhub + apewisdom + eodhd + hn
                  (+ reddit / google_trends permanent stubs)
  ingest/         collector — timestamped on-disk cache, dedup by content hash
  scoring/        lexicon + finbert + llm + hybrid
  features/       registry + sentiment_level / sentiment_momentum / abnormal_mention_volume
  configs/        dev.yaml (offline smoke)
```

See:
- [`docs/plans/task-2-news-sentiment-model.md`](../../docs/plans/task-2-news-sentiment-model.md) — the full plan
- [`docs/adr/0001-multi-model-repo-architecture.md`](../../docs/adr/0001-multi-model-repo-architecture.md) — why the package is shaped this way
- [`docs/research/data-sources.md`](../../docs/research/data-sources.md) — provider notes + the point-in-time caveat
