# news_sentiment

A competing model for `lidr-models`: turn recent news and internet discussion
about high-news individual stocks into a BUY / HOLD / SELL recommendation,
scored by the same `lidr_core` harness as `ta_ensemble` so the two compete
apples-to-apples.

**Hypothesis.** Synthesis / attention edge — recognizing shifts in aggregate
sentiment, attention spikes, and post-event drift. **Not** a latency edge; we
are always late to a single headline by construction.

## Status (PR-A — scaffolding + Phase 0 free adapters)

- Package layout, `DataSource` and `Feature` registries wired against
  `lidr_core` protocols.
- Free data adapters: `synthetic`, `edgar`, `gdelt`, `reddit`, `google_trends`.
  Real implementations; heavy deps (`praw`, `pytrends`) are lazy-imported so
  the offline dev path works without them. `tiingo` is a stub that raises
  `NotImplementedError` until PR-B.
- Collector with timestamped on-disk cache + dedup-by-content-hash so the data
  clock starts now (Reddit history can't be backfilled — every day of delay
  is lost training data).
- Scoring: deterministic `lexicon` (Loughran-McDonald-style word counts) works
  offline. `finbert` and `llm` are stubs with the cache + budget-cap +
  spend-log scaffolding in place; both raise until PR-B.
- Three features: `sentiment_level`, `sentiment_momentum`, `abnormal_mention_volume`.
- `configs/dev.yaml` runs the full pipeline on synthetic prices + synthetic
  items end-to-end (no internet, no keys).

PR-A intentionally does **not** ship a real `news_v0.yaml` backtest or a
news-vs-TA comparison. Those need Tiingo News + Anthropic + Reddit credentials
and FinBERT installed; they land in PR-B and PR-C.

## CLI

```bash
make install                                                   # installs all three packages editable
python -m news_sentiment list-features                         # discover registered features
python -m news_sentiment list-sources                          # discover registered data adapters
python -m news_sentiment backtest packages/news_sentiment/configs/dev.yaml
```

## Layout

```
src/news_sentiment/
  cli.py  pipeline.py  types.py
  datasources/    synthetic + edgar + gdelt + reddit + google_trends + tiingo
  ingest/         collector — timestamped on-disk cache, dedup by content hash
  scoring/        lexicon (working) + finbert / llm (stubs, real interfaces)
  features/       registry + sentiment_level / sentiment_momentum / abnormal_mention_volume
  configs/        dev.yaml (offline smoke)
```

See:
- [`docs/plans/task-2-news-sentiment-model.md`](../../docs/plans/task-2-news-sentiment-model.md) — the full plan
- [`docs/adr/0001-multi-model-repo-architecture.md`](../../docs/adr/0001-multi-model-repo-architecture.md) — why the package is shaped this way
- [`docs/research/data-sources.md`](../../docs/research/data-sources.md) — provider notes + the point-in-time caveat
