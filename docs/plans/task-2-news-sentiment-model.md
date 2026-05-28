# Task 2 — Build the news-sentiment model

> **For:** a Claude Code session, **after Task 1 has merged.** **Type:** new model logic.
> **Read first:** `CLAUDE.md`, [`../adr/0001-multi-model-repo-architecture.md`](../adr/0001-multi-model-repo-architecture.md) (incl. its "The artifact contract (schema v2)" section), [`../research/data-sources.md`](../research/data-sources.md).

## Goal

Fill in the `news_sentiment` package: a standalone model that turns recent news / internet discussion about **high-news individual stocks** into a BUY / HOLD / SELL recommendation, scored by the same `lidr_core` harness as `ta_ensemble` so the two compete apples-to-apples. The hypothesis is a **synthesis/attention edge** (recognizing shifts in aggregate sentiment, attention spikes, and post-event drift), **not** a latency edge — we will always be late to a single headline, so don't optimize for speed.

## Design priorities (from Boon)

1. **Swappable data sources.** Adding or swapping a source must be one adapter + one config line. No source hard-coded into the model.
2. **Easy iterate-and-compare loop.** Running a new version and comparing it against prior versions and against `ta_ensemble` must be trivial — lean on `results_log` + `manifest.json` from `lidr_core`.
3. **Swappable features and model.** Features and learner are config-selected, like the signals already are.
4. **LLM that works, with cost control.** Use the FinBERT + LLM hybrid, but cap and cache LLM spend.

## Package structure

```
news_sentiment/src/news_sentiment/
  datasources/
    base.py          # implements lidr_core DataSource protocol: fetch(ticker, start, end) -> items[]
    edgar.py         # SEC EDGAR 8-K / full-text   (free, point-in-time clean)
    gdelt.py         # GDELT volume + tone          (free, deep history)
    reddit.py        # PRAW live                    (free, live-only history)
    google_trends.py # pytrends attention proxy     (free)
    tiingo.py        # Tiingo News add-on           (paid ~$10/mo, ~10yr tagged history)
  ingest/
    collector.py     # pulls items, dedups, caches raw + PUBLISH timestamps to disk
  scoring/
    finbert.py       # local, free, bulk scorer
    llm.py           # LLM scorer: sentiment + relevance + ticker + event-type; cached, capped
    lexicon.py       # VADER / Loughran-McDonald fallback
  features/
    registry.py      # name -> feature fn, like signals/registry
    sentiment_level.py  sentiment_momentum.py  abnormal_mention_volume.py
    dispersion.py    novelty.py   # each: items -> daily Series aligned to price index
  configs/
    dev.yaml         # synthetic/offline-safe smoke config
    news_v0.yaml     # first real config: which sources, scorer, features, model, horizon
  pipeline.py        # wiring; calls lidr_core for backtest/eval/contract
```

## Build order (phased)

**Phase 0 — start the data clock immediately.** Implement `collector.py` + the free adapters (EDGAR, GDELT, Reddit, Google Trends) and begin logging raw items with publish timestamps to disk *now*. Reddit history can't be backfilled, so every day of delay is lost training data. Pick 5–10 high-news names (e.g. mega-cap tech + a couple of high-chatter names) for the first universe; make the universe a config field.

**Phase 1 — historical backtest depth via Tiingo.** Implement `tiingo.py`. Tiingo's ~10yr tagged history is what lets a backtest run *now* instead of waiting months. **Point-in-time discipline is mandatory:** store true publish timestamps, use only items published strictly before each prediction timestamp, and extend the lookahead test to cover news features (mirror `packages/ta_ensemble/tests/test_no_lookahead.py`).

**Phase 2 — scoring (FinBERT + LLM hybrid).** FinBERT scores the bulk for free. The LLM handles ambiguous / high-relevance items and adds relevance + entity-linking + event-type. **Cost controls (required):**
- Cache every LLM result keyed by a content hash — never pay twice for the same text.
- A per-run budget cap in config (`scoring.llm.max_calls` / `max_usd`); fall back to FinBERT when exceeded.
- Route to the LLM only when FinBERT confidence is low or relevance is uncertain (tiered, not blanket).
- Log spend per run so cost-vs-quality is measurable across versions.

**Phase 3 — features.** Implement the feature set as pluggable functions (config-selected): sentiment level, sentiment momentum (Δ), **abnormal mention volume** (z-score vs trailing baseline — the spike is the signal), dispersion/disagreement, novelty. Features emit raw continuous quantities; the model learns thresholds (same philosophy as the TA signals).

**Phase 4 — backtest, evaluate, compare.** Wire `pipeline.py` to `lidr_core`'s backtest + eval. Run `news_v0.yaml`, append a `results_log` row (`model_id: news_sentiment`), regenerate `manifest.json`, and compare `skill_score` and per-period strategy returns against `ta_ensemble` and buy-and-hold. Use the longer prediction horizon discussion from the existing roadmap (the 5-day-sign target is very noisy — a 20-day horizon likely suits news/drift better; make `target.horizon_days` a config field).

## Definition of done (for a first checkpoint)

- Free adapters + collector running and caching timestamped items; universe is a config field.
- Tiingo adapter working; a backtest runs on real historical news.
- FinBERT scoring works offline; LLM scoring works with caching + budget cap + spend logging.
- At least 3 features implemented and config-selectable; lookahead test covers them.
- `news_v0.yaml` produces a schema-v2 artifact that validates, plus a `results_log` row and updated `manifest.json`.
- A comparison (chart + per-period table) of `news_sentiment` vs `ta_ensemble` vs buy-and-hold, per the outcome-changing-PR evidence convention.
- Honest verdict on whether there's any edge — same edge-gate framing as `ta_ensemble`. A negative result, clearly shown, is a valid outcome.
- **Delete this plan doc** (`docs/plans/task-2-news-sentiment-model.md`) once the checkpoint PR merges — the PR description and CLAUDE.md Recent Changes capture the outcome, so the plan itself is disposable. The ADR and `docs/research/data-sources.md` stay as the durable record.

## Watch out for

- **Lookahead / backfill leakage** — the #1 way sentiment backtests lie. Validate Tiingo timestamps are true publish times (see the point-in-time caveat in the research doc).
- **Survivorship bias** — worse for high-news individual names (blown-up meme stocks delist and vanish). The existing yfinance gotcha applies harder here.
- **Don't chase latency.** If results only look good with intraday timing, that's not a free-data-realistic edge.

---

## Claude Code kickoff prompt (Task 2 — run only after Task 1 merges)

```
Read CLAUDE.md, then docs/adr/0001-multi-model-repo-architecture.md
(including its "The artifact contract (schema v2)" section),
docs/research/data-sources.md, and docs/plans/task-2-news-sentiment-model.md.

Execute Task 2: build the news_sentiment model in packages/news_sentiment, reusing
the lidr_core harness so it competes apples-to-apples with ta_ensemble. Follow the
phased build order. Hard requirements: pluggable/config-selected data sources,
features, and model; strict point-in-time discipline with a lookahead test covering
news features; FinBERT+LLM hybrid scoring with LLM result caching, a per-run budget
cap, and spend logging. Use free sources (EDGAR/GDELT/Reddit/Google Trends) plus the
paid Tiingo News adapter for historical backtest depth.

Deliver a first checkpoint: news_v0.yaml producing a validated schema-v2 artifact, a
results_log row, an updated manifest.json, and an honest comparison (chart +
per-period table) of news_sentiment vs ta_ensemble vs buy-and-hold. Follow the
protected-main PR workflow and the outcome-changing-PR evidence convention.
```
