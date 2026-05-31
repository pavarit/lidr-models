# PR-C — Real `news_v0.yaml` backtest (end-to-end intent)

> **Kind:** disposable plan (see [`../README.md`](../README.md)). **Self-deletes on PR-C merge**, alongside [`task-2-news-sentiment-model.md`](task-2-news-sentiment-model.md). The ADR and [`../research/data-sources.md`](../research/data-sources.md) stay as the durable record.
> **For:** the Claude Code session that runs PR-C, after revised PR-B merged ([PR #40](https://github.com/pavarit/lidr-models/pull/40), live-verified 2026-05-30).
> **Read first:** `CLAUDE.md`, [`task-2-news-sentiment-model.md`](task-2-news-sentiment-model.md) (PR-C section), [`../adr/0001-multi-model-repo-architecture.md`](../adr/0001-multi-model-repo-architecture.md) (artifact contract), [`../research/data-sources.md`](../research/data-sources.md) (history-depth + point-in-time caveat), and the **`verify-evidence`** skill.

## What PR-C is for

PR-C is the first **honest backtest** of `news_sentiment`: it runs the real data stack end-to-end on a high-news individual-stock universe and reports, against the same harness as `ta_ensemble`, whether news features carry any directional edge. **A clearly-shown negative result is a valid, expected outcome** — the bottleneck across this repo is features/target, not model class, and the news hypothesis (a synthesis/attention edge, not a latency edge) has not yet been tested on real data. PR-C settles that question for `news_v0`.

PR-C builds **no new pipeline infrastructure.** Everything from the feature matrix forward already exists in `lidr_core`; PR-B made every adapter and scorer real. PR-C's work is: write the config(s), run the loop, and produce the comparison evidence.

## The intent, stage by stage (beginning to end)

This is the full `news_v0` pipeline. Each stage names the concrete choice for v0 and points at the code that already implements it (`packages/news_sentiment/src/news_sentiment/pipeline.py::run_pipeline`).

**0. Universe + run topology.**
Universe is **NVDA, TSLA, GME, AMC, PLTR, MU, USO** (7 names) — high-news names, deliberately including meme/retail-attention names where the hypothesis should bite hardest, plus MU (semis) and USO (a commodity/oil ETF, which broadens the test beyond single-company news to macro-news flow). The pipeline runs **one ticker at a time** (`pipeline.py` raises `NotImplementedError` for >1 ticker — this is a hard guard, not a bug). PR-C therefore runs the pipeline **once per ticker** (the per-ticker-loop decision): one config per ticker plus a `make backtest-news-v0` target that runs all seven, producing one report + one artifact + one `results_log` row per ticker. No pooling, no removal of the guard. One ticker-specific note to verify during build: USO is an ETF, so EODHD's company-news endpoint may return thinner/different coverage than for the individual names — check it isn't empty before relying on its features.

**1. Prices.** `lidr_core.data.loaders.load_prices` pulls OHLCV from yfinance, `auto_adjust=True`, cached per `(ticker, start, end)`. Survivorship caveat bites harder here than on ETFs — see *Watch out for*.

**2. News ingestion.** `ingest/collector.collect` fans out across the configured sources for the ticker over `[start, end)`, dedups by `content_hash`, and persists each item with its **true publish timestamp** to a JSONL cache. **v0 sources: EODHD + GDELT + EDGAR only.** These three have real multi-year history at run-time. Finnhub (1-yr history) and Apewisdom (live-snapshot, no accumulated history yet) are **excluded from v0** — including either would give a degenerate, mostly-empty timeseries over a multi-year window. They return in later phases (see *Phasing*).

**3. Scoring.** **FinBERT-only for the v0 headline run** — `scoring/finbert.py` (`ProsusAI/finbert`, `sentiment = pos − neg`, `confidence = max softmax`). Chosen because it is free, deterministic, and fully reproducible, which is what a first honest baseline needs. The LLM hybrid is deferred to v0.1 so we can measure what it adds rather than entangle it with the first read. No LLM spend in this PR.

**4. Features.** Config-selected from the registry. v0 uses the three shipped features — `sentiment_level`, `sentiment_momentum`, `abnormal_mention_volume` — computed over the EODHD/GDELT/EDGAR item stream. `features/_common.align_to_trading_days` reindexes daily aggregates onto trading days and **shifts forward one trading day** — this is the point-in-time enforcement (a feature row at date `t` only sees items published strictly before `t`). The parametrized lookahead test guards it.

**Cross-source coverage must degrade gracefully, not error (signed-off requirement, decision #2).** The three sources have *different history depths* (EDGAR/GDELT deep, EODHD ~2022+), so within a single ticker the per-source daily aggregates start on different dates. The feature math must treat a source with no items on a day as a legitimate zero/NaN, not raise. The harness already aligns via `pd.concat(...).dropna()` at the target-join step — the build must confirm this drops only genuinely-unusable rows and does **not** silently empty the feature matrix when one source is sparse early in the window. Add an assertion that the post-alignment matrix is non-empty with a clear error naming the offending source/ticker if it isn't, so a coverage gap fails loudly instead of producing a degenerate backtest.

**5. Target.** `forward_return_binary`, **`horizon_days: 5`** (settled by the 2026-05-28 horizon spike — skill_score is least-bad at h5 and degrades monotonically with longer horizons; short horizon also matches news's short-lived impact), `threshold: 0.0`. Same target shape as `ta_ensemble`, so the two are strictly comparable.

**6. Backtest.** `expanding_window_backtest` — fit on train, predict OOS on test, discard the model each split. **Split params shrink for the short window (signed off): `initial_train_years: 1`, `test_period_months: 6`.** The smoke default of 3 train years over a ~4-year EODHD window would leave almost no out-of-sample data; 1 year + 6-month tests gives several OOS folds. Report `n_obs` and the small-sample caveat plainly — this is a real statistical-power limit of cheap news history, not papered over.

**7. Eval + report.** `classification_metrics` (accuracy beside `base_rate`/`pred_rate`, `log_loss` beside `base_logloss`) + `strategy_metrics` (cagr, sharpe, max_drawdown) beside the buy-and-hold benchmark, plus per-year breakdowns. Equity curve runs on **1-day-forward returns**, not the 5-day target (the documented gotcha — `add_strategy_returns` already does this correctly). HTML report per run under `reports/<config>-<timestamp>/`.

**8. Artifact + manifest.** Each run writes a `schema_version: 2` artifact to `artifacts/predictions/news_sentiment/<config>-<timestamp>.json` (`model_id: news_sentiment`), validated on write. `manifest.json` is refreshed so the leaderboard discovers the runs. `build_manifest` already excludes smoke runs and picks the headline artifact by embedded `generated_at`.

**9. Comparison evidence (the deliverable).** Per the outcome-changing-PR convention: a chart + dated per-period table comparing **news_sentiment vs ta_ensemble vs buy-and-hold**. `ta_ensemble`'s existing runs are on indices/ETFs, so a like-for-like comparison needs an h5 logistic `ta_ensemble` re-run on these same seven names (decision #6).

## Phasing — v0 now, lift tests next

Your window and scorer choices set up a deliberate progression. PR-C ships **v0**; the rest are later configs, not this PR.

- **`news_v0.yaml` (this PR)** — EODHD+GDELT+EDGAR, FinBERT-only, 3 features, 7 tickers, h5 binary, logistic. The honest baseline.
- **`news_v0.1.yaml` (next)** — (a) add Finnhub features and test whether **all-sources** gives lift over the v0 source mix; (b) swap in the **FinBERT+LLM hybrid** scorer (with the $5/mo cap + cache) and measure what the LLM adds over FinBERT-only. Two independent lift tests, each against the v0 baseline.
- **`news_v0.2+` (later)** — fold in **Apewisdom** retail-attention features once enough history has forward-accumulated (months, not days). Optionally add **Quiver Quantitative** ($30/mo) for historical WSB depth *only if* the news side shows a hint of edge.

## Decisions resolved (your calls, 2026-05-30)

| # | Decision | Choice |
|---|---|---|
| 1 | Multi-ticker topology | **Per-ticker loop** — one config + run + artifact + results_log row per ticker; pipeline core untouched. Driven by a `make backtest-news-v0` target. |
| 2 | Window + source mix | **EODHD+GDELT+EDGAR first** (Finnhub excluded); **uniform window across tickers**; cross-source coverage gaps must degrade gracefully (see below + pipeline stage 4). All-sources lift test in v0.1. |
| 3 | Scorer | **FinBERT-only** for the headline; FinBERT+LLM hybrid in v0.1. |
| 4 | Universe | **NVDA, TSLA, GME, AMC, PLTR, MU, USO** (7 names: high-news + meme + semis + an oil ETF). |
| 5 | Backtest folds | **`initial_train_years: 1`, `test_period_months: 6`** — short EODHD window can't support 3 train years; report `n_obs` + small-sample caveat. |
| 6 | Comparison baseline | Re-run an **h5 logistic `ta_ensemble`** config on the same 7 tickers; compare news_sentiment vs that TA baseline vs buy-and-hold per ticker. |
| 7 | Model/learner | **Logistic regression, `class_weight=None`** only for v0 (the gotcha: `balanced` is harmful on this ~60/40 problem). **LightGBM flagged for the next phase.** |
| 8 | Quota | **Cache the cold pull** — one-time collect into the append-only JSONL cache; re-runs are then free. Surface the EODHD call-count estimate (5 calls/request) before the live pull and stage across days if the daily quota is tight. |

### Notes on the signed-off items

- **Uniform window (decision #2).** Set `start_date` to the **latest** of the seven per-ticker EODHD earliest-article dates so every ticker shares one comparable window. The build should probe EODHD for each ticker's earliest article first, then pick that common start. The cross-source robustness clause (pipeline stage 4) is the second half of this decision: differing per-source history depths within a ticker must not error or silently empty the matrix — fail loudly with a named source/ticker if coverage is unusable.
- **USO coverage check.** USO is an ETF; confirm EODHD's company-news endpoint returns real items for it before counting on its features — if it's empty, note it and let USO lean on GDELT/EDGAR macro flow.

## Watch out for

- **Backfill / lookahead leakage — the #1 way sentiment backtests lie.** EODHD does not guarantee strict point-in-time correctness. PR-B's timestamp spot-check was the cheap pre-validation; the `align_to_trading_days` one-day shift + the parametrized lookahead test are the enforcement. Don't relax either.
- **Survivorship bias bites harder on meme names.** yfinance only has currently-listed tickers; blown-up names vanish. GME/AMC/PLTR are currently listed, but treat individual-name results with the gotcha in mind and don't over-read them.
- **Small-sample statistics.** A ~4-year window with shrunk folds produces few OOS observations per ticker. Report `n_obs` prominently; a flattering metric on a tiny sample is noise.
- **Don't chase latency.** If results only look good with intraday timing, that's not a free-data-realistic edge — the hypothesis is synthesis/attention, not speed.
- **The equity-curve gotcha is already handled** — `add_strategy_returns` uses 1-day-forward returns, not the 5-day target. Don't "fix" it.

## Per-ticker loop (signed off)

**One config per ticker** (`news_v0_nvda.yaml` … `news_v0_uso.yaml`, identical but for the ticker) plus a **`make backtest-news-v0`** target that runs all seven. This touches no pipeline code, keeps each artifact/`results_log` row clean and independently re-runnable, and the manifest already keys off `config_name`. The seven configs share the same `start_date`/`end_date` (the uniform window), sources, scorer, features, target, fold params, and model — only `data.tickers` and `news.universe` differ.

## Definition of done

- `news_v0` configured (one config per ticker + `make backtest-news-v0`) and run across all seven tickers (EODHD+GDELT+EDGAR, FinBERT-only, h5 binary, 1yr-train/6mo-test folds, uniform window).
- Cross-source coverage degrades gracefully — a sparse source neither errors nor silently empties the matrix; the non-empty-matrix assertion is in place.
- One `schema_version: 2` artifact per ticker under `artifacts/predictions/news_sentiment/`, all validating on write; `manifest.json` refreshed.
- One `results_log.csv` row per ticker (`model_id: news_sentiment`).
- Stage-1 pressure-test (per `verify-evidence`) run and recorded; every committed figure recomputed from the artifact JSON and cross-checked against its `results_log` row within tolerance.
- **A complete, self-contained backtesting report published to `docs/reports/<merge-date>-news_sentiment-v0/`** following [`../reports/README.md`](../reports/README.md). The folder contains: `report.md` (intent, methodology, inputs/outputs, data sources & rationale, key choices, outcomes analysis, limitations, Stage-1 checklist), `chart.png` (+ recent-window zoom), `configs/` (frozen copies of all seven news configs **and** the TA-baseline config), `artifacts/` (frozen copies of the seven prediction JSONs + the matching `results_log` rows — these are gitignored at their source, so the copy is what makes the report reproducible), `REPRODUCE.md` (PR #, full 40-char commit SHA, exact commands, environment + `env.lock`, data provenance incl. yfinance/EODHD pull dates, seeds, EODHD call-count + any LLM spend), and optionally the rendered `report.html`. The comparison (news_sentiment vs h5-logistic ta_ensemble re-run on the same seven tickers vs buy-and-hold) lives in `report.md`, not the PR description.
- **The PR description points to the report folder** rather than inlining the analysis.
- Honest verdict stated at the top of `report.md` — edge / no edge / inconclusive, with `n_obs` and the small-sample caveat visible — and what it gates (the edge gate stays closed unless the result clears it).
- Index row added to `docs/reports/README.md`.
- **Cleanup commit deletes both this doc and `task-2-news-sentiment-model.md`** (the report folder is durable and stays). Update `CLAUDE.md` Active Task (PR-C → done), Architecture (`news_sentiment` status), `docs/README.md` (drop both plans from the index), and append a `docs/changelog.md` entry linking the report folder. Cross-link to lidr only if the artifact/integration contract changed (it shouldn't here).

## Claude Code kickoff prompt (PR-C — all decisions signed off 2026-05-30)

```
Read CLAUDE.md, docs/plans/pr-c-news-v0-backtest.md, docs/adr/0001-multi-model-repo-architecture.md
(artifact contract section), docs/research/data-sources.md, and the verify-evidence skill.

Execute PR-C: the first real news_v0 backtest of the news_sentiment package.
Build NO new pipeline infra — the pipeline, adapters, and scorers are all real as of PR-B.

Universe: NVDA, TSLA, GME, AMC, PLTR, MU, USO. Topology: ONE config per ticker
(news_v0_<ticker>.yaml) + a `make backtest-news-v0` target that runs all seven. The
single-ticker guard stays. Sources: EODHD + GDELT + EDGAR only (Finnhub + Apewisdom
excluded from v0 — insufficient history). Scorer: FinBERT-only. Features: the three shipped
(sentiment_level, sentiment_momentum, abnormal_mention_volume). Target: forward_return_binary,
horizon_days 5, threshold 0.0. Model: logistic_regression, class_weight=None (no LightGBM —
that's the next phase).

Window: UNIFORM across tickers. Probe EODHD for each ticker's earliest article, set start_date
to the latest of the seven so every ticker shares one window. Backtest folds: initial_train_years
1, test_period_months 6 (short EODHD window can't support 3 train years; report n_obs + the
small-sample caveat).

Cross-source robustness: differing per-source history depths within a ticker must NOT error or
silently empty the feature matrix. Assert the post-alignment matrix is non-empty and fail loudly
naming the source/ticker if coverage is unusable. USO is an ETF — confirm EODHD company-news
returns real items for it; if empty, note it and let it lean on GDELT/EDGAR.

Quota: do a one-time cold collect into the append-only JSONL cache (re-runs are then free);
surface the EODHD call-count estimate (5 calls/request) before the live pull, stage across days
if the daily quota is tight.

Per ticker: write a schema-v2 artifact (model_id news_sentiment), append a results_log row,
refresh the manifest. Run verify-evidence Stage 1; recompute every figure from the artifact
JSON and cross-check vs results_log. Comparison: news_sentiment vs ta_ensemble (re-run an h5
logistic ta_ensemble config on these same seven tickers) vs buy-and-hold; a negative result
clearly shown is a valid outcome.

Publish a complete backtesting report to docs/reports/<merge-date>-news_sentiment-v0/ per
docs/reports/README.md — report.md (intent, methodology, inputs/outputs, data sources +
rationale, key choices, outcomes analysis, limitations, Stage-1 checklist, verdict at top),
chart.png (+ recent-window zoom), configs/ (frozen copies of all seven news configs + the TA
baseline config), artifacts/ (frozen copies of the seven prediction JSONs + their results_log
rows — gitignored at source, so copy them), REPRODUCE.md (PR #, full 40-char commit SHA,
commands, env + env.lock, data provenance incl. yfinance/EODHD pull dates, seeds, EODHD
call-count). The PR description POINTS TO the folder; it does not inline the analysis. Add an
index row to docs/reports/README.md. Transcribe every committed figure from same-turn output.

Cleanup commit: delete docs/plans/pr-c-news-v0-backtest.md and
docs/plans/task-2-news-sentiment-model.md (the report folder is durable, stays); update
CLAUDE.md Active Task + Architecture, docs/README.md plans index, and append a
docs/changelog.md entry linking the report folder.

Follow the protected-main PR workflow; CI green is required.
```
