# CLAUDE.md

This file orients any AI assistant (Claude Code, Claude Cowork, etc.) joining this project. Read it before doing anything else. Then keep it current — see "Maintenance Instructions" at the bottom.

**Hosting repo**: https://github.com/pavarit/lidr-models (renamed from `lidr-ml` on 2026-05-27 after Task 1 merged; the old URL still redirects).

## Project Goal

`lidr-models` is the Python research monorepo that turns market data into **empirically calibrated BUY / HOLD / SELL recommendations** consumed by the sibling project [`lidr`](https://github.com/pavarit/lidr) (the Next.js front-end). lidr's confidence values today are heuristics (normalized strength scores). This repo's job is to replace them with probabilities learned from historical data — and to host *several competing* models, each scored on the same harness and benchmark, so the best-performing one feeds production.

Models are backtested with walk-forward validation from 2005 to today and emit a versioned JSON artifact (schema_version: 2) that lidr's `/api/signals/[ticker]` route can read directly. No running Python service is needed until lidr's FastAPI-service roadmap item is triggered.

## Architecture

Two-repo setup, deliberately kept separate:

- **`lidr`** (Next.js, `C:\Users\smnk1\Claude\Projects\lidr`) — the website. Computes signals live in TS for display, will eventually overlay calibrated probabilities from the artifacts this repo produces.
- **`lidr-models`** (this repo, Python) — a research monorepo that produces the artifacts. Three packages under `packages/`:
  - `lidr_core` — shared harness: backtest engine, eval/metrics, results_log + leaderboard, the JSON artifact contract (schema + writer + loader), the `SignalFn` / `Model` / `Feature` / `DataSource` protocols, base data loaders, generic learners (logistic / LightGBM). Owned once, reused by every model.
  - `ta_ensemble` — the six technical-analysis signals + their pipeline. Today's only complete model.
  - `news_sentiment` — Task 2, in development. PR-A merged 2026-05-28 (commit `b9ce76a`): scaffolding + four free data adapters (EDGAR / GDELT / Reddit / Google Trends + synthetic) + collector with timestamped on-disk cache + a deterministic lexicon scorer + three lookahead-safe features + an offline dev pipeline. PR-B (revised 2026-05-28 plan, see Active Task) will rewire data sources — delete Tiingo, convert Reddit + Google Trends to permanent stubs, add Finnhub / Apewisdom / EODHD adapters — and light up FinBERT + LLM scoring. The real `news_v0.yaml` backtest + comparison evidence is PR-C.

Integration is via JSON files written to `artifacts/predictions/<model_id>/<config>-<timestamp>.json`, validated against the contract on write, plus a top-level `artifacts/manifest.json` leaderboard lidr will use to discover models. Cheap, debuggable, version-controllable. A FastAPI service is on the lidr roadmap and will be added here when the lidr side is ready to consume live predictions.

## Stack

- **Python** ≥ 3.10
- **pandas / numpy** — data manipulation
- **scikit-learn** — base learners + walk-forward CV (`TimeSeriesSplit`)
- **LightGBM** — second base learner (added 2026-05-27, [PR #20](https://github.com/pavarit/lidr-models/pull/20))
- **yfinance** — free historical price data (with synthetic-data fallback for offline dev)
- **PyYAML** — config files
- **Typer** — CLI
- **jsonschema** — artifact contract validation on write/read
- **Matplotlib** — chart embedded in HTML reports (base64, no internet needed to view)
- **pytest** — tests
- **ruff** — lint + format
- **requests** — HTTP for news adapters (added 2026-05-28 with `news_sentiment` PR-A); EDGAR, GDELT, Tiingo all go through it
- **Optional extras** (lazy-imported by the relevant adapter — `pip install -e ./packages/news_sentiment[reddit,trends,scoring,llm]`): `praw` (Reddit), `pytrends` (Google Trends), `transformers + torch` (FinBERT, PR-B), `anthropic` (LLM scorer, PR-B). The dev path uses none of these.

Workspace: each package under `packages/<name>/` has its own `pyproject.toml`; the root `pyproject.toml` carries only the dev-tool config (`ruff`, `pytest`) and a meta dev install. `make install` installs all three packages editable so cross-package imports resolve.

Planned but not yet added: **MLflow** (experiment tracking — on the roadmap but deferred behind the CSV results log), **vectorbt** (faster backtest sweeps — not yet on the roadmap; revisit if the custom engine becomes the bottleneck).

## Commands

See [README.md → Quick start](README.md#quick-start) and [README.md → CLI](README.md#cli) for the canonical command reference. Targets: `make install`, `make backtest CONFIG=...` (ta_ensemble), `make backtest-news CONFIG_NEWS=...` (news_sentiment), `make test`, `make lint`, `make clean`, `make clean-reports`. Override the Python interpreter with `PYTHON=` if `python3` isn't right for your environment.

## How the pipeline works

Reading top-down:

1. **Config** (`packages/ta_ensemble/configs/*.yaml`) — declares what to run: tickers, date range, data source, which signals, which model, backtest method, transaction costs. Now also carries `model_id` + `model_version` so produced artifacts identify their model family.
2. **Data loader** (`packages/lidr_core/src/lidr_core/data/loaders.py`) — pulls OHLCV from yfinance (or generates synthetic series for offline dev). yfinance is called with `auto_adjust=True`, so close prices are total-return-adjusted (dividends + splits folded in); a "200-day high" is a 200-day high of the adjusted series, not the raw close. Cached by `(ticker, start, end)` in `data/raw/<ticker>_<start>_<end>.pkl`; the cache never expires, so `rm data/raw/*.pkl` to force a refresh from yfinance.
3. **Signals** (`packages/ta_ensemble/src/ta_ensemble/signals/`) — each signal is a pure function that takes a DataFrame of prices and returns a Series of feature values aligned to the price index. Every signal must be **lookahead-safe** (only uses data up to time *t* to compute the value at time *t*). Tested in `packages/ta_ensemble/tests/test_no_lookahead.py`.
4. **Target** (computed inline in `packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline`) — for now, binary: was the *N*-day forward return positive? Will become 3-class (BUY/HOLD/SELL) once we have a useful model.
5. **Backtest engine** (`packages/lidr_core/src/lidr_core/backtest/engine.py`) — expanding-window walk-forward. For each split: fit model on train slice, predict on test slice, store predictions. Never trains on data after the test period.
6. **Model** (`packages/lidr_core/src/lidr_core/models/`) — pluggable. Today: logistic regression and LightGBM (both generic, hence in `lidr_core`). As of 2026-05-27, both have been backtested on the six-signal feature set against the 5d-forward-return-sign target; neither beats the no-skill baseline (see Recent Changes). The roadmap is currently pivoting to target/feature reformulation rather than more model classes.
7. **Eval + report** (`packages/lidr_core/src/lidr_core/eval/`) — classification metrics (`classification_metrics` → `accuracy`, `base_rate`, `pred_rate`, `log_loss`, `base_logloss`, `n_obs`), strategy metrics (`strategy_metrics` → `cagr`, `sharpe`, `max_drawdown`, `final_equity`), and per-year breakdowns (`by_year` reports the classification fields per calendar year; `performance_by_year` reports strategy vs buy-and-hold returns and the excess). `base_rate` and `pred_rate` are deliberately shown beside `accuracy` — without them, an accuracy of 0.55 looks fine until you notice the base rate is 0.61. The equity curve itself is built in `lidr_core/backtest/engine.py::add_strategy_returns` ("go long when prediction = 1, cash otherwise"). HTML report written to `reports/<config>-<timestamp>/`. The report shows a **Strategy vs Buy & Hold** comparison table (CAGR, Sharpe, max drawdown, final equity for both legs) and a **per-year performance** table (strategy vs buy-hold return + excess) so regime-dependence is visible. A **Summary** section at the top translates the config into English and states the out-of-sample span (derived from the predictions, not the raw data range). Both the top classification metrics and the per-year table show **base_logloss** (the no-skill floor = entropy of the base rate) beside log loss, and the comparison/per-year tables green/red-highlight whichever side wins. The equity curve is marked to market with **1-day-forward returns**, not the N-day classification target — see Gotchas.
8. **Cross-run results log** (`packages/lidr_core/src/lidr_core/eval/results_log.py`) — `append_run()` appends one row per backtest to `artifacts/results_log.csv` (skill score, full-OOS + per-period log loss with base-rate floors, strategy vs benchmark CAGR/Sharpe/max-drawdown, excess) so "did this change help?" is answerable without opening every HTML report. Opt out per-config with `output.results_log: false`; defaults to true. `dev_synthetic.yaml` sets it false so the smoke test (run on every push/PR) doesn't pollute the tracked CSV with synthetic rows — `packages/ta_ensemble/tests/test_pipeline_smoke.py` asserts the file size doesn't grow.
9. **Artifact contract + leaderboard** (`packages/lidr_core/src/lidr_core/contract/`, `packages/lidr_core/src/lidr_core/eval/leaderboard.py`) — `contract/writer.py::build_artifact` assembles the schema_version: 2 payload (`model_id`, `model_version`, `config_name`, `ticker`, `metrics`, `predictions[].{date,recommendation,probability_up,y_pred,y_true}`); `write_artifact` validates against `contract/schema/artifact.schema.json` before writing. `leaderboard.py::write_manifest` scans `artifacts/predictions/<model_id>/` and emits `artifacts/manifest.json` so lidr can discover every produced model + its headline OOS skill score.

Everything for the TA model is wired together by `packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline(config_path)`.

## Folder map

```
packages/
  lidr_core/                            shared harness — owned once, reused by every model
    pyproject.toml
    src/lidr_core/
      backtest/engine.py                expanding-window walk-forward backtester + strategy returns
      data/loaders.py                   yfinance + synthetic OHLCV loader (with on-disk pickle cache)
      models/                           generic learners; reused by every model
        logistic.py                     sklearn logistic regression wrapper
        lightgbm.py                     LightGBM classifier wrapper
        __init__.py                     MODEL_REGISTRY + build_model(spec)
      eval/
        metrics.py                      classification_metrics, strategy_metrics,
                                        by_year, performance_by_year
        report.py                       HTML report generator (base64-embedded chart)
        results_log.py                  appends one row per run to artifacts/results_log.csv
        leaderboard.py                  scans artifacts/predictions/<model_id>/ → writes manifest.json
      contract/
        schema/artifact.schema.json     formalized schema_version: 2 contract
        writer.py                       build_artifact + write_artifact (validates on write)
        loader.py                       load_artifact (validates on read)
      protocols/
        signal.py                       SignalFn protocol (was lidr_ml.signals.base)
        model.py                        Model protocol (was lidr_ml.models.base)
        feature.py                      Feature protocol (Task 2 placeholder)
        datasource.py                   DataSource protocol (Task 2 placeholder)
    tests/
      test_backtest_engine.py           guards the shared backtest engine
      test_strategy_returns.py          guards the 1-day-forward equity-curve rule
  ta_ensemble/                          today's six-signal TA model — depends on lidr_core
    pyproject.toml
    configs/                            experiment configs (YAML, one per run)
      baseline.yaml                     SPY, 2005–today, SMA crossover + logistic regression
      baseline_six_signals.yaml         all six signals — edge-gate checkpoint (loses)
      baseline_six_signals_unweighted.yaml  same, no class_weight; parity-baseline for Task 1
      baseline_six_signals_lightgbm.yaml    LightGBM checkpoint
      dev_synthetic.yaml                offline-safe smoke test (results_log opt-out)
    src/ta_ensemble/
      __init__.py
      __main__.py                       entry for `python -m ta_ensemble`
      cli.py                            Typer CLI (`backtest <config>`, `list-signals`)
      pipeline.py                       end-to-end orchestrator — calls lidr_core for harness work
      signals/
        registry.py                     name → signal-callable lookup
        sma_crossover.py / rsi.py / macd.py / bollinger.py / breakout.py / volume.py
                                        ports of lidr's lib/signals/*.ts
    tests/
      conftest.py                       shared `synthetic_prices` fixture
      test_no_lookahead.py              asserts every registered signal is lookahead-safe
      test_signal_accuracy.py           element-wise correctness + hand-derived spot checks
      test_pipeline_smoke.py            runs dev_synthetic config end-to-end
  news_sentiment/                       Task 2 model — depends on lidr_core. PR-A: scaffolding + Phase 0
    pyproject.toml  README.md
    configs/
      dev.yaml                          offline smoke (synthetic prices + synthetic news, lexicon scorer)
    src/news_sentiment/
      __init__.py  __main__.py  cli.py  pipeline.py  types.py
      datasources/
        base.py                         BaseNewsSource ABC + (start,end) window contract
        synthetic.py                    deterministic offline items — backs the smoke test
        edgar.py                        SEC EDGAR full-text/8-K  (free, point-in-time clean)
        gdelt.py                        GDELT 2.0 DOC API         (free, deep history)
        reddit.py                       PRAW                       (free, lazy-imports praw, live-only)
        google_trends.py                pytrends                   (free, lazy-imports pytrends)
        tiingo.py                       stub — raises until PR-B
      ingest/collector.py               fan-out, dedup by content_hash, JSONL cache per (ticker,source)
      scoring/
        lexicon.py                      Loughran-McDonald-style word counts (PR-A's only real scorer)
        finbert.py                      stub — raises until PR-B
        llm.py                          cache + budget cap + spend log scaffolding real; live call PR-B
        _lm_wordlist.py                 small embedded LM word list
      features/
        registry.py  _common.py         registry + items_to_daily + the PIT shift+align helper
        sentiment_level.py              trailing-window mean of per-item sentiment
        sentiment_momentum.py           fast-minus-slow trailing sentiment (MACD-shaped)
        abnormal_mention_volume.py      z-score of item count vs trailing baseline (attention spike)
    tests/
      conftest.py                       trading_days + synthetic_scored_items fixtures
      test_pipeline_smoke.py            runs dev.yaml end-to-end, asserts schema-v2 artifact validates
      test_no_lookahead.py              every registered news feature is lookahead-safe
      test_features.py                  hand-derived spot checks per feature
      test_scoring.py                   lexicon correctness + LLM cache/budget invariants
      test_datasources.py               registry + synthetic determinism + Tiingo stub
      test_collector.py                 dedup across runs + timestamp round-trip
data/
  raw/                                  cached OHLCV pulled from yfinance
  news/                                 news_sentiment collector cache — JSONL per (ticker, source), gitignored
reports/                                generated HTML reports (gitignored except .gitkeep)
artifacts/
  results_log.csv                       cross-run results log (one row per backtest, tracked in git)
  manifest.json                         leaderboard — one entry per model_id with latest artifact + skill
  predictions/<model_id>/<config>-<timestamp>.json
                                        v2 artifacts written here, one subdir per model
docs/
  adr/0001-multi-model-repo-architecture.md  the keystone decision + schema-v2 design (durable)
  research/data-sources.md                   news/sentiment source comparison (durable)
  plans/task-2-news-sentiment-model.md       Task 2 kickoff (disposable, deleted on Task 2 merge)
  signals.md                                 first-time-reader explainer for the six TA signals
  signals/                                   per-signal PNG charts embedded by signals.md
  sample-report/report.html                  committed sample of the HTML backtest report
pyproject.toml                          root: ruff + pytest config, dev install only — no top-level package
Makefile                                make install / backtest / test / lint / clean
.github/workflows/test.yml              CI: installs all packages editable, then `make test` + `make lint`
```

## Conventions (read before writing code)

- **`main` is protected — all changes land via PR.** Direct pushes are blocked. Workflow: branch → commit → push → `gh pr create` → wait for green CI → `gh pr merge --squash --delete-branch`. Full procedure in [CONTRIBUTING.md → Workflow](CONTRIBUTING.md#workflow). Unlike the sibling lidr project there is no Vercel preview here — CI passing is the only required gate, because lidr-models doesn't auto-deploy anywhere.
- **Python ≥3.10**, src-layout packages under `packages/`; ruff for lint + format (`line-length = 100`, `target-version = "py310"`). Snake_case modules and functions.
- **`lidr_core` is the harness; per-model packages depend on it. Never the other way around.** If a piece of logic is reused by more than one model (eval, contract, backtest engine, base learners, data loaders), it lives in `lidr_core`. Model-specific signals/features/configs/pipelines live in the model's package.
- **Signals are pure functions**: `(prices: DataFrame, params: dict) → Series` aligned to the input index. Conform to `lidr_core.protocols.signal.SignalFn`. Must be lookahead-safe — `f(prices[:t])[t] == f(prices)[t]`.
- **Models conform to `lidr_core.protocols.model.Model`**: three methods — `fit(X, y) -> None`, `predict_proba(X) -> np.ndarray` (shape `(n_samples, n_classes)`), and `predict(X) -> np.ndarray`.
- **Artifacts are validated on write.** Every produced `predictions/<model_id>/*.json` must validate against `lidr_core/contract/schema/artifact.schema.json`; `write_artifact` enforces this. Breaking schema changes bump `schema_version`; additive optional fields don't. New models must populate `model_id` + `model_version` in config; the writer reads them from there.
- **When adding a signal, three things must land in the same PR** (the signal lives in its owning model's package, e.g. `ta_ensemble/signals/`):
  1. Register in `ta_ensemble/signals/registry.py`.
  2. Add to the `SIGNAL_CASES` table in `packages/ta_ensemble/tests/test_no_lookahead.py` (the lookahead test is non-negotiable).
  3. Add an `ACCURACY_CASES` entry in `packages/ta_ensemble/tests/test_signal_accuracy.py` — a 5-tuple `(name, params, reference_fn, prices_factory, spot_checks)`. The `reference_fn` should be a *structurally different* implementation of the same algorithm (e.g., loop-based numpy reference vs the signal's vectorized pandas path) so a shared bug between signal and reference is unlikely. The `prices_factory` returns the price fixture used for the layer-2 spot checks (typically chosen so the expected values are derivable by hand without running code; ≥2 spot checks required). The layer-1 element-wise tolerance is `rtol=1e-8`.
- **Outcome-changing PRs need verification evidence in the PR description.** PRs that affect what the pipeline outputs (signal ports, model updates, scoring/calibration changes, JSON-artifact schema changes) need a `## Summary` section explaining the change in plain English plus an embedded chart and dated sanity-check table from real SPY data. For signal ports, also cite a max-absolute-difference number vs lidr's TS implementation. Refactors, doc-only changes, and infra/CI changes don't need this. **Mechanics:** `scripts/verify_<thing>.py` generates `docs/_pr_evidence/<thing>/chart.png` + `evidence.md`, committed to the branch then **removed in a final commit before squash-merge** so `main` stays free of review artifacts; the chart URL in the PR description pins to a commit SHA via `raw.githubusercontent.com` so it survives branch deletion. **Always get the full 40-char SHA from `git rev-parse <short>` and paste it verbatim — do not type SHA hex from memory or extend a short SHA by hand.** The raw URL needs an exact-match SHA or it returns 404; the first 7 chars looking right is not enough. PR #15 shipped with two broken chart URLs because the trailing 33 hex chars were typed from memory; caught only when the user clicked through. See PRs #5 / #7 / #8 / #9 / #10 (the five signal ports of 2026-05-27) for the template.
- **Backtests use expanding-window walk-forward only.** Random k-fold leaks information across time and is rejected on sight.
- **Transaction costs are modeled in every backtest.** Default 5 bps; configurable but never zero in a config that gets compared to buy-and-hold.
- **Every report must show a benchmark.** Strategy metrics rendered beside buy-and-hold; log loss rendered beside `base_logloss` (the no-skill floor). New comparison columns get appended to the *right* of existing benchmark columns.
- **Errors at config boundaries are `NotImplementedError`** with a message naming the unsupported value (see `pipeline.py` for the pattern). Internal invariants use `assert` or raise `ValueError`. No silent fallbacks.
- **CI runs `make test` + `make lint` on every push** (`.github/workflows/test.yml`). Don't merge red.
- **Refresh the committed sample report when report formatting changes.** If a PR touches `packages/lidr_core/src/lidr_core/eval/report.py`, `packages/lidr_core/src/lidr_core/eval/metrics.py`, or `packages/ta_ensemble/configs/baseline.yaml`, run `make refresh-sample-report` and include the updated `docs/sample-report/report.html` (and the new `artifacts/results_log.csv` row that the run appends) in the same commit. Keeps the README's "Example report" link in sync with the headline SPY baseline numbers. Requires internet (yfinance).

See **Gotchas** below for the past failures that motivated several of these rules.

## Key Decisions

Strategic forks in the road — *why we chose X over Y*. For procedural rules see Conventions; for things that bit us see Gotchas.

- **One Python project, separate from lidr.** Considered as a folder inside lidr; rejected because Python venvs + Node modules in one tree gets messy fast and the two pieces deploy on different cadences (Vercel vs. eventually Railway/Render). The deliberate seam is the JSON artifact.
- **Source of truth for signal logic stays in lidr until cutover.** Each Python signal in `packages/ta_ensemble/src/ta_ensemble/signals/` is a *port* of the corresponding TS signal in lidr's `lib/signals/`; numerical parity is tested so the two implementations don't drift. When the calibrated model goes live, we'll decide whether the TS versions get deleted, kept as fallback, or only used for display.
- **Synthetic data fallback is part of the contract.** The `dev_synthetic` config lets the pipeline run with no network — makes CI, tests, and Cowork-sandbox verification trivial. Not just a dev convenience; the existence of an offline-runnable config is load-bearing for the test suite.
- **Outputs to lidr are calibrated probabilities, not heuristics.** Today lidr's confidence values are normalized strength scores (heuristics). The whole point of this project is to replace them with `predict_proba` calibrated via Platt or isotonic regression. (The current baseline's probabilities are *not* calibrated — see Gotchas re: `class_weight="balanced"` — so the calibration step is required before any artifact goes live.)
- **The backtest does NOT produce a deployable model.** Expanding-window walk-forward fits a fresh model per split, predicts that split's test slice, and discards it — the deliverable is the stitched out-of-sample prediction series, used only for evaluation. There is no single trained model and nothing is written to `artifacts/models/`. Fitting + serializing a deployable model is a separate not-yet-built step (on the roadmap, edge-gated).

## Gotchas

Non-obvious things that bit us. Each entry earned its place by causing a real problem.

- **The equity curve runs on 1-day-forward returns, not the N-day classification target.** Compounding the N-day return on every daily row counts the same window ~N times and inflates final equity by an order of magnitude. See `packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline` (`daily_fwd_return` is the equity input; `fwd_clean` is the classifier target only) and `packages/lidr_core/tests/test_strategy_returns.py`, which is what makes sure the bug stays fixed.
- **Python 3.14 has no parquet wheels yet.** `lidr_core/data/loaders.py` caches OHLCV in pickle, not parquet. Don't "fix" the loader by switching back to parquet without verifying `pyarrow`/`fastparquet` wheels exist for the target Python — the comment at `loaders.py:50` documents why.
- **Typer collapses single-command apps into a flat CLI.** Removing `list-signals` from `ta_ensemble/cli.py` would silently break `python -m ta_ensemble backtest <config>` (Typer would re-flatten the entry point). Don't remove `list-signals` until a third real command lands.
- **`yfinance` only has currently-listed tickers** — survivorship bias. Fine for SPY/QQQ/sector ETFs; suspicious for individual-name backtests. Don't trust individual-stock results without a CRSP-style source.
- **`class_weight="balanced"` in the baseline is actively harmful, not just "distorts probabilities."** Empirically (2026-05-27 sanity check, see Recent Changes): removing it on the six-signal logistic config shrank the distance from no-skill baseline ~7× (`skill_score` -0.0374 → -0.0051). The reweighting was forcing confidently-wrong predictions on a 60/40 problem. Removing it makes the model "predict ≈base_rate every day" (a no-skill baseline), which is mechanically aggregate-calibrated but useless to lidr because there's no day-to-day variation. **Default new configs to `class_weight=None`**; revisit only if a future model class shows genuine class-balance trouble. Calibration via Platt/isotonic is still needed for shipping any probability artifact to lidr (on the roadmap: calibration).
- **`artifacts/results_log.csv` rows from before 2026-05-27 are slightly off.** The expanding-window backtester used an inclusive right endpoint on the test slice, so the boundary date between split N and split N+1 (`test_end` of N == `test_start` of N+1) was predicted twice and double-compounded in the equity curve. ~0.3% of rows affected; tiny effect on every metric (`accuracy`, `log_loss`, `skill_score`, `cagr`, `sharpe`, `n_oos`). Fixed by making the right endpoint exclusive except on the final split — see `lidr_core/backtest/engine.py::expanding_window_backtest` and `packages/lidr_core/tests/test_backtest_engine.py`. Pre-fix rows weren't re-run; treat any cross-row comparison that straddles 2026-05-27 with this in mind.

## Next Up

Priority order, framed around a single near-term goal: **prove the model has an edge over buy-and-hold** before building any serving/integration plumbing. As of 2026-05-27, neither logistic nor LightGBM beats no-skill on the six TA signals → 5d-forward-return-sign target (see Recent Changes → LightGBM checkpoint). The diagnostic finding was that the model class is *not* the bottleneck — three independent well-behaved configs (unweighted logistic, tiny LightGBM, calibrated LightGBM) all cluster at the no-skill floor. So the next move is to attack the target/feature setup, not try more models.

Cross-references to Next Up items use names, not numbers — see Maintenance Instructions for why. Completed items are removed from this section and live in Recent Changes instead.

1. **Reformulate the target or features.** The LightGBM result confirmed the bottleneck is here, not in model capacity. Three concrete directions, ordered roughly by cost-to-test:
   - **(a) Longer prediction horizon** (5d → 20d). Cheapest: change `target.horizon_days` in a config. The 5-day-sign target is extremely noisy — most weekly moves are noise. A monthly horizon should have a higher signal-to-noise ratio, and the existing TA signals (RSI, MACD, Bollinger) are arguably better matched to weekly/monthly motion than daily.
   - **(b) Return-magnitude regression target** instead of binary sign. Larger change: requires a new target type in `pipeline.py`, a regressor instead of a classifier, and rethinking `Model` protocol return shape. Lets the model express *how confident* and *how much*, which carries more information than sign alone.
   - **(c) Regime features.** VIX level, yield-curve slope, 60-day realized vol. New feature axes that don't overlap with the six TA signals. Needs multi-ticker support (`^VIX`, `^TNX`) wired into the data loader; partly implementable but not yet exercised.
2. **Introduce stacking.** Once two base learners exist *with non-zero skill*, add a `StackedModel` whose `fit` trains the base learners via out-of-fold predictions and then trains a meta-learner (logistic regression) on top. **Currently parked** — neither logistic nor LightGBM has skill, so a stacker over them inherits no signal. Revisit after item #1 produces a config that clears the no-skill floor.

— Edge gate: items below stay parked until something above beats buy-and-hold. —

3. **Add a final-model fit + serialize step.** The backtest only evaluates (throwaway per-split models); nothing fits a model on all data or writes `artifacts/models/`. Before serving live predictions, fit one model on all available history, serialize it, and write the prediction artifact. Trigger: once a model beats buy-and-hold.
4. **Calibrate `predict_proba` via Platt or isotonic regression.** *Empirically validated as a needed step by the LightGBM PR diagnostics (2026-05-27):* wrapping LightGBM in `CalibratedClassifierCV(isotonic, cv=3)` moved its skill_score from -0.148 to -0.004. Raw LightGBM probabilities are confidently miscalibrated; calibration is required before any probability artifact ships to lidr. Logistic regression's `predict_proba` is closer to calibrated out of the box but should also be wrapped for consistency.
5. **Migrate the output from binary to 3-class (BUY / HOLD / SELL).** Today's target is binary (`fwd_return > 0`). The project's stated deliverable is 3-class. Two paths: (a) post-hoc bucketing of the calibrated probability into BUY / HOLD / SELL bands, (b) reformulate the target itself (e.g., three quantile bins of `fwd_return`). Decision punt until a binary model with edge exists; the 3-class formulation changes how "beats buy-and-hold" is measured.
6. **Evolve the artifact JSON schema as lidr's needs firm up.** Schema is formalized at `schema_version: 2` in `packages/lidr_core/src/lidr_core/contract/schema/artifact.schema.json` and produced via `lidr_core.contract.writer.build_artifact` / `write_artifact`. Fields: `schema_version`, `model_id`, `model_version`, `config_name`, `ticker`, `generated_at`, `metrics.{classification,strategy,benchmark}`, `predictions[].{date,recommendation,probability_up,y_pred,y_true}`. Bump the version on a *breaking* change (removing/renaming a field, changing a type); additive optional fields do not need a bump. Likely additions before lidr starts consuming: per-signal/feature contributions for explainability.
7. **Wire lidr's `/api/signals/[ticker]` to read the artifact.** That's the bridge moment. Coordinate with the lidr CLAUDE.md.
8. **Wire up MLflow for experiment tracking.** Replace the timestamped-folder report + CSV log with proper logged runs and a comparison UI. *Deferred: the existing `artifacts/results_log.csv` covers the "did this help?" need until the run count is large enough to justify the heavier tooling.*

## Active Task

**Plan revised 2026-05-28.** PR-A merged (commit `b9ce76a`). The original PR-B plan assumed Reddit (PRAW) + Tiingo News at $10/mo; credential setup found Reddit is blocked by the **Responsible Builder Policy** and Tiingo is $30/mo with only 3 months of news history at that tier. Full revised data-source picture in [`docs/research/data-sources.md`](docs/research/data-sources.md). Boon approved Path 4 (rewire PR-B around the validated stack: EODHD $19.99/mo replacing Tiingo, Apewisdom replacing Reddit's live role, Finnhub free in the backbone, Reddit + Google Trends out as permanent stubs).

**Sequenced as:** (1) **horizon-spike on TA model first** (cheap, free, ~1–2 days — settles the 5d-vs-20d target-noise question that directly shapes `news_v0.yaml`'s horizon choice). (2) **Revised PR-B** wires the new data sources + lights up FinBERT + LLM. (3) **PR-C** runs the real `news_v0.yaml` backtest + comparison.

### Step 1 — Horizon spike (active next)

Run target-horizon sweep on the existing six-signal TA model to settle whether the 5d-sign target's noise floor is itself the bottleneck. If 20d helps meaningfully, design `news_v0.yaml` against the same horizon. Free, doesn't need creds.

Concrete: clone `packages/ta_ensemble/configs/baseline_six_signals_unweighted.yaml` → variants at `target.horizon_days: 1, 5, 10, 20, 60`. Do the same for the LightGBM config. Run all, append rows to `results_log.csv`, plot **skill_score vs horizon** for both model classes + per-period strategy returns. PR with the chart + per-period table as evidence per the outcome-changing-PR convention.

**Claude Code kickoff prompt (horizon spike):**

```
Read CLAUDE.md, then the Active Task → "Step 1 — Horizon spike" section.

Execute the horizon spike (Next Up: target/feature reformulation, direction (a)).
This is a free, no-credentials task that runs entirely on existing TA infrastructure.

Steps:
1. In a new branch, clone baseline_six_signals_unweighted.yaml and the LightGBM
   config to variants with target.horizon_days in {1, 5, 10, 20, 60} — 10 configs total.
2. Run each. Each run appends a row to artifacts/results_log.csv with the existing
   per-period breakdown.
3. Build the evidence chart: skill_score vs horizon for both model classes (10 points),
   plus a per-period strategy returns table for the most interesting horizons.
4. PR with the chart + table as PR-evidence per the outcome-changing-PR convention
   (scripts/verify_*.py → docs/_pr_evidence/horizon_sweep/ → removed in cleanup commit).
5. Document the finding in this Active Task section (which horizon won, by how much,
   what it implies for news_v0.yaml). Do NOT start the revised PR-B.

Follow protected-main PR workflow; CI green is required.
```

### Step 2 — Revised PR-B (after horizon spike merges)

Full spec + Claude Code kickoff prompt in [`docs/plans/task-2-news-sentiment-model.md`](docs/plans/task-2-news-sentiment-model.md). Headline changes from the original PR-B plan: **delete** `tiingo.py`; **convert to permanent stubs** `reddit.py` (Responsible Builder Policy) and `google_trends.py` (pytrends archived 2025-04-17); **add real adapters** `finnhub.py` (free, US company news 1yr + real-time), `apewisdom.py` (free, retail-attention live), `eodhd.py` (paid, $19.99/mo, historical news + per-article sentiment); **drop the Quiver Quant idea from PR-B** — defer to optional later if news side shows promise. FinBERT + live LLM scoring still light up in this PR through PR-A's existing cost-control harness.

### Step 3 — PR-C (after revised PR-B merges)

`news_v0.yaml` real backtest on a 5–10 ticker universe using the rewired stack and `target.horizon_days` set per the horizon-spike outcome; `results_log` row + refreshed `manifest.json` + comparison chart + per-period table per the outcome-changing-PR evidence convention. Deletes [`docs/plans/task-2-news-sentiment-model.md`](docs/plans/task-2-news-sentiment-model.md) in the cleanup commit per the plan's DoD. The ADR and `docs/research/data-sources.md` stay as the durable record.

<!-- Update this section when work is in progress. Replace with `_Nothing currently in-flight._`
     when paused. Keep it short: what's being built, where it was left off, mid-flight decisions. -->

## Recent Changes

### 2026-05-28 — CLAUDE.md maintenance: fold oldest 5 entries + restore missing Task 1 header (docs only)

Maintenance pass acting on the file's own Maintenance Instructions. Two clear issues addressed:

**(1) Recent Changes exceeded the ≤10-entry rule.** Count had grown to 11 with a 12th about to land (this entry). Folded the oldest 5 entries (Drift-fix Batch 1 of 6, Drift-fix Batch 6 of 6 / README split, Five lidr signals + PR-evidence convention, Signals explainer doc, Dup-date fix) into a single new sub-section in Archived Summary: "Six-signal TA model build-out + CLAUDE.md drift-fix arc (2026-05-26 → 2026-05-27)". Decisions and rationale preserved (PR-evidence cleanup mechanics, per-signal docs template, ACCURACY_CASES 5-tuple schema, dup-date Gotcha caveat, rtol=1e-8 tolerance rationale); narratives compressed roughly 5× by dropping the per-PR commentary and step-by-step diagnostics that have since been absorbed into Conventions / Gotchas / per-signal source files.

**(2) Task 1 monorepo-restructure entry was unheadered.** The "Mechanical, no-behavior-change restructure that executes the plan from the 2026-05-27 planning session..." paragraph (the PR #23 entry) had lost its `### YYYY-MM-DD — title` header and sat orphaned between the 2026-05-28 Task 2 PR-A entry and the 2026-05-27 Planning entry. Added `### 2026-05-27 — Task 1: monorepo restructure into lidr_core / ta_ensemble / news_sentiment + schema v2 (PR #23)`, matching the file's header style.

Deferred from this pass: (a) verify whether "20 new tests added; full suite is now 44" in the PR-A entry is the pytest-collected count or stale (current grep shows 27 `def test_` definitions, plausibly 44 once parametrized — low-stakes, needs `pytest --collect-only` in a configured env to confirm); (b) when revised PR-B lands, the Stack section's `EDGAR, GDELT, Tiingo all go through it` line will need the Tiingo deletion reflected — worth flagging in the PR-B kickoff prompt next time that doc is touched.

### 2026-05-28 — Plan revision: rewire PR-B around validated data sources (docs only, no code)

Planning-only session in Cowork *after* PR-A merged, driven by two findings during credential setup that contradicted the original plan's assumptions: **(1) Reddit is effectively blocked** — Reddit's Responsible Builder Policy (verified directly at support.reddithelp.com/.../42728983564564) requires explicit moderation or accredited-academic justification for any Data API access; personal research collection no longer qualifies. **(2) Tiingo is $30/mo with only 3 months of news history at that tier** (verified from `tiingo.com/about/pricing`) — the original `~$10/mo + ~10yr history` thesis turned out wrong on both numbers; the 15-yr archive is sales-quoted enterprise. Also re-verified: **pytrends was archived 2025-04-17** so Google Trends as a free adapter is dead too.

Worked through three alternatives with Boon: (a) pay $30 Tiingo as planned, (b) skip Tiingo and run free-only, (c) pivot to Next Up #1 (target/feature reformulation) and defer Task 2. None was right. Picked **Path 4 — rewire PR-B around the validated stack**: drop Reddit, Tiingo, Google Trends; add **EODHD News + Calendar at $19.99/mo** as the historical-news source (multi-year history + per-article sentiment scores included — better than Tiingo at lower price), **Apewisdom (free)** as the live retail-attention replacement for Reddit (they've already done Reddit's policy work, we consume aggregates), and **Finnhub (free tier)** promoted to backbone (1yr US company news + real-time at 60/min). **Optional later:** Quiver Quantitative Hobbyist $30/mo for historical WSB mentions back to 2018 — defer until news side shows promise. Net cost: $19.99/mo (vs original budget ~$10/mo, vs the $30/mo Tiingo shock we avoided).

**Sequencing.** Boon also approved doing **Next Up #1's horizon spike first** (free, ~1–2 days) before revised PR-B — the 5d-vs-20d target-noise question directly shapes how `news_v0.yaml`'s `target.horizon_days` should be set, and it's cheap to settle. Horizon-spike kickoff prompt embedded in the Active Task section; revised PR-B kickoff prompt embedded in `docs/plans/task-2-news-sentiment-model.md`.

**Files updated in this session.** [`docs/research/data-sources.md`](docs/research/data-sources.md) — major rewrite of the verification + paid-options tables to reflect the validated picture; added a dated 2026-05-28 decision-log entry preserving the 05-27 entry below it. [`docs/plans/task-2-news-sentiment-model.md`](docs/plans/task-2-news-sentiment-model.md) — restructured around revised PR-B and PR-C (PR-A is shipped); embedded the code-level instructions Claude Code will execute (delete `tiingo.py`, convert `reddit.py`+`google_trends.py` to permanent stubs, add three new adapters, update pyproject extras). This file (CLAUDE.md) — Active Task section now carries horizon-spike-first sequencing + the horizon-spike kickoff prompt, with PR-B and PR-C as Steps 2 and 3. No code touched; all changes execute in Claude Code per the PR workflow.

### 2026-05-28 — Task 2 PR-A: news_sentiment scaffolding + Phase 0 free adapters + offline dev path

Filled in the `news_sentiment` package shell from Task 1 with the structural pieces the rest of Task 2 will bolt logic onto. **No model edge claim**; this PR is the harness that PR-B and PR-C will produce real results through.

**The slice.** Task 2's DoD ("`news_v0.yaml` produces a validated artifact + comparison evidence chart") needs paid Tiingo News, an Anthropic key, Reddit app credentials, and a 440MB FinBERT model download — none of which are available in this session. Confirmed up-front with Boon: ship PR-A only, defer PR-B (paid + heavy scorers) and PR-C (real backtest + comparison) until creds are set up. This means PR-A is *intentionally* structural — every file would be in a one-shot version of Task 2 anyway, but the lines that hit external services raise with the next-PR's required wiring spelled out in the error message rather than failing silently.

**Package shape (~25 new files).** Mirrors `ta_ensemble` so the two models are strictly comparable through `results_log` and the schema-v2 artifact contract. Top-level `pipeline.py` reuses `lidr_core` for backtest, eval, results_log, leaderboard, and contract; the only model-specific pieces are `datasources/`, `ingest/collector.py`, `scoring/`, and `features/`. CLI is `python -m news_sentiment {backtest, list-features, list-sources}`; `Makefile` got a `backtest-news` target.

**Data sources.** Six adapters registered (`synthetic`, `edgar`, `gdelt`, `reddit`, `google_trends`, `tiingo`); five real, one stub. Pattern: every adapter subclasses `BaseNewsSource` and overrides `fetch_raw(ticker, start, end) -> list[NewsItem]`; the base class enforces the `[start, end)` window and sorts by `published_at`. Heavy deps (`praw`, `pytrends`) are lazy-imported inside `fetch_raw` so `pip install -e ./packages/news_sentiment` works without them and only the adapters you actually call need their extras (`[reddit]`, `[trends]`, `[scoring]`, `[llm]`). `TiingoSource.fetch_raw` raises `NotImplementedError("…PR-B…")` with the four steps the next PR needs to land — preferable to a silent no-op that would let a config typo here go unnoticed.

**Collector.** `ingest/collector.py::collect(sources, ticker, start, end, cache_dir)` fans out to each adapter, dedups across sources by `content_hash` (SHA-1 of `source|url|title|body`), and persists raw items to one JSONL file per `(ticker, source)` in `data/news/<config>/<ticker>__<source>.jsonl`. **Point-in-time discipline lives at the cache layer**: items are stored with their `published_at` (true publish timestamp), not the ingestion time, so re-runs can't quietly revise history. The data clock starts whenever PR-B lights up the real adapters — Reddit history can't be backfilled (Pushshift died in 2023), every day of delay is lost training data. `.gitignore` covers `data/news/*` so the smoke run doesn't bloat the repo.

**Scoring.** `LexiconScorer` (Loughran-McDonald-style word counts against a small embedded list) is the only working scorer in PR-A — deterministic, zero-dep, good enough for the dev path. `FinBertScorer` raises until PR-B (instructions baked into the error). `LlmScorer` is **half-real**: the cache (content-hash keyed, on-disk JSONL surviving process restarts), the per-run budget cap (`max_calls`, `max_usd`, raises `LlmBudgetExceeded`), and the spend log (`artifacts/llm_spend.csv`, append-only with token counts + estimated USD) are all functioning PR-A code; only the live Anthropic call is stubbed. Tests in `test_scoring.py` pin that a cache hit consumes zero budget and that the budget cap actually trips. That way PR-B drops the live call into a known-good cost-control harness rather than reinventing one.

**Features.** Three registered + the `_common.items_to_daily` + `align_to_trading_days` helper that mechanises the point-in-time rule. Pattern: items are first daily-aggregated by `published_at.normalize()`, then the aligned trading-day series is **shifted forward by one trading day** so a feature value at trading day `t` can only depend on items published strictly before the start of day `t`. The lookahead test asserts this by re-running each feature on items truncated at the check date — same shape as `ta_ensemble`'s `test_no_lookahead.py`. Features: `sentiment_level` (trailing-window mean of weighted sentiment), `sentiment_momentum` (fast MA minus slow MA), `abnormal_mention_volume` (z-score of daily item count vs 30-day baseline). Each emits a raw continuous quantity — the model learns thresholds, same philosophy as the TA signals.

**Tests.** 20 new tests added; full suite is now 44, all green; lint clean. Smoke test runs the full pipeline on synthetic prices + synthetic news, validates the produced schema-v2 artifact via `lidr_core.contract.loader.load_artifact`, and asserts `output.results_log: false` keeps `artifacts/results_log.csv` from growing (same guard ta_ensemble uses). Lookahead test parametrized over all three news features. Feature spot checks pin the math + the one-day PIT shift (e.g. "one +1 item on day 0 must show at day 1 with lookback=1, then fall out at day 2").

**One bug caught during testing.** First version of `align_to_trading_days` returned `aligned.shift(1)` directly, so position 0 of the trading-day index was NaN, breaking `test_sentiment_level_picks_up_one_day_shift`. Right fix is `shift(1).fillna(0.0)` because semantically "nothing visible yet at the first trading day" should produce 0.0 (same value a no-news day produces), not NaN. Without that, every feature would propagate NaN through the first row of the feature matrix and `dropna` would silently lose the first trading day forever.

**Plan doc stays alive.** `docs/plans/task-2-news-sentiment-model.md` is not deleted in this PR — Task 2's DoD ("delete the plan once the checkpoint merges") refers to PR-C. The CLAUDE.md Active Task section now carries the durable framing (the three-PR slice + what each PR ships); the plan doc keeps the per-phase build order.

### 2026-05-27 — Task 1: monorepo restructure into lidr_core / ta_ensemble / news_sentiment + schema v2 (PR #23)

Mechanical, no-behavior-change restructure that executes the plan from the 2026-05-27 planning session. `src/lidr_ml/` is gone; the code now lives in three packages under `packages/`:

- **`lidr_core`** — the shared harness: backtest engine, eval (metrics, report, results_log, the new `leaderboard.py`), data loaders, generic learners (logistic + LightGBM), and the new `contract/` (artifact JSON Schema + writer + loader + the new `Feature` / `DataSource` protocols).
- **`ta_ensemble`** — the six TA signals + their pipeline + their configs. Today's only complete model. Depends on `lidr_core`; CLI is now `python -m ta_ensemble backtest <config>`.
- **`news_sentiment`** — empty shell, README points at the Task 2 plan.

Imports updated repo-wide (no `lidr_ml.*` left). All moves used `git mv` so file history is preserved. Each package owns its own `pyproject.toml`; root `pyproject.toml` is now dev-tool-only. `make install` installs all three editable. CI workflow updated to do the same. Configs gain `model_id: ta_ensemble` + `model_version` so produced artifacts identify their model family.

**Artifact contract formalized at `schema_version: 2`** in `lidr_core/contract/schema/artifact.schema.json`. v1's implicit dict is replaced by `build_artifact` + `write_artifact` which validates against the schema before writing (via `jsonschema`; falls back to a narrow in-tree check if not installed). Predictions now land under `artifacts/predictions/<model_id>/` so multiple models don't collide, and a top-level `artifacts/manifest.json` leaderboard is generated by `leaderboard.write_manifest` (picks the latest artifact per model by mtime, not lex filename order — a `dev_synthetic` file sorts after `baseline_*` alphabetically and the lex version silently picked the wrong "latest").

**Parity gate passed.** Re-ran `baseline_six_signals_unweighted.yaml` before (run_id `20260527-183934`) and after (run_id `20260527-190244`) the moves: `skill_score = -0.005104`, `cagr = 0.142454`, `n_oos = 3851`, and every other metric column in `results_log.csv` matches bit-for-bit between the two rows. The relocation didn't perturb the pipeline. Tests: 24 / 24 passing; lint clean across the three packages.

**Workflow notes worth keeping.** The plan doc `docs/plans/task-1-repo-restructure.md` was deleted as the final step of this PR (mirrors the PR-evidence cleanup habit — squash-merge collapses the add+delete pair so `main` stays free of execution-plan churn); the durable docs (ADR 0001, `docs/research/data-sources.md`, Task 2 plan) stay. The GitHub repo *URL* is still `pavarit/lidr-ml`; renaming it is a UI-only step that can be done later without touching the code.

### 2026-05-27 — Planning: multi-model architecture + news-sentiment model (docs only, no code)

Planning-only session in Cowork. The project framework firmed up: `lidr` is the front-end; multiple *competing* models will feed it recommendations through the JSON artifact. Decided to reorganize around the contract rather than around models — rename `lidr-ml` → `lidr-models` and split it into a `lidr_core` shared harness (backtest, eval, results_log, the formalized artifact contract, and the Signal/Model/Feature/DataSource protocols) plus per-model packages (`ta_ensemble` = today's six-signal pipeline; `news_sentiment` = the new model). Rationale, alternatives, and the designed-for-change requirements (swappable data sources / features / model; easy iterate-and-compare loop) are in `docs/adr/0001-multi-model-repo-architecture.md`.

Wrote four planning docs under `docs/` (adr/, research/, plans/) — see the Active Task section for the index. No code moved; the restructure and the model build are handed off to Claude Code as Task 1 (mechanical, parity-gated restructure) and Task 2 (news-sentiment model, blocked by Task 1), each with a kickoff prompt embedded in its plan doc. Doc hygiene by design: the **ADR** (which now also holds the artifact-contract schema-v2 design, folded in from a separate doc) and **`docs/research/data-sources.md`** are durable knowledge and stay; the two **`docs/plans/` docs are disposable** and instruct Claude Code to delete themselves in the cleanup commit once their task merges (mirroring the existing PR-evidence cleanup habit), so the repo doesn't accumulate stale execution plans. Also captured verified news/sentiment data-source research (free-tier status + paid pricing + leverage call; decision to trial Tiingo News at ~$10/mo) in `docs/research/data-sources.md` so it's not re-derived later. When Task 1 executes, this repo's folder map, Stack, and Conventions sections will need updating to match the new monorepo layout.

### 2026-05-27 — LightGBM checkpoint: still no edge, and the model class is not the bottleneck

Shipped LightGBM as the second base learner ([PR #20](https://github.com/pavarit/lidr-models/pull/20), commit `c4f0044`). New module `src/lidr_ml/models/lightgbm.py`, registered as `"lightgbm"` in the model registry, configured via `configs/baseline_six_signals_lightgbm.yaml` (identical to `baseline_six_signals_unweighted.yaml` except `model.type`). Conservative defaults: `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`. New row in `artifacts/results_log.csv` at `run_id=20260527-143507`.

**Headline: LightGBM is worse than logistic on this problem.** `skill_score = -0.1478` vs the unweighted logistic's -0.005. Strategy CAGR 9.3% vs B&H 14.0%.

**The diagnostics reframed the headline.** Wrapping LightGBM in `CalibratedClassifierCV(method='isotonic', cv=3)` moves skill_score from -0.148 to -0.004 — back to the no-skill floor. So LightGBM's raw predictions aren't anti-informative, they're badly miscalibrated. **The calibration step in Next Up is now empirically validated as needed** before any prediction artifact ships to lidr (was previously a theoretical concern; now there's a 0.14-point skill_score delta backing it up).

**The deeper finding strengthened.** Three independent well-behaved configs converge on the same answer:

| config | skill_score |
| --- | --- |
| six_signals_unweighted (logistic) | -0.005 |
| LightGBM tiny (4 leaves, 20 trees, min_child=200) | -0.007 |
| LightGBM default + isotonic calibration | -0.004 |

When given enough freedom *and* calibrated probabilities, the model learns to predict the prior. There's no day-to-day signal in these six features against the 5-day-forward-return-sign target. **The bottleneck is the features/target, not the model class.** Roadmap pivoted: Next Up #1 was "LightGBM" (this entry replaces it); the new #1 is target/feature reformulation. Stacking (was #2) is parked under the edge gate — a stacker over two no-skill base learners inherits no signal.

**Diagnostic checks worth keeping in mind for future PRs.**
1. **Column-order spot check** (`classes_ = [0,1]` + in-sample P(class=1) higher on up-days than down-days) — cheap rule-out for "are we silently reading P(down) as P(up)?". Trivially passes for sklearn-conforming models but takes 5 lines to verify; the cost of skipping is silent inversion of every conclusion.
2. **In-sample fit check** — accuracy on the model's own training data must be meaningfully above base rate, else the model isn't fitting at all and the OOS collapse is a fit failure not a generalization failure.
3. **Seed-stability sweep** — for LightGBM with default settings, this is *trivial* (all seeds bit-identical) because `feature_fraction = bagging_fraction = 1.0` means there's no randomness consuming `random_state`. To get real seed-stability evidence need to enable subsampling first. Hyperparameter sweep is the stronger robustness check.
4. **Hyperparameter sensitivity** — for a low-SNR problem, expect monotonic behavior in capacity: tiny config → no-skill floor; default → confidently wrong; large → even more confidently wrong. If the relationship isn't monotonic, the result is hyperparameter-noisy.
5. **Calibration wrapper** — for any tree ensemble producing `predict_proba`, run with and without `CalibratedClassifierCV`. The delta is large; without it, log-loss-based conclusions about "the model fits noise" are conflated with "the model has miscalibrated probabilities."

**Per-period breakdown surfaced a regime story the full-window aggregate hides.** LightGBM is the *worst* strategy in 2024 (+14% vs B&H +26%) but the *best* in 2025 (+22% vs +17%) and Q1 2026 (+0.0% vs -4.5%). Per-period skill_score confirms it's not skill — Q1 2026 LightGBM (-0.192) is essentially tied with unweighted logistic (-0.194). LightGBM has a structural bias toward sometimes-cash positions: helps when down/choppy, hurts when up. Not signal.

**Workflow lesson reinforced (third time now).** First draft of the per-period "reading this" narrative was written from memory, *before* I generated the per-period numbers, and contradicted them on 2025 and Q1 2026. Caught it during review and rewrote from the actual table. Same lesson as the six-signal PR (#15) and the chart-vs-log cross-check from the verify-script bug. **Never write interpretation before you've generated the data it interprets, and sanity-check chart numbers against the logged metrics.**

**Pacing observation worth keeping.** Reviewer pushback ("are you sure?", "how do I trust this?", "add the logistic lines and per-period table") was responsible for *all* of the most valuable framing changes — the diagnostic suite, the calibration reframe, the per-period regime story. Headline framing pre-pushback was "LightGBM fits noise"; post-pushback was "no model has signal because the features/target is the bottleneck." Different conclusions, different next moves. Suggests: when reporting a negative result, build in at least one round of "what would convince me this is wrong?" before declaring done.

### 2026-05-27 — `class_weight=None` sanity check before LightGBM

Quick sanity check motivated by the Gotcha bullet about `class_weight=balanced` distorting `predict_proba`. New config `configs/baseline_six_signals_unweighted.yaml` (identical to `baseline_six_signals.yaml` except `class_weight` is omitted → sklearn default `None`). New row in `artifacts/results_log.csv` at `run_id=20260527-134952`.

**Surprise finding: `class_weight=balanced` was actively harmful, not just probability-distorting.** Removing it shrank the distance from the no-skill baseline ~7×: `skill_score` -0.0374 → -0.0051. Pre-experiment prediction was "almost certainly won't move skill_score" — wrong. The reweighting was forcing confidently-wrong predictions on a 60/40 problem.

**But it's still not skill.** The unweighted model has `pred_rate = 0.996` (long virtually every day), `mean P(up) = 0.590`, and a P(up) distribution that's a ~0.10-wide spike entirely above 0.5 (max value 0.70). Accuracy 0.610 ≈ `base_rate` 0.611 because the model is essentially "predict the unconditional base rate every day" — a constant predictor that mechanically matches the no-skill baseline.

**The strategy CAGR 14.25% vs benchmark 14.04% (excess +0.20%) is noise, not edge.** Three independent reasons it can't be skill:
1. **`skill_score = -0.005` is negative.** The probability predictions themselves carry no information about outcomes — strictly worse than just predicting `base_rate`. Whatever the strategy gains, it can't be from informative predictions.
2. **The "excess" comes from ~15 lucky cash days.** Pred_rate = 0.996 means the strategy is in cash on 0.4% × 3,851 OOS days ≈ 15 days total. By chance, those 15 happened to be slightly negative. A 20 bps/year difference is well inside the noise band of any sensible significance test (SPY daily-return std ≈ 1%, annual std ≈ 16% — t-stat of the excess is ~0).
3. **Per-period evidence kills it definitively.** Per the PR #20 per-period table: in 2024, 2025, and Q1 2026 the unweighted-logistic strategy returns are *bit-identical* to B&H (the model is long every day in those windows). The full-window 20 bps came entirely from earlier-year cash-day luck and is unrepeatable.

**Functionally equivalent to no model at all.** An intercept-only logistic (zero features) would predict P(up) ≈ base_rate every day and produce the same strategy. The six features are non-load-bearing in this configuration. The full-window "+0.20 pp excess" line in `results_log.csv` is technically true but should not be read as edge.

So: `class_weight=balanced` was forcing confidently-wrong predictions (anti-informative); removing it lets the model collapse to no-skill baseline; the linear-features-via-logistic setup still can't extract day-to-day signal from these six features. **Recommendation for LightGBM stands stronger.** The failure mode is now clearly about the linear model's inability to express feature interactions, not class balance.

**Gotcha bullet strengthened.** Was "distorts predict_proba" (descriptive); now "actively harmful, default `class_weight=None`" with concrete skill_score numbers. Empirical updates that contradict a Gotcha's claim are the right time to revise it in place.

**Dup-date fix verified benign on headline metrics.** Re-ran `baseline_six_signals.yaml` post-fix at `run_id=20260527-135054` → `skill_score = -0.0374` (same as the pre-fix row `20260527-120203` to 4 decimals). The dup-date Gotcha's "below the noise floor" claim is now empirically confirmed for this config, not just estimated. Both rows kept in results_log as a reference comparison.

### 2026-05-27 — Six-signal logistic baseline checkpoint (edge gate stays closed)

Ran `baseline_six_signals.yaml` — same target/model/backtest/costs as `baseline_v1`, only the feature set changed (added rsi, macd, bollinger, breakout, volume to sma_crossover). New row appended to `artifacts/results_log.csv` at `run_id=20260527-120203`. OOS: 2010-12-30 → 2026-04-23, 3,862 days (later start than baseline_v1 because of the 252-day breakout warmup, so only `excess_*` is directly comparable across the two full-window rows). The row was generated *before* the dup-date fix landed (same day, PR #16) so it carries the same ~0.3% double-counting bias as other pre-fix rows — see Gotchas.

**Verdict: edge gate stays closed.** Neither model beats buy-and-hold and neither has positive skill.

**Important reframe.** The initial read was "six features is worse than one feature" (lower accuracy, lower CAGR, lower final equity). That's mechanically true on the full window but **misleading on what the models actually do**:

| | baseline_v1 | baseline_six_signals |
|---|---|---|
| `skill_score` (log-loss vs no-skill floor) | -0.0378 | -0.0374 |
| `mean P(up)` | 0.503 | 0.505 |
| `pred_rate` (model says up) | 0.753 | 0.595 |
| `base_rate` (truth = up) | 0.613 | 0.611 |

In probability space the two models are essentially **equally non-skilled** — skill_score differs by 0.0004 over 3,800+ observations (sampling noise). Both are 0.5-spitters; 99% of six's probabilities fall in [0.40, 0.60]. The headline equity gap is **exposure (pred_rate), not skill**: baseline_v1 is long 75% of days vs six's 60%, and in a 14%-CAGR market more exposure compounds harder. The accuracy gap (-4.1 pp) is the same effect — in a market with `base_rate` ≈ 0.61, predicting "up" 75% of the time accidentally matches truth more often than predicting "up" 60% of the time.

**Recent windows favor six.** On 2025 calendar (n=250, same window for both), six **beats v1 decisively** — accuracy 0.548 vs 0.496, skill_score -0.053 vs -0.057, total return +15.2% vs +6.4% (excess vs B&H -3.0 pp vs -11.8 pp). On 2026 Q1 (n=61), six predicts up every day and **exactly matches B&H** (0% excess); v1 takes a few wrong cash days and loses 3.15 pp. baseline_v1's equity curve has long flatlines (cash for months at a time, including the entire post-April-2025 rally) that baseline_six does not.

**Conclusion: the bottleneck is the linear-model assumption, not the feature count.** Adding five orthogonal signals didn't move skill_score and made accuracy *worse* on the full window — but modestly improved recent-period behavior. Next move per the roadmap is LightGBM (now Active Task).

**Workflow lessons worth keeping.**
- **Don't conflate accuracy with skill when `base_rate` is far from 0.5.** Accuracy of 0.557 sounds OK until you notice base_rate is 0.613 (predict-always-up gets 0.613). `skill_score` (= 1 − log_loss / base_logloss) controls for this; report it next to accuracy whenever both are visible.
- **When a result is surprising or contradicts a prior, dig before publishing.** Initial PR framing led with "six is worse"; the user pushed back ("Are you sure?"), the diagnostic exposed the pred_rate/skill_score story, the PR was reframed before merge. The reframe was the more interesting finding.
- **Two charts beat one when the data spans years.** The full-window log chart compresses recent behavior into pixels; a 2024-onwards linear zoom (`docs/_pr_evidence/six_signal_baseline/chart_recent.png` at the time of merge, then removed in cleanup) is where the actual recent behavior is legible.
- **Aborted bug worth flagging.** The verify script's first version had `prices["Close"]` (capital C). The yfinance loader normalizes columns to lowercase (`loaders.py:77`); the script's fallback `prices.iloc[:, 0]` silently returned the `open` column, producing an equity curve that ended at ~0.6× when results_log said 2.47×. Caught by sanity-checking the chart against the logged final equity. Lesson: chart-vs-log cross-check is a fast first-pass test for any one-off verification script that re-derives metrics.

## Archived Summary

Older entries folded down per the Maintenance Instructions rule (Recent Changes exceeds 10 → fold oldest 5). Decisions and rationale preserved; narratives compressed. Sources: PRs and full entries in git history before commit `<this PR>`.

### Six-signal TA model build-out + CLAUDE.md drift-fix arc (2026-05-26 → 2026-05-27)

**Drift-fix Batch 1 of 6 (2026-05-26).** Drift audit between CLAUDE.md and code fixed several stale facts: protocol name `Signal` → `SignalFn`; Model protocol's three methods (`fit` / `predict_proba` → `np.ndarray` / `predict`); signal test table `SIGNALS` → `SIGNAL_CASES`; phantom `hit rate` removed from metrics.py walkthrough; backtest range "pre-2008" → "2005" to match `baseline.yaml`; Cowork-specific Gotchas dropped; Next Up #8 rewritten around already-implemented schema; calibration + 3-class items promoted to standalone roadmap entries; dead `output.report_html` field removed from configs; `strategy_metrics` empty-path return shape made consistent.

**Drift-fix Batch 6 of 6 — README as public face / CLAUDE.md as Claude-facing (2026-05-26).** Final batch of the drift-fix arc. README gained an Architecture section (ASCII pipeline diagram), Stack paragraph, real Project layout block, and a description of the HTML report. CLAUDE.md Commands trimmed to a one-line pointer at README's Quick Start + CLI sections. Net effect across the six commits: every CLAUDE.md claim matches code; every public contract (CLI, YAML schema, JSON artifact, results_log columns, license, contributing rules) documented in README/CONTRIBUTING; SPY baseline numbers in README trace to a real `results_log.csv` row.

**Five lidr signals ported + PR-evidence convention established (2026-05-27, PRs #5/#7/#8/#9/#10).** Shipped RSI, MACD, Bollinger, breakout, volume. Pipeline now exposes six features (sma_crossover + the five new ones). Each port matched lidr's TS implementation: RSI/MACD/breakout/volume exact bit-match; Bollinger 1.5e-11 (pandas Welford-style rolling std vs naive std). Features emitted are raw continuous quantities (not the TS's discrete BUY/SELL), so the model learns its own thresholds. Cross-signal validation: all five flagged early April 2025 SPY plunge as the most extreme bearish event in the 2023–2026 window — five independent measurements agreeing is strong correctness signal. **PR-evidence convention formalized** (now in Conventions): `scripts/verify_<thing>.py` → `docs/_pr_evidence/<thing>/{chart.png, evidence.md}`, embedded chart pinned to full-40-char commit SHA via `raw.githubusercontent.com`, removed in cleanup commit so `main` stays free of review artifacts while PR-comment chart URLs survive branch deletion. `ACCURACY_CASES` schema extended to 5-tuple with per-case `prices_factory` so each signal picks the fixture that makes hand-derivable spot checks possible. Tolerance loosened `rtol=1e-12` → `rtol=1e-8` to accommodate pandas' streaming rolling std (still ~1e5 tighter than any real-bug threshold). Spawned-task PR #6 added `output.results_log` config flag (default true; dev_synthetic sets false) + smoke-test assertion that the file size doesn't grow — stops smoke runs from polluting the tracked CSV. Test count 7 → 22.

**Signals explainer doc (2026-05-27, PR #13).** Shipped [`docs/signals.md`](docs/signals.md) — standalone first-time-reader explainer for all six signals with per-signal SPY charts in `docs/signals/`. Closes the gap that in-code docstrings explained *how* but never *why a signal exists* or *how to read it*. **Per-signal template** (apply when adding future signals): what it watches → what the number means (trend-following / mean-reversion / conviction family) → math table → SPY chart → recent history (3–4 named events) → failure modes → parameters used + why. Format choices kept: one combined file (not per-signal); throwaway top-level chart-generator script that gets deleted after committing PNG outputs (same spirit as the PR-evidence cleanup); full 2005–2026 history on log-scale so 2008/2020/2022/April-2025 all simultaneously visible; HTML `<img>` tags with `width="100%"` for cross-renderer compatibility.

**Dup-date fix in expanding-window backtest (2026-05-27, PR #16).** Caught while diagnosing the six-signal checkpoint: `expanding_window_backtest` had inclusive right endpoint on the test slice plus `train_end = test_end`, so the boundary date between split N and N+1 was predicted in **both** splits' output (~0.3% of rows; 11–12 dups per ~3,900-row config). Fix: right endpoint exclusive except on the final split. New regression tests `test_predictions_index_is_unique` + `test_consecutive_splits_do_not_overlap` (both fail against pre-fix engine); engine now also raises if `pd.concat(preds)` yields non-unique index, so the bug can't silently reappear. Pre-fix `results_log.csv` rows are very slightly off — documented in Gotchas; cross-row comparisons that straddle 2026-05-27 should be aware.

### Repo hygiene + workflow setup (2026-05-22 → 2026-05-26)

**Signal accuracy test harness + CI (2026-05-22).** Added `tests/test_signal_accuracy.py` — two-layer validation per signal: (1) element-wise comparison against an inline reference formula; (2) hand-derived spot checks on an arithmetic price series. `ACCURACY_CASES` table is the extension point when porting new signals. Extracted shared `synthetic_prices` fixture to `tests/conftest.py`. Added `.github/workflows/test.yml` (Python 3.11, runs `make test` + `make lint` on every push and PR). Pre-existing lint errors in `src/` cleaned up; lint is now clean baseline.

**Adopted Conventions + Gotchas sections; partitioned to one-fact-one-place (2026-05-26).** Two missing template sections added to CLAUDE.md: **Conventions** (procedural rules) and **Gotchas** (non-obvious things that bit us). Then repartitioned the file so each fact lives in exactly one section: **Conventions = the rule**, **Key Decisions = why we chose X over Y**, **Gotchas = what bit us**. Cross-references replace duplicates. Trimmed Key Decisions from 10 items to 5 (one Python project / source of truth in lidr / synthetic-data fallback / outputs are calibrated probabilities / backtest doesn't produce a deployable model). One-fact-one-place rule added to Maintenance Instructions.

**MIT → PolyForm Noncommercial 1.0.0 relicense (2026-05-26).** Both lidr-ml and lidr relicensed after explicitly working through "is MIT the right license?" — answer was no. Desired stance: anyone can read and learn, non-commercial use welcome without asking, commercial use requires permission. PolyForm Noncommercial is the lawyer-drafted license that fits this. Source-available, not OSI-open-source. Replaced LICENSE files, updated `pyproject.toml` / `package.json` metadata, swapped README badges, added "License of contributions" to CONTRIBUTING.md. **Note for the future:** anyone who grabbed either repo under MIT pre-2026-05-26 keeps MIT rights to *that snapshot*; exposure is nil today (no external adoption). Author can commercialize freely or dual-license to specific parties; relaxing back to a permissive license later would require contributor consent — easiest now while no external contributors exist.

**Adopted protected-main PR workflow (2026-05-26).** GitHub branch protection on `main`, all changes via PR, CI gate ("CI green" is the only required gate — no code-review requirement, sole developer). CONTRIBUTING.md gained a Workflow section with the full branch → PR → CI → merge → cleanup sequence. CLAUDE.md Conventions added a top-of-section bullet flagging protection. Settings live in GitHub UI, not in git.

**Documentation + tech-debt cleanup pass (2026-05-26).** Hygiene pass, no behavior changes. README "What's in the box" updated to reflect work that landed 2026-05-20 → 22; "What's next" rewritten around the **edge gate** framing (prove edge before serving/integration). CLAUDE.md Folder map reconciled with reality (added `eval/results_log.py`, `tests/conftest.py`, etc.; removed never-used `data/processed/`). De-stubbed code language in `pipeline.py` and `engine.py::add_strategy_returns`. Repo hygiene: deleted stray `pytest-cache-files-*` dirs, added `.gitignore` + `make clean` rules, added `make clean-reports`.

### Initial development phase (2026-05-19 → 2026-05-22)

**Bootstrap (2026-05-19).** Stood up the src-layout package, config-driven pipeline, expanding-window walk-forward backtest, HTML report, yfinance + synthetic data loaders, logistic-regression baseline, and an end-to-end smoke test on the synthetic config. Two early gotchas surfaced and remain documented in **Gotchas**: Typer collapses single-command apps into a flat CLI (the `list-signals` command exists to keep `backtest` as a subcommand — don't remove it until a third real command exists); and Python 3.14 lacks parquet wheels, so the OHLCV cache uses pickle.

**Equity curve made trustworthy (2026-05-20).** The original equity curve was compounding the N-day forward classification target on every daily row, which counts overlapping windows ~N times and inflated final equity by an order of magnitude. Fixed to mark to market with **1-day-forward returns**; the N-day return stays purely as the classification target. `tests/test_strategy_returns.py` was added at the same time and guards the rule (one regression test per failure mode: compounds each return once / cash days earn nothing / costs charged on position change). The same change added a Strategy-vs-Buy-and-Hold comparison to every report (additive to `metrics.benchmark` in the JSON artifact, `schema_version` unchanged) and a per-year breakdown table so regime-dependence is visible. Also hardened the JSON serializer to handle `np.bool_`.

**Report readability + first real SPY baseline (2026-05-21, AM).** The HTML report gained a Summary section translating the config into plain English plus a quick-reference table; `base_logloss` (the no-skill floor) was added next to log loss everywhere; comparison and per-year tables got green/red winner-highlighting (Strategy column, excess cells, per-year log-loss cells). End-to-end SPY baseline ran for the first time on real data: OOS 2010-10 → 2026-04, 3,914 days. Single-signal logistic loses to buy-and-hold on every axis — final equity 3.28× vs 8.23×, CAGR ~8.0% vs ~14.5%, Sharpe 0.67 vs 0.89, near-identical max drawdown (~−33.7%); accuracy 0.556 is below the base rate 0.613, log loss sits at the no-skill floor. **This is the floor every real model must clear** — the framing for the current "edge gate" in Next Up.

**Roadmap reset around the edge gate (2026-05-21, PM).** Reordered Next Up to serve one near-term goal: prove edge before building serving/integration plumbing. Promoted the cross-run results log to #1 (cheapest item, makes every later experiment measurable). Kept the feature → model build-out sequence (parity harness → port five signals → LightGBM → stacking → regime features) as the core edge-hunt, with an explicit note that re-running the logistic baseline on all six ported signals is a cheap checkpoint to tell whether the bottleneck is features or model. Added the **edge gate** line — items past it (final-model fit/serialize, artifact schema evolution, lidr wiring) stay parked until something beats buy-and-hold. Deferred MLflow to #10 since the CSV log covers the "did this help?" need until run volume justifies heavier tooling. Mirrored the framing in lidr's CLAUDE.md.

**Cross-run results log (Next Up #1 shipped, 2026-05-22).** New file `src/lidr_ml/eval/results_log.py` — `append_run()` appends one row to `artifacts/results_log.csv` after every backtest run. Columns include `run_id`, `config_name`, `ticker`, OOS span, `n_oos`, `skill_score` (= 1 − log_loss/base_logloss), full-OOS and per-period log losses with base-rate floors, strategy and benchmark CAGR/Sharpe/max-drawdown/final-equity, and excess. The file accumulates across sessions; `pipeline.py` calls `append_run` non-fatally (wrapped in try/except). Later PR #6 (2026-05-26) added the `output.results_log` opt-out so the dev_synthetic smoke test stops polluting the tracked CSV.

## Maintenance Instructions

If you (a future AI assistant joining this project) make meaningful changes, also update this file in the same session.

- **Keep evergreen sections current.** Project Goal, Architecture, Stack, How the pipeline works, Folder map, Conventions, Key Decisions, Gotchas, Next Up should reflect reality. If your work invalidates a fact in any of these sections, update it before ending the session.
- **Each fact lives in one section.** Conventions = the rule. Key Decisions = why we chose X over Y. Gotchas = what bit us. If you find yourself writing the same fact in two places, pick the canonical home and cross-reference from the other.
- **Append a dated entry to Recent Changes for each session that produces real changes.** Use a `### YYYY-MM-DD — short title` header followed by a paragraph describing what was done and why. Include decisions and rationale future-Claude would benefit from knowing.
- **When a Next Up item ships, remove it (don't strike it through) and renumber the rest.** Document the completion in Recent Changes — that's the single source of truth for "what's been done." Keeping completed items in Next Up duplicates information and makes the section grow monotonically. Cross-references to Next Up items elsewhere in this file should use **names** ("see roadmap: LightGBM"), not numbers, so renumbering doesn't break references. Historical references in Recent Changes (e.g., "Next Up #3 done") are fine to leave as-is — they describe state at the time.
- **Cross-link to lidr.** If a change here affects the integration with lidr (artifact format, signal parity, what the website should consume), update lidr's CLAUDE.md too in the same session.
- **Archive when Recent Changes exceeds 10 entries.** Fold the oldest 5 into a `## Archived Summary` section at the bottom. Preserve decisions and rationale; compress narratives, not insights.
- **Before declaring a session closed, ask the user if there's anything else.** After the last PR merges and the task list is clean, give a short 2–3 line session summary and explicitly ask "anything else before we wrap up?" Don't write a terminal-sounding summary that implicitly closes the conversation — users often surface important follow-up questions (test the conclusion harder, reframe a finding, spawn a follow-up PR) only *after* seeing the wrap-up. Cutting them off forces a new session to handle what could have been one more exchange. The closing question goes in the same message as the wrap-up; the form matters less than the *signal* that the conversation is still open.
