# CLAUDE.md

This file orients any AI assistant (Claude Code, Claude Cowork, etc.) joining this project. Read it before doing anything else. Then keep it current — see "Maintenance Instructions" at the bottom.

**Hosting repo**: https://github.com/pavarit/lidr-ml

## Project Goal

`lidr-ml` is the Python ML/backtesting pipeline that turns the technical signals from the sibling project [`lidr`](https://github.com/pavarit/lidr) into **empirically calibrated BUY / HOLD / SELL recommendations**. lidr's confidence values today are heuristics (normalized strength scores). This project's job is to replace them with probabilities learned from historical data — by training an ensemble model over the signals, backtested with walk-forward validation from pre-2008 to today.

The pipeline emits a versioned JSON artifact that lidr's `/api/signals/[ticker]` route can read directly. No running Python service is needed until lidr's FastAPI-service roadmap item is triggered.

## Architecture

Two-project setup, deliberately kept separate:

- **`lidr`** (Next.js, `C:\Users\smnk1\Claude\Projects\lidr`) — the website. Computes signals live in TS for display, will eventually overlay calibrated probabilities from the artifact this project produces.
- **`lidr-ml`** (this project, Python) — produces the model and the artifact. Iterates on its own cadence with its own tooling.

Integration is via a JSON file (`artifacts/predictions/<config>-<timestamp>.json`), not an HTTP service. Cheap, debuggable, version-controllable. A FastAPI service is on the lidr roadmap and will be added here when the lidr side is ready to consume live predictions.

## Stack

- **Python** ≥ 3.10
- **pandas / numpy** — data manipulation
- **scikit-learn** — base learners + walk-forward CV (`TimeSeriesSplit`)
- **yfinance** — free historical price data (with synthetic-data fallback for offline dev)
- **PyYAML** — config files
- **Typer** — CLI
- **Matplotlib** — chart embedded in HTML reports (base64, no internet needed to view)
- **pytest** — tests
- **ruff** — lint + format

Planned but not yet added: **LightGBM** (second base learner), **MLflow** (experiment tracking), **vectorbt** (faster backtest sweeps once we outgrow the custom engine).

## Commands

```bash
# One-time setup
make install                              # pip install -e .[dev]

# Run a backtest end-to-end (loads config, fits model, generates report)
make backtest CONFIG=configs/baseline.yaml

# Quick offline run using synthetic data (no internet needed)
make backtest CONFIG=configs/dev_synthetic.yaml

# Tests + lint
make test
make lint
```

The CLI is `python -m lidr_ml backtest <config>` if you prefer it directly.

## How the pipeline works

Reading top-down:

1. **Config** (`configs/*.yaml`) — declares what to run: tickers, date range, data source, which signals, which model, backtest method, transaction costs.
2. **Data loader** (`src/lidr_ml/data/loaders.py`) — pulls OHLCV from yfinance (or generates synthetic series for offline dev). Cached by `(ticker, start, end)` in `data/raw/`.
3. **Signals** (`src/lidr_ml/signals/`) — each signal is a pure function that takes a DataFrame of prices and returns a Series of feature values aligned to the price index. Every signal must be **lookahead-safe** (only uses data up to time *t* to compute the value at time *t*). Tested in `tests/test_no_lookahead.py`.
4. **Target** (computed inline in `src/lidr_ml/pipeline.py::run_pipeline`) — for now, binary: was the *N*-day forward return positive? Will become 3-class (BUY/HOLD/SELL) once we have a useful model.
5. **Backtest engine** (`src/lidr_ml/backtest/engine.py`) — expanding-window walk-forward. For each split: fit model on train slice, predict on test slice, store predictions. Never trains on data after the test period.
6. **Model** (`src/lidr_ml/models/`) — pluggable. Today: logistic regression. Next: LightGBM, then a stacking meta-learner over both.
7. **Eval + report** (`src/lidr_ml/eval/`) — metrics (accuracy, log loss, hit rate by year), equity curve assuming "go long when prediction = 1", HTML report written to `reports/<config>-<timestamp>/`. The report shows a **Strategy vs Buy & Hold** comparison table (CAGR, Sharpe, max drawdown, final equity for both legs) and a **per-year performance** table (strategy vs buy-hold return + excess) so regime-dependence is visible. A **Summary** section at the top translates the config into English and states the out-of-sample span (derived from the predictions, not the raw data range). Both the top classification metrics and the per-year table show **base_logloss** (the no-skill floor = entropy of the base rate) beside log loss, and the comparison/per-year tables green/red-highlight whichever side wins. The equity curve is marked to market with **1-day-forward returns**, not the N-day classification target — see Key Decisions.

Everything is wired together by `src/lidr_ml/pipeline.py::run_pipeline(config_path)`.

## Folder map

```
configs/                  experiment configs (YAML, one per run)
  baseline.yaml           SPY, 2005–today, SMA crossover + logistic regression
  dev_synthetic.yaml      same as baseline but synthetic data — offline-safe smoke test
data/
  raw/                    cached OHLCV pulled from yfinance
  processed/              feature DataFrames per config (cache)
src/lidr_ml/
  __init__.py
  __main__.py             entry for `python -m lidr_ml`
  cli.py                  Typer CLI
  pipeline.py             end-to-end orchestrator
  data/
    loaders.py            yfinance + synthetic loader
  signals/
    base.py               Signal protocol
    registry.py           name → signal-callable lookup
    sma_crossover.py      ported from lidr's lib/signals/sma.ts
  models/
    base.py               Model protocol
    logistic.py           sklearn logistic regression wrapper
  backtest/
    engine.py             expanding-window walk-forward backtester
  eval/
    metrics.py            accuracy, log loss, hit rate, equity curve
    report.py             HTML report generator (base64-embedded chart)
reports/                  generated HTML reports (gitignored except .gitkeep)
artifacts/
  models/                 (planned) final trained model — NOT yet written; backtest models are throwaway
  predictions/            JSON predictions consumed by lidr
tests/
  test_no_lookahead.py    asserts every registered signal is lookahead-safe
  test_pipeline_smoke.py  runs dev_synthetic config end-to-end
```

## Key Decisions

Things future-Claude would benefit from knowing before touching related code.

- **One Python project, separate from lidr.** Tried as a folder inside lidr first (mentally); rejected because Python venvs + Node modules in one tree gets messy fast and the two pieces deploy on different cadences (Vercel vs. eventually Railway/Render). The deliberate seam is the JSON artifact.
- **Source of truth for signal logic stays in lidr until cutover.** Each Python signal in `src/lidr_ml/signals/` is a *port* of the corresponding TS signal in lidr's `lib/signals/`. We assert numerical parity in tests so the two implementations don't drift. When the calibrated model finally goes live in production, *then* we'll decide whether the TS versions get deleted, kept as a fallback, or only used for display while predictions come from the artifact.
- **Lookahead bias is the #1 thing to guard against.** Every signal goes through `tests/test_no_lookahead.py`, which compares "value at time t computed from full series" vs. "value at time t computed from series truncated at t" and asserts they match. Easy to forget; catastrophic if missed. Do not add a signal without adding it to this test's registry.
- **Backtests use expanding-window walk-forward, never random k-fold.** Time-series data leaks information across random splits. `sklearn.model_selection.TimeSeriesSplit` does the slicing.
- **Synthetic data fallback exists for a reason.** The dev_synthetic config lets the pipeline run with no network, which makes CI, tests, and Cowork-sandbox verification trivial. Keep it functional. Do not delete it.
- **Confidence values in lidr today are heuristics, not probabilities.** The whole point of this project is to fix that. Anything we ship back to lidr from here should be a calibrated probability (sklearn's `predict_proba`, or LightGBM's, calibrated via Platt scaling or isotonic regression if needed). Note: the baseline uses `class_weight=balanced`, which deliberately distorts output probabilities away from true frequencies (fine for the up/down decision, bad for calibration). When probabilities start going to the site, likely switch to `class_weight=None` plus an explicit calibration step.
- **The report judges everything against a benchmark, never in absolute terms.** Metric values are uninterpretable alone, so the report always shows the no-skill floor (`base_logloss` = entropy of the base rate) next to log loss, and the buy-and-hold leg next to the strategy. Tables green/red-highlight whichever side wins (for log loss, green = below the floor; for the strategy, green = beats buy-and-hold; max drawdown is stored negative, so higher = better). Add any new comparison columns to the *right* of the existing benchmark.
- **The backtest does NOT produce a deployable model.** Expanding-window walk-forward fits a fresh model per split, predicts that split's test slice, and discards it — the deliverable is the stitched out-of-sample prediction series, used only for evaluation. There is no single trained model and nothing is written to `artifacts/models/`. Creating the deployable model is a separate, not-yet-built step: fit once on all available data, then serialize. See Next Up.
- **Transaction costs are modeled even in the stub.** Default 5 bps per trade. A strategy that's only profitable with zero costs is not a strategy. Baked into the equity curve from day one.
- **The equity curve uses 1-day-forward returns, NOT the N-day classification target.** The model predicts an N-day-forward direction (the classification target), but the strategy is marked to market daily: position taken on day *t* earns the return from *t* to *t+1*. Compounding the N-day return on every daily row (the original bug) overlaps the same window ~N times and massively inflates the curve. Keep these two return series separate: `fwd_clean` (N-day) is the target only; `daily_fwd_return` drives the equity. Regression-tested in `tests/test_strategy_returns.py`.
- **No survivorship-bias-free data yet.** yfinance only has currently-listed tickers. For SPY/QQQ/sector ETFs this is fine. For individual-stock backtests we'd want CRSP — out of scope for the personal-use phase, but worth flagging if/when results on individual names start looking suspiciously good.

## Next Up

Priority order, reset on 2026-05-21 around a single near-term goal: **prove the model has an edge over buy-and-hold** before building any serving/integration plumbing. The SPY baseline currently shows no edge (see Active Task), so the path below is instrument → add features → improve the model. Everything past #6 is explicitly gated on an edge actually appearing.

1. **Add a lightweight cross-run results log.** Append one row per run (config name, skill score = 1 − log_loss/base_logloss, CAGR & Sharpe vs buy-and-hold, max drawdown) to a CSV, so "did this change help?" is answerable without opening each HTML. *Moved to #1 because it's the cheapest item on the list and it makes every experiment below measurable — do it before the experiments, not after.* Interim before MLflow (#10).
2. **Build the TS→Python signal parity-test harness.** A shared fixed price series, a one-off TS script in lidr that dumps each signal's expected outputs to a JSON fixture, and a Python test asserting the ported signal matches. Prerequisite to porting more signals safely — prove the round-trip on the already-ported `sma_crossover` first.
3. **Port the remaining five lidr signals to Python**: `rsi`, `macd`, `bollinger`, `breakout`, `volume`. Each registered, added to `tests/test_no_lookahead.py`, and parity-tested via the harness from #2. *Biggest single lever on edge — takes the model from one feature to six. Re-run the logistic baseline on all six as a cheap checkpoint: if six features still can't beat the benchmark, the bottleneck is the model, not the features.*
4. **Add LightGBM as a second base learner.** Drop-in: implement `models/lightgbm.py` against the same `Model` protocol, add a config that uses it. *A nonlinear learner over six signals is the most likely place an edge first appears.*
5. **Introduce stacking.** Once two base learners exist, add a `StackedModel` whose `fit` trains the base learners via out-of-fold predictions and then trains a meta-learner (logistic regression) on top.
6. **Add regime features.** VIX level, yield-curve slope, 60-day realized vol. Fed into the meta-learner so it can lean on different base models in different environments.

— Edge gate: items below stay parked until something above beats buy-and-hold. —

7. **Add a final-model fit + serialize step.** The backtest only evaluates (throwaway per-split models); nothing fits a model on all data or writes `artifacts/models/`. Before serving live predictions, fit one model on all available history, serialize it, and write the prediction artifact. Trigger: once a model beats buy-and-hold. Revisit `class_weight`/calibration here (see Key Decisions).
8. **Define and write the artifact JSON schema.** Decide what fields lidr needs (predicted class? probability? per-signal contribution?). Document and version it (currently `schema_version: 1`, with `metrics.classification/strategy/benchmark`).
9. **Wire lidr's `/api/signals/[ticker]` to read the artifact.** That's the bridge moment. Coordinate with the lidr CLAUDE.md.
10. **Wire up MLflow for experiment tracking.** Replace the timestamped-folder report + CSV log with proper logged runs and a comparison UI. *Deferred: the CSV results log (#1) covers the "did this help?" need until the run count is large enough to justify the heavier tooling.*

## Active Task

_Nothing currently in-flight._

Iteration-readiness pass complete (2026-05-20 → 2026-05-21, see Recent Changes): equity-curve bug fixed; the report now answers "does the model beat buy-and-hold?" via a config Summary, Strategy-vs-Buy&Hold and per-year tables, base_logloss no-skill floors, and green/red highlighting; regression tests in place. The SPY baseline was run and reviewed end to end — the single-signal logistic model **underperforms buy-and-hold on every metric** (CAGR ~8.0% vs ~14.5%, Sharpe 0.67 vs 0.89, ~equal drawdown) and its log loss sits at the no-skill floor: no edge, exactly as expected for a one-feature baseline. That establishes the bar every real model must beat. Next (per the 2026-05-21 roadmap reprioritization): stand up the cross-run results log (Next Up #1), then the TS→Python parity-test harness (#2), then port the five remaining signals starting with RSI (#3).

<!-- Update this section when work is in progress. Replace with `_Nothing currently in-flight._`
     when paused. Keep it short: what's being built, where it was left off, mid-flight decisions. -->

## Recent Changes

### 2026-05-21 — Roadmap reprioritization around proving the edge

Reset the Next Up order to serve one near-term goal Boon confirmed: prove the model beats buy-and-hold before building any serving/integration plumbing. No code changed — this is a planning pass. Three substa