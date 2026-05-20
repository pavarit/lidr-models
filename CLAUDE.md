# CLAUDE.md

This file orients any AI assistant (Claude Code, Claude Cowork, etc.) joining this project. Read it before doing anything else. Then keep it current — see "Maintenance Instructions" at the bottom.

## Project Goal

`lidr-ml` is the Python ML/backtesting pipeline that turns the technical signals from the sibling project [`lidr`](https://github.com/pavarit/lidr) into **empirically calibrated BUY / HOLD / SELL recommendations**. lidr's confidence values today are heuristics (normalized strength scores). This project's job is to replace them with probabilities learned from historical data — by training an ensemble model over the signals, backtested with walk-forward validation from pre-2008 to today.

The pipeline emits a versioned JSON artifact that lidr's `/api/signals/[ticker]` route can read directly. No running Python service is needed until the architecture decision in lidr's Next Up #6 is triggered.

## Architecture

Two-project setup, deliberately kept separate:

- **`lidr`** (Next.js, `C:\Users\smnk1\Claude\Projects\lidr`) — the website. Computes signals live in TS for display, will eventually overlay calibrated probabilities from the artifact this project produces.
- **`lidr-ml`** (this project, Python) — produces the model and the artifact. Iterates on its own cadence with its own tooling.

Integration is via a JSON file (`artifacts/predictions/<config>-<timestamp>.json`), not an HTTP service. Cheap, debuggable, version-controllable. A FastAPI service is on the lidr roadmap (Next Up #6) and will be added here when the lidr side is ready to consume live predictions.

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
4. **Target** (`src/lidr_ml/pipeline.py::make_target`) — for now, binary: was the *N*-day forward return positive? Will become 3-class (BUY/HOLD/SELL) once we have a useful model.
5. **Backtest engine** (`src/lidr_ml/backtest/engine.py`) — expanding-window walk-forward. For each split: fit model on train slice, predict on test slice, store predictions. Never trains on data after the test period.
6. **Model** (`src/lidr_ml/models/`) — pluggable. Today: logistic regression. Next: LightGBM, then a stacking meta-learner over both.
7. **Eval + report** (`src/lidr_ml/eval/`) — metrics (accuracy, log loss, hit rate by year), equity curve assuming "go long when prediction = 1", HTML report written to `reports/<config>-<timestamp>/`.

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
  models/                 pickled trained models per run
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
- **Confidence values in lidr today are heuristics, not probabilities.** The whole point of this project is to fix that. Anything we ship back to lidr from here should be a calibrated probability (sklearn's `predict_proba`, or LightGBM's, calibrated via Platt scaling or isotonic regression if needed).
- **Transaction costs are modeled even in the stub.** Default 5 bps per trade. A strategy that's only profitable with zero costs is not a strategy. Baked into the equity curve from day one.
- **No survivorship-bias-free data yet.** yfinance only has currently-listed tickers. For SPY/QQQ/sector ETFs this is fine. For individual-stock backtests we'd want CRSP — out of scope for the personal-use phase, but worth flagging if/when results on individual names start looking suspiciously good.

## Next Up

In rough priority order:

1. **Port the remaining five lidr signals to Python**: `rsi`, `macd`, `bollinger`, `breakout`, `volume`. For each, write the Python version, register it, add a parity test against the TS implementation (run the TS version once over a fixed series, dump expected outputs to a JSON fixture, then assert the Python version matches).
2. **Add LightGBM as a second base learner.** Drop-in: implement `models/lightgbm.py` against the same `Model` protocol, add a config that uses it.
3. **Wire up MLflow for experiment tracking.** Replace the timestamped-folder report with proper logged runs. Web UI for comparing across configs.
4. **Introduce stacking.** Once two base learners exist, add a `StackedModel` whose `fit` trains the base learners via out-of-fold predictions and then trains a meta-learner (logistic regression) on top.
5. **Add regime features.** VIX level, yield-curve slope, 60-day realized vol. Fed into the meta-learner so it can lean on different base models in different environments.
6. **Define and write the artifact JSON schema.** Decide what fields lidr needs (predicted class? probability? per-signal contribution?). Document the schema, version it (`schema_version: 1`), write to `artifacts/predictions/`.
7. **Wire lidr's `/api/signals/[ticker]` to read the artifact.** That's the bridge moment. Coordinate with the lidr CLAUDE.md.

## Active Task

**Scaffolded by Cowork on 2026-05-19, verified on Boon's WSL machine same day.** End-to-end run works on both the synthetic config (offline) and the baseline config (SPY 2005–today via yfinance). First substantive iteration in Claude Code: tackle Next Up #1 — port the remaining signals starting with RSI (simplest after SMA).

<!-- Update this section when work is in progress. Replace with `_Nothing currently in-flight._`
     when paused. Keep it short: what's being built, where it was left off, mid-flight decisions. -->

## Recent Changes

### 2026-05-19 — Initial scaffold + first end-to-end run

Stood up the project structure described in this file: src-layout package, config-driven pipeline, expanding-window walk-forward backtest, HTML report with embedded equity curve, yfinance loader with synthetic-data fallback, one ported signal (SMA crossover), one base model (logistic regression), pytest smoke test that runs the pipeline end-to-end on the synthetic config. No real model performance to speak of yet — the point of this commit is working plumbing, not a working strategy. Verified the synthetic config runs cleanly inside the Cowork sandbox, then Boon ran both `dev_synthetic` and `baseline` configs on his WSL machine (Python 3.14 venv) — both produced HTML reports and JSON prediction artifacts as expected.

Two small fixes shook out during verification that are worth knowing about: (1) Typer collapses single-command apps into a flat CLI, which broke `python -m lidr_ml backtest ...`; added a second command (`list-signals`) so Typer keeps `backtest` as a real subcommand — do not remove `list-signals` until a third real command exists. (2) The data cache originally used parquet, which requires `pyarrow` or `fastparquet`; on Python 3.14 neither has wheels yet. Switched to pickle — zero extra deps, fine for a local-only regeneratable cache. Inline comment in `loaders.py` explains why; don't switch it back without a real reason.

## Maintenance Instructions

If you (a future AI assistant joining this project) make meaningful changes, also update this file in the same session.

- **Keep evergreen sections current.** Project Goal, Architecture, Stack, How the pipeline works, Folder map, Key Decisions, Next Up should reflect reality. If your work invalidates a fact in any of these sections, update it before ending the session.
- **Append a dated entry to Recent Changes for each session that produces real changes.** Use a `### YYYY-MM-DD — short title` header followed by a paragraph describing what was done and why. Include decisions and rationale future-Claude would benefit from knowing.
- **Cross-link to lidr.** If a change here affects the integration with lidr (artifact format, signal parity, what the website should consume), update lidr's CLAUDE.md too in the same session.
- **Archive when Recent Changes exceeds 10 entries.** Fold the oldest 5 into a `## Archived Summary` section at the bottom. Preserve decisions and rationale; compress narratives, not insights.
