# lidr-models

[![CI](https://github.com/pavarit/lidr-models/actions/workflows/test.yml/badge.svg)](https://github.com/pavarit/lidr-models/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm--Noncommercial%201.0.0-blue.svg)](LICENSE)

A Python research monorepo that produces empirically calibrated BUY / HOLD / SELL recommendations for the [lidr](https://github.com/pavarit/lidr) Next.js front-end. Models compete against each other on a shared backtest harness and a shared JSON artifact contract.

Three packages under `packages/`:

- **`lidr_core`** — the shared harness (backtest engine, eval/metrics, results_log + leaderboard, artifact JSON Schema + writer/loader, protocols, base data loaders, generic learners).
- **`ta_ensemble`** — the six TA signals + pipeline. Today's only complete model.
- **`news_sentiment`** — in development (Task 2 PR-A scaffolding: free data adapters + collector + lexicon scorer + three features + offline dev pipeline). Real backtest and news-vs-TA comparison land in PR-B/C.

**New here?** Start with this README (run instructions, the [pipeline walkthrough](#how-the-pipeline-works), and [current status](#current-status-at-a-glance)), then the [`docs/` index](docs/README.md) for the ADR, the signals explainer, and research notes. [`CLAUDE.md`](CLAUDE.md) is the deep, AI-facing playbook (full history, conventions, gotchas, roadmap) — the most complete doc, and the longest.

## Current status at a glance

_Updated 2026-05-29. This is the canonical "where is the project right now" summary; [`CLAUDE.md`](CLAUDE.md) carries the full experiment history behind it._

- **No model beats buy-and-hold yet — and closing that gap is the whole near-term focus.** The six TA signals fed to logistic regression *and* LightGBM all land at or below the no-skill floor (best `skill_score` ≈ −0.005, the unweighted six-signal logistic). Three well-behaved configs cluster at the floor, and a target-horizon sweep (5 → 60 day) only made it worse — so the bottleneck is the **features/target, not the model class**.
- **Next lever:** reformulate the target/features — return-magnitude regression instead of binary sign, and regime features (VIX, yield-curve slope, realized vol). Stacking, final-model serving, probability calibration, 3-class output, and lidr wiring are all gated behind a model first clearing buy-and-hold.
- **Second model — `news_sentiment` (Task 2):** PR-A scaffolding merged (free adapters + collector + lexicon scorer + features + offline pipeline). Real backtest + news-vs-TA comparison land in PR-B/PR-C.

## Quick start

```bash
# One-time setup (installs all three packages editable)
make install

# Offline smoke test — no internet required, uses synthetic data
make backtest CONFIG=packages/ta_ensemble/configs/dev_synthetic.yaml

# Real backtest — pulls SPY history from yfinance back to 2005
make backtest CONFIG=packages/ta_ensemble/configs/baseline.yaml
```

Each run drops a self-contained HTML report into `reports/<config-name>-<timestamp>/report.html`. Open it in a browser.

After `make install`, you can also invoke the CLI directly as `python -m ta_ensemble backtest <config>` or via the installed console script `ta-ensemble backtest <config>`.

## Architecture

One pipeline per model, top-to-bottom — orchestrated from each model's `pipeline.py` but using shared harness code in `lidr_core`. For `ta_ensemble`, the entry point is [`packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline`](packages/ta_ensemble/src/ta_ensemble/pipeline.py):

```
                  packages/<model>/configs/<name>.yaml
                                  │
                                  ▼
  ┌──────────── <model>/pipeline.py::run_pipeline ──────────────┐
  │                                                              │
  │  lidr_core.data.loaders  →  <model>/signals/*  →  target     │
  │  (yfinance cached            (TA signals for                  │
  │   or synthetic)               ta_ensemble; news/             │
  │                               sentiment features                │
  │                               coming in news_sentiment)         │
  │                                          │                    │
  │                                          ▼                    │
  │                          lidr_core.backtest.engine            │
  │                          (expanding-window walk-forward;      │
  │                           fits a fresh model from              │
  │                           lidr_core.models per split)         │
  │                                          │                    │
  │                                          ▼                    │
  │            lidr_core.eval.{metrics, report, results_log}      │
  │            lidr_core.contract.writer (validates artifact)     │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────────┐
        ▼                        ▼                             ▼
   reports/<run>/    artifacts/predictions/<model_id>/   artifacts/
   report.html       <run>.json (v2 schema, opt-in)      results_log.csv
   (HTML + chart)    + artifacts/manifest.json           (one row appended)
                     (leaderboard)
```

**Stack**: Python 3.10+, pandas / numpy, scikit-learn (walk-forward CV + base learners), LightGBM, yfinance for data, PyYAML for configs, Typer for the CLI, jsonschema for contract validation, Matplotlib for the embedded chart, pytest + ruff for tests + lint.

The HTML report contains: a config summary, top-line classification metrics with the no-skill floor (`base_logloss`) beside log loss, the Strategy-vs-Buy&Hold comparison table, the per-year classification breakdown, the per-year strategy-vs-benchmark returns with excess column, and the equity-curve chart. Wins are green/red-highlighted against the benchmark.

**Example report**: [view rendered](https://htmlpreview.github.io/?https://github.com/pavarit/lidr-models/blob/main/docs/sample-report/report.html) (via `htmlpreview.github.io`) or [view raw HTML](docs/sample-report/report.html). Same baseline run cited in the SPY baseline status below (`baseline_v1-20260526-124439`).

## How the pipeline works

The [Architecture](#architecture) diagram above shows the data flow; this is the per-step detail, reading top-down. Everything for the TA model is wired together by [`packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline`](packages/ta_ensemble/src/ta_ensemble/pipeline.py).

1. **Config** (`packages/ta_ensemble/configs/*.yaml`) — declares what to run: tickers, date range, data source, which signals, which model, backtest method, transaction costs. Also carries `model_id` + `model_version` so produced artifacts identify their model family. See [Config schema](#config-schema) and [`packages/ta_ensemble/configs/`](packages/ta_ensemble/configs/).
2. **Data loader** (`packages/lidr_core/src/lidr_core/data/loaders.py`) — pulls OHLCV from yfinance (or generates synthetic series for offline dev). yfinance is called with `auto_adjust=True`, so close prices are total-return-adjusted (dividends + splits folded in); a "200-day high" is a 200-day high of the adjusted series, not the raw close. Cached by `(ticker, start, end)` in `data/raw/<ticker>_<start>_<end>.pkl`; the cache never expires, so `rm data/raw/*.pkl` to force a refresh.
3. **Signals** (`packages/ta_ensemble/src/ta_ensemble/signals/`) — each signal is a pure function that takes a DataFrame of prices and returns a Series of feature values aligned to the price index. Every signal must be **lookahead-safe** (only uses data up to time *t* to compute the value at time *t*), enforced by `packages/ta_ensemble/tests/test_no_lookahead.py`.
4. **Target** (computed inline in `run_pipeline`) — for now, binary: was the *N*-day forward return positive? Will become 3-class (BUY/HOLD/SELL) once a model has edge.
5. **Backtest engine** (`packages/lidr_core/src/lidr_core/backtest/engine.py`) — expanding-window walk-forward. For each split: fit the model on the train slice, predict on the test slice, store predictions. Never trains on data after the test period.
6. **Model** (`packages/lidr_core/src/lidr_core/models/`) — pluggable; today logistic regression and LightGBM (both generic, hence in `lidr_core`). Neither beats the no-skill baseline on the six-signal feature set yet (see [Current status](#current-status-at-a-glance)).
7. **Eval + report** (`packages/lidr_core/src/lidr_core/eval/`) — classification metrics (`accuracy`, `base_rate`, `pred_rate`, `log_loss`, `base_logloss`, `n_obs`), strategy metrics (`cagr`, `sharpe`, `max_drawdown`, `final_equity`), and per-year breakdowns. `base_rate` / `pred_rate` sit beside `accuracy` so a 0.55 accuracy can't hide a 0.61 base rate. The equity curve is marked to market with **1-day-forward returns**, not the N-day classification target (compounding the N-day return daily would inflate equity ~N×). HTML report written to `reports/<config>-<timestamp>/`.
8. **Cross-run results log** (`packages/lidr_core/src/lidr_core/eval/results_log.py`) — `append_run()` adds one row per backtest to `artifacts/results_log.csv` (skill score, full-OOS + per-period log loss with base-rate floors, strategy vs benchmark CAGR/Sharpe/max-drawdown, excess) so "did this change help?" is answerable without opening every report. Opt out per-config with `output.results_log: false` (default true).
9. **Artifact contract + leaderboard** (`packages/lidr_core/src/lidr_core/contract/`, `.../eval/leaderboard.py`) — `build_artifact` assembles the `schema_version: 2` payload and `write_artifact` validates it against `artifact.schema.json` before writing under `artifacts/predictions/<model_id>/`; `write_manifest` emits `artifacts/manifest.json` so lidr can discover every model + its headline OOS skill. See [Outputs](#outputs).

## What's in the box right now

- Config-driven pipeline (YAML in, HTML report out)
- Expanding-window walk-forward backtest (no lookahead, regression-tested)
- Six signals: SMA crossover, RSI, MACD, Bollinger Bands, breakout, volume (the latter five ported from lidr's TS with numerical parity) — see [docs/signals.md](docs/signals.md) for what each one measures, with charts on real SPY data
- Two base learners: logistic regression and LightGBM (both generic, in `lidr_core`)
- yfinance loader with a synthetic-data alternative (`source: synthetic` in any config) for offline development
- Transaction costs (5 bps default) baked into the equity curve
- Report benchmarks the strategy against buy-and-hold (CAGR, Sharpe, max drawdown, per-year excess) with base-rate floors on log loss
- Cross-run results log at `artifacts/results_log.csv` — one row appended per backtest
- Signal accuracy + no-lookahead test harness; CI runs `make test` + `make lint` on every push

**SPY single-signal baseline** — the simplest config (SMA crossover → logistic) and the first real bar (run `20260526-124439`, OOS 2010-10-18 → 2026-04-23, 3,914 predictions across 16 walk-forward splits):

| | Strategy | Buy & Hold |
| --- | --- | --- |
| CAGR | **7.96 %** | **14.54 %** |
| Sharpe | 0.67 | 0.89 |
| Max drawdown | −33.75 % | −33.72 % |
| Final equity | 3.28× | 8.23× |

The single-signal logistic model **underperforms buy-and-hold on every dimension**, and the classifier itself is worse than no-skill:

- **Skill score −0.038** — log loss `0.6925` sits *above* the no-skill floor `0.6672` (entropy of the 61.3 % base rate).
- **Accuracy `0.556` < base rate `0.613`** — predicting "up" every day beats this model on raw correctness.
- **Pred-rate `0.753` vs base-rate `0.613`** — model predicts "up" 75 % of the time despite reality being 61 %. Strongly biased even with `class_weight="balanced"`.

That's the floor every model has to clear — and nothing has yet: adding all six signals and swapping in LightGBM reached the same verdict (see [Current status](#current-status-at-a-glance)), which is why the roadmap now targets the features/target rather than the model class. The full row for this run (`run_id` `20260526-124439`) is in [`artifacts/results_log.csv`](artifacts/results_log.csv) — the git-tracked record of every backtest. (Per-run prediction JSONs under `artifacts/predictions/` are gitignored build outputs and aren't present on a fresh clone — see [Outputs](#outputs).)

## CLI

Two commands, available as `python -m ta_ensemble <command>` or (after `make install`) `ta-ensemble <command>`:

| Command | Args | What it does |
| --- | --- | --- |
| `backtest` | `<config>` (path to a YAML) | Runs the full pipeline: load data → compute signals → train walk-forward → write HTML report + JSON artifact + CSV log row. |
| `list-signals` | — | Prints every signal name currently registered, one per line. Useful when authoring a new config. |

## Config schema

A config is YAML. All fields below are accepted by [`packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline`](packages/ta_ensemble/src/ta_ensemble/pipeline.py); unrecognized top-level keys are ignored. Reference examples live in [`packages/ta_ensemble/configs/`](packages/ta_ensemble/configs/).

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `name` | str | yes | — | Identifies the run; used in report dir name (`reports/<name>-<timestamp>/`) and the `config_name` column of `results_log.csv`. |
| `model_id` | str | no | `ta_ensemble` | Stable identifier for the producing model family; written into the v2 artifact and used by the leaderboard. |
| `model_version` | str | no | `0.0.0` | Version of this model's code+config combination; written into the v2 artifact. |
| `description` | str | no | — | Free text; rendered into the report's Summary section. |
| `data.source` | str | yes | — | `yfinance` or `synthetic`. `yfinance` requires internet; `synthetic` generates a deterministic geometric-Brownian-motion price series. |
| `data.tickers` | list[str] | yes | — | Must be a list, but pipeline currently rejects length > 1. Multi-ticker is gated on adding cross-sectional features. |
| `data.start_date` | str (ISO) | yes | — | e.g. `2005-01-01`. |
| `data.end_date` | str (ISO) | yes | — | e.g. `2026-05-01`. |
| `data.synthetic.drift_annual` | float | no | `0.07` | Used only when `data.source: synthetic`. |
| `data.synthetic.vol_annual` | float | no | `0.16` | Used only when `data.source: synthetic`. |
| `data.synthetic.seed` | int | no | `42` | Used only when `data.source: synthetic`. |
| `signals` | list[dict] | yes | — | Each entry: `{name, params}`. `name` must match a registered signal — see `list-signals`. |
| `target.type` | str | yes | — | Only `forward_return_binary` is currently supported. |
| `target.horizon_days` | int | yes | — | N for "was the N-day forward return > `threshold`?" |
| `target.threshold` | float | no | `0.0` | Return > threshold → class 1, else class 0. |
| `model.type` | str | yes | — | `logistic_regression` or `lightgbm` (registered in `lidr_core.models`). |
| `model.params` | dict | no | `{}` | Forwarded to the model class's `__init__`. For logistic, sklearn `LogisticRegression` kwargs (`C`, `class_weight`, …). |
| `backtest.cv` | str | yes | — | Only `expanding_window` is currently supported. |
| `backtest.initial_train_years` | int | yes | — | Size of the first training window. |
| `backtest.test_period_months` | int | yes | — | How far the window expands per step. |
| `backtest.transaction_cost_bps` | float | no | `5.0` | Charged on every position change in the equity curve. |
| `output.predictions_json` | bool | no | `false` | If true, writes the prediction artifact (see Outputs → JSON artifact). The HTML report is always written. |
| `output.results_log` | bool | no | `true` | If false, the run is **not** appended to `artifacts/results_log.csv`. Set to false in `dev_synthetic.yaml` so smoke-test / CI runs don't pollute the tracked log with synthetic rows. |

## Outputs

Every run writes three things. **Only `results_log.csv` is tracked in git** — the per-run reports, prediction JSONs, and `manifest.json` are all gitignored build outputs, so a fresh clone has none of them and they don't need cleaning up to commit. Locally they accumulate (45+ loose JSONs is normal); clear them with the `make clean-*` targets.

| Output | Path | Tracked in git? | Notes |
| --- | --- | --- | --- |
| HTML report | `reports/<run>/report.html` | No (gitignored) | One dir per run. Clear with `make clean-reports`. |
| Prediction JSON | `artifacts/predictions/<model_id>/<run>.json` | No (gitignored) | Written only when `output.predictions_json: true`. Older runs predating the monorepo restructure sit loose under `artifacts/predictions/`. Clear with `make clean-predictions`. |
| Leaderboard | `artifacts/manifest.json` | No (gitignored) | Regenerated each run from whatever artifacts are on disk. Each model is headlined by its **most recent real run** (synthetic/`dev_synthetic` smoke runs are excluded; a model with only smoke runs is omitted entirely). "Most recent" is the artifact's embedded `generated_at`, not file mtime. Note this is *latest*, not *best* — after an experiment sweep the headline is whatever config ran last, which may not be the model's best-scoring config. |
| Results log | `artifacts/results_log.csv` | **Yes** | The durable cross-run record; one row per backtest, accumulates across machines. This is the authoritative "what has been run." |

The three per-run outputs:

### HTML report — `reports/<config-name>-<timestamp>/report.html`

Self-contained (chart is base64-embedded; no internet needed to view). Contains a config summary, top-line classification metrics with the no-skill floor (`base_logloss`) beside log loss, Strategy-vs-Buy&Hold comparison table (CAGR, Sharpe, max drawdown, final equity), per-year classification + per-year strategy returns with excess-vs-buy-and-hold, and the equity curve.

### JSON artifact — `artifacts/predictions/<model_id>/<config-name>-<timestamp>.json`

Written only when `output.predictions_json: true`. Built and validated by [`lidr_core.contract.writer`](packages/lidr_core/src/lidr_core/contract/writer.py) against [`artifact.schema.json`](packages/lidr_core/src/lidr_core/contract/schema/artifact.schema.json). This is what lidr's `/api/signals/[ticker]` will eventually consume. Current shape (`schema_version: 2`):

```json
{
  "schema_version": 2,
  "model_id": "ta_ensemble",
  "model_version": "0.2.0",
  "config_name": "baseline_six_signals_unweighted",
  "ticker": "SPY",
  "generated_at": "20260527-190244",
  "metrics": {
    "classification": { "accuracy": 0.6100, "base_rate": 0.6111, "log_loss": 0.6717, "base_logloss": 0.6683, "n_obs": 3851 },
    "strategy":       { "cagr": 0.1425, "sharpe": 0.9036, "max_drawdown": -0.2671, "final_equity": 7.6497 },
    "benchmark":      { "cagr": 0.1404, "sharpe": 0.8537, "max_drawdown": -0.3372, "final_equity": 7.4436 }
  },
  "predictions": [
    { "date": "2010-12-30", "recommendation": "HOLD", "probability_up": 0.51, "y_pred": 1, "y_true": 1 },
    "..."
  ]
}
```

A top-level `artifacts/manifest.json` (built by `lidr_core.eval.leaderboard.write_manifest`) lists every model_id and points to its latest *real* artifact (smoke runs excluded), so lidr can discover what's available. See [Outputs](#outputs) for the selection rule.

Bump `schema_version` whenever a field is renamed, removed, or its type changes. Additive changes (new optional fields) don't require a bump.

### Cross-run results log — `artifacts/results_log.csv`

One row appended per backtest. Git-tracked so experiment history accumulates across machines. Columns (in order):

| Column | Type | Meaning |
| --- | --- | --- |
| `run_id` | str | Timestamp `YYYYMMDD-HHMMSS` — matches the `reports/` subdirectory. |
| `config_name` | str | From `name` in the YAML. |
| `ticker` | str | Single ticker (see config schema note). |
| `oos_start`, `oos_end` | date | First and last date in the out-of-sample prediction series. |
| `n_oos` | int | Number of OOS predictions (rows in the stitched walk-forward output). |
| `skill_score` | float | `1 − log_loss / base_logloss`. Positive → beats the no-skill floor. |
| `base_logloss`, `log_loss`, `accuracy` | float | Full-OOS classification metrics. |
| `base_logloss_2025`, `log_loss_2025` | float | Same, sliced to calendar year 2025. Empty when the year is missing or degenerate. |
| `base_logloss_2026q1`, `log_loss_2026q1` | float | Same, sliced to Q1 2026. |
| `strategy_cagr`, `strategy_sharpe`, `strategy_max_dd`, `strategy_final_equity` | float | Strategy (the model's equity curve, net of transaction costs). |
| `bench_cagr`, `bench_sharpe`, `bench_max_dd`, `bench_final_equity` | float | Buy-and-hold benchmark on the same OOS span. |
| `excess_cagr`, `excess_sharpe` | float | Strategy minus benchmark. The "did this help?" columns. |
| `report_path` | str | Relative path to the HTML report from the project root. |

To remove a bad row (e.g. a buggy run), edit the CSV directly — git diff will show what changed.

## What's next

Single near-term goal: **prove a model has an edge over buy-and-hold** before building any serving/integration plumbing. The six TA signals, LightGBM, and a target-horizon sweep are all done — none cleared the no-skill floor, and the diagnosis is that the **features/target** are the bottleneck, not the model class. So the live directions are reformulating the target (return-magnitude regression instead of binary sign) and adding regime features (VIX, yield-curve slope, realized vol); stacking is parked until a base learner shows non-zero skill. Final-model fit/serialize, probability calibration, 3-class output, lidr wiring, and MLflow are all explicitly **gated** on something actually beating buy-and-hold first. See [`CLAUDE.md`](CLAUDE.md) → Next Up for the full priority list and [Current status](#current-status-at-a-glance) for where things stand.

## Project layout

```
packages/
  lidr_core/        shared harness — backtest engine, eval, contract, protocols, base learners
  ta_ensemble/      the six TA signals + pipeline + configs (today's only complete model)
  news_sentiment/   in development — Task 2 PR-A scaffolding (adapters, collector, scorer, features)
data/raw/           cached OHLCV pulled from yfinance (regeneratable)
docs/               ADR, research, sample report, signal explainer
.github/workflows/  CI (test + lint on every push)
reports/            generated HTML reports (gitignored except .gitkeep)
artifacts/
  predictions/<model_id>/  v2 JSON artifacts (one subdir per model)
  manifest.json            leaderboard — every model + its latest artifact + OOS skill
  results_log.csv          cross-run results log (one row per backtest, git-tracked)
```

The full per-module breakdown lives in [`CLAUDE.md`](CLAUDE.md) → Folder map.

## Requirements

- Python 3.10+ (CI tests on 3.11; Python 3.14 works for everything except parquet-based caches — we use pickle on purpose).
- Internet access for `packages/ta_ensemble/configs/baseline.yaml` (yfinance). The `dev_synthetic` config runs offline.

## Contributing

PRs and issues welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the three-things-must-land rule for adding a signal, the pattern for adding a model, and the backtest invariants enforced by tests + CI.

## License

**[PolyForm Noncommercial 1.0.0](LICENSE)** — full text in the LICENSE file.

In plain English:

- ✅ Free to use, modify, and share for any **noncommercial** purpose — personal research, education, hobby projects, evaluation, work at a charitable / educational / public-research organization.
- ❌ **Commercial use requires a separate license** from the copyright holder. This includes running this code as a hosted service, embedding it in a product you sell, or otherwise using it as part of revenue-generating activity.

If you want to use lidr-models commercially, open a GitHub issue or contact the author directly to discuss licensing.
