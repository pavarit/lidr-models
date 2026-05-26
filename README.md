# lidr-ml

[![CI](https://github.com/pavarit/lidr-ml/actions/workflows/test.yml/badge.svg)](https://github.com/pavarit/lidr-ml/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm--Noncommercial%201.0.0-blue.svg)](LICENSE)

Python ML pipeline that turns the technical signals from [lidr](https://github.com/pavarit/lidr) into empirically calibrated BUY / HOLD / SELL recommendations via backtested ensemble models.

This is the data-science sibling of the lidr Next.js app. See `CLAUDE.md` for the full architecture, design decisions, and roadmap.

## Quick start

```bash
# One-time setup (creates editable install)
make install

# Offline smoke test — no internet required, uses synthetic data
make backtest CONFIG=configs/dev_synthetic.yaml

# Real backtest — pulls SPY history from yfinance back to 2005
make backtest CONFIG=configs/baseline.yaml
```

Each run drops a self-contained HTML report into `reports/<config-name>-<timestamp>/report.html`. Open it in a browser.

After `make install`, you can also invoke the CLI directly as `python -m lidr_ml backtest <config>` or via the installed console script `lidr-ml backtest <config>`.

## Architecture

One pipeline, top-to-bottom. Everything is invoked from [`src/lidr_ml/pipeline.py::run_pipeline`](src/lidr_ml/pipeline.py):

```
                  configs/<name>.yaml
                          │
                          ▼
  ┌─────────────── pipeline.py::run_pipeline ───────────────┐
  │                                                          │
  │   data/loaders.py  →  signals/*.py  →  target builder    │
  │   (yfinance cached    (SMA crossover    (N-day forward    │
  │    or synthetic)       today; RSI/MACD   return > thr,    │
  │                        coming)           binary)          │
  │                                  │                        │
  │                                  ▼                        │
  │                       backtest/engine.py                  │
  │                       (expanding-window walk-forward;     │
  │                        fits a fresh model from models/*   │
  │                        per split — today: logistic reg.)  │
  │                                  │                        │
  │                                  ▼                        │
  │            eval/metrics.py + eval/report.py +             │
  │            eval/results_log.py                            │
  └──────────────────────────┬───────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   reports/<run>/    artifacts/predictions/   artifacts/
   report.html       <run>.json (opt-in)      results_log.csv
   (HTML + chart)    (JSON for lidr)          (one row appended)
```

**Stack**: Python 3.10+, pandas / numpy, scikit-learn (walk-forward CV + base learners), yfinance for data, PyYAML for configs, Typer for the CLI, Matplotlib for the embedded chart, pytest + ruff for tests + lint.

The HTML report contains: a config summary, top-line classification metrics with the no-skill floor (`base_logloss`) beside log loss, the Strategy-vs-Buy&Hold comparison table, the per-year classification breakdown, the per-year strategy-vs-benchmark returns with excess column, and the equity-curve chart. Wins are green/red-highlighted against the benchmark.

**Example report**: [view rendered](https://htmlpreview.github.io/?https://github.com/pavarit/lidr-ml/blob/main/docs/sample-report/report.html) (via `htmlpreview.github.io`) or [view raw HTML](docs/sample-report/report.html). Same baseline run cited in the SPY baseline status below (`baseline_v1-20260526-124439`).

## What's in the box right now

- Config-driven pipeline (YAML in, HTML report out)
- Expanding-window walk-forward backtest (no lookahead, regression-tested)
- One ported signal: SMA crossover (parity with lidr's `lib/signals/sma.ts`)
- One base model: logistic regression
- yfinance loader with a synthetic-data alternative (`source: synthetic` in any config) for offline development
- Transaction costs (5 bps default) baked into the equity curve
- Report benchmarks the strategy against buy-and-hold (CAGR, Sharpe, max drawdown, per-year excess) with base-rate floors on log loss
- Cross-run results log at `artifacts/results_log.csv` — one row appended per backtest
- Signal accuracy + no-lookahead test harness; CI runs `make test` + `make lint` on every push

**SPY baseline status** (run `20260526-124439`, OOS 2010-10-18 → 2026-04-23, 3,914 predictions across 16 walk-forward splits):

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

That's the bar every future model has to clear. Full row in [`artifacts/results_log.csv`](artifacts/results_log.csv); raw artifact at `artifacts/predictions/baseline_v1-20260526-124439.json`.

## CLI

Two commands, available as `python -m lidr_ml <command>` or (after `make install`) `lidr-ml <command>`:

| Command | Args | What it does |
| --- | --- | --- |
| `backtest` | `<config>` (path to a YAML) | Runs the full pipeline: load data → compute signals → train walk-forward → write HTML report + JSON artifact + CSV log row. |
| `list-signals` | — | Prints every signal name currently registered, one per line. Useful when authoring a new config. |

## Config schema

A config is YAML. All fields below are accepted by [`src/lidr_ml/pipeline.py::run_pipeline`](src/lidr_ml/pipeline.py); unrecognized top-level keys are ignored. Two reference examples live in [`configs/`](configs/).

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `name` | str | yes | — | Identifies the run; used in report dir name (`reports/<name>-<timestamp>/`) and the `config_name` column of `results_log.csv`. |
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
| `model.type` | str | yes | — | Only `logistic_regression` is currently supported. |
| `model.params` | dict | no | `{}` | Forwarded to the model class's `__init__`. For logistic, sklearn `LogisticRegression` kwargs (`C`, `class_weight`, …). |
| `backtest.cv` | str | yes | — | Only `expanding_window` is currently supported. |
| `backtest.initial_train_years` | int | yes | — | Size of the first training window. |
| `backtest.test_period_months` | int | yes | — | How far the window expands per step. |
| `backtest.transaction_cost_bps` | float | no | `5.0` | Charged on every position change in the equity curve. |
| `output.predictions_json` | bool | no | `false` | If true, writes the prediction artifact (see Outputs → JSON artifact). The HTML report is always written. |

## Outputs

Every run writes three things:

### HTML report — `reports/<config-name>-<timestamp>/report.html`

Self-contained (chart is base64-embedded; no internet needed to view). Contains a config summary, top-line classification metrics with the no-skill floor (`base_logloss`) beside log loss, Strategy-vs-Buy&Hold comparison table (CAGR, Sharpe, max drawdown, final equity), per-year classification + per-year strategy returns with excess-vs-buy-and-hold, and the equity curve.

### JSON artifact — `artifacts/predictions/<config-name>-<timestamp>.json`

Written only when `output.predictions_json: true`. This is what lidr's `/api/signals/[ticker]` will eventually consume. Current shape (`schema_version: 1`), abridged from the live SPY baseline (`artifacts/predictions/baseline_v1-20260526-124439.json`):

```json
{
  "schema_version": 1,
  "config_name": "baseline_v1",
  "ticker": "SPY",
  "generated_at": "20260526-124439",
  "metrics": {
    "classification": { "accuracy": 0.5565, "base_rate": 0.6134, "pred_rate": 0.7534, "log_loss": 0.6925, "base_logloss": 0.6672, "n_obs": 3914 },
    "strategy":       { "cagr": 0.0796, "sharpe": 0.6656, "max_drawdown": -0.3375, "final_equity": 3.2847 },
    "benchmark":      { "cagr": 0.1454, "sharpe": 0.8869, "max_drawdown": -0.3372, "final_equity": 8.2275 }
  },
  "predictions": [
    { "date": "2010-10-18", "y_true": 1, "y_pred": 0, "probability_up": 0.4999 },
    "..."
  ]
}
```

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

Single near-term goal: **prove the model has an edge over buy-and-hold** before building any serving/integration plumbing. See `CLAUDE.md` → Next Up for the full priority list. Short version: port the remaining five lidr signals (RSI, MACD, Bollinger, breakout, volume) → add LightGBM as a second base learner → stacking → regime features. Final-model fit/serialize, artifact schema, lidr wiring, and MLflow are all explicitly **gated** on something actually beating buy-and-hold first.

## Project layout

```
configs/                  experiment configs (YAML, one per run)
data/raw/                 cached OHLCV pulled from yfinance (regeneratable)
src/lidr_ml/              the package — pipeline, signals, models, backtest, eval, CLI
tests/                    pytest suite — lookahead-safety, signal accuracy,
                          strategy-return invariants, pipeline smoke test
.github/workflows/        CI (test + lint on every push)
reports/                  generated HTML reports (gitignored except .gitkeep)
artifacts/
  predictions/            JSON predictions consumed by lidr
  models/                 (planned) final trained model — not yet written
  results_log.csv         cross-run results log (one row per backtest, git-tracked)
```

The full per-module breakdown lives in [`CLAUDE.md`](CLAUDE.md) → Folder map.

## Requirements

- Python 3.10+ (CI tests on 3.11; Python 3.14 works for everything except parquet-based caches — we use pickle on purpose).
- Internet access for `configs/baseline.yaml` (yfinance). The `dev_synthetic` config runs offline.

## Contributing

PRs and issues welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the three-things-must-land rule for adding a signal, the pattern for adding a model, and the backtest invariants enforced by tests + CI.

## License

**[PolyForm Noncommercial 1.0.0](LICENSE)** — full text in the LICENSE file.

In plain English:

- ✅ Free to use, modify, and share for any **noncommercial** purpose — personal research, education, hobby projects, evaluation, work at a charitable / educational / public-research organization.
- ❌ **Commercial use requires a separate license** from the copyright holder. This includes running this code as a hosted service, embedding it in a product you sell, or otherwise using it as part of revenue-generating activity.

If you want to use lidr-ml commercially, open a GitHub issue or contact the author directly to discuss licensing.
