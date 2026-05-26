# lidr-ml

[![CI](https://github.com/pavarit/lidr-ml/actions/workflows/test.yml/badge.svg)](https://github.com/pavarit/lidr-ml/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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

**SPY baseline status:** the single-signal logistic model **does not** beat buy-and-hold (CAGR ~8.0% vs ~14.5%, log loss at the no-skill floor). That's the bar every future model must clear.

## What's next

Single near-term goal: **prove the model has an edge over buy-and-hold** before building any serving/integration plumbing. See `CLAUDE.md` → Next Up for the full priority list. Short version: port the remaining five lidr signals (RSI, MACD, Bollinger, breakout, volume) → add LightGBM as a second base learner → stacking → regime features. Final-model fit/serialize, artifact schema, lidr wiring, and MLflow are all explicitly **gated** on something actually beating buy-and-hold first.

## Project layout

See `CLAUDE.md` → Folder map.

## Requirements

- Python 3.10+ (CI tests on 3.11; Python 3.14 works for everything except parquet-based caches — we use pickle on purpose).
- Internet access for `configs/baseline.yaml` (yfinance). The `dev_synthetic` config runs offline.

## Contributing

PRs and issues welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the three-things-must-land rule for adding a signal, the pattern for adding a model, and the backtest invariants enforced by tests + CI.

## License

[MIT](LICENSE) — see the LICENSE file for full text.
