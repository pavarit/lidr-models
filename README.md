# lidr-ml

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

## What's in the box right now

- Config-driven pipeline (YAML in, HTML report out)
- Expanding-window walk-forward backtest (no lookahead)
- One ported signal: SMA crossover (parity with lidr's `lib/signals/sma.ts`)
- One base model: logistic regression
- yfinance loader with a synthetic-data fallback for offline development
- Transaction costs (5 bps default) baked into the equity curve

## What's next

See `CLAUDE.md` → Next Up. Short version: port the other five lidr signals, add LightGBM, add MLflow, add stacking + regime features, write the artifact JSON that lidr will read.

## Project layout

See `CLAUDE.md` → Folder map.

## Requirements

- Python 3.10+
- Internet access for `configs/baseline.yaml` (yfinance). The `dev_synthetic` config runs offline.
