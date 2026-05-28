# lidr-core

The shared harness every model in this monorepo builds against. Lives at
`packages/lidr_core/` and exposes:

- `lidr_core.backtest` — expanding-window walk-forward engine + strategy returns
- `lidr_core.eval` — classification + strategy metrics, results_log, leaderboard
- `lidr_core.contract` — the JSON Schema for the prediction artifact + writer/loader
- `lidr_core.protocols` — `SignalFn`, `Model`, `Feature`, `DataSource` interfaces
- `lidr_core.data.loaders` — yfinance / synthetic OHLCV loader
- `lidr_core.models` — generic learners (`logistic_regression`, `lightgbm`)

Models (`ta_ensemble`, `news_sentiment`) depend on this package; do not flip
the dependency direction. See [ADR 0001](../../docs/adr/0001-multi-model-repo-architecture.md)
for the rationale.
