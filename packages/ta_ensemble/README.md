# ta_ensemble

The six-signal technical-analysis ensemble — historically the only model in
this repo, now one of (potentially) several. Computes the `sma_crossover`,
`rsi`, `macd`, `bollinger`, `breakout`, and `volume` features from OHLCV
prices and feeds them to a logistic-regression or LightGBM classifier
trained walk-forward against the 5d-forward-return-sign target.

- **Configs:** `packages/ta_ensemble/configs/*.yaml`
- **Pipeline entry point:** `python -m ta_ensemble backtest <config>`
- **CLI:** `ta-ensemble backtest <config>` (after `make install`)
- **Tests:** `packages/ta_ensemble/tests/`

Depends on `lidr_core` for the backtest engine, eval/metrics, contract writer,
data loader, and base learners. See [ADR 0001](../../docs/adr/0001-multi-model-repo-architecture.md).
