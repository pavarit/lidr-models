# ta_ensemble configs

YAML experiment configs for the TA model. Each declares tickers, date range, data source, signals, target, model, and backtest settings — see [README → Config schema](../../../README.md#config-schema) for the field reference. Run one with:

```bash
make backtest CONFIG=packages/ta_ensemble/configs/<file>
```

## Current state

No config beats buy-and-hold yet; the closest to the no-skill floor is **`baseline_six_signals_unweighted.yaml`** (`skill_score` ≈ −0.005). See [README → Current status](../../../README.md#current-status-at-a-glance).

## Baselines

| Config | Signals | Model | Notes |
| --- | --- | --- | --- |
| `baseline.yaml` | sma_crossover only | logistic | The single-signal first bar (`name: baseline_v1`). Drives the committed sample report + README headline; `make refresh-sample-report` runs this one. |
| `baseline_six_signals_unweighted.yaml` | all six | logistic | **Current best** (least far from no-skill). New configs should default to `class_weight=None`. |
| `baseline_six_signals.yaml` | all six | logistic, `class_weight=balanced` | The balanced-weight six-signal run. Balanced weighting turned out *actively harmful* — see CLAUDE.md → Gotchas. |
| `baseline_six_signals_lightgbm.yaml` | all six | LightGBM | The LightGBM checkpoint — worse than logistic (`skill_score` −0.148 raw). |
| `dev_synthetic.yaml` | synthetic | logistic | Offline smoke config: synthetic prices, no internet, `output.results_log: false` so CI runs don't pollute the tracked log. Used by the smoke test. |

## Horizon sweep (`horizon_h{N}_{model}.yaml`, 12 files)

The 2026-05-28 spike: `target.horizon_days ∈ {5, 10, 20, 60}` crossed with `{logistic, logistic_weighted, lightgbm}`. Result: longer horizons make skill monotonically *worse*; h5 is least-bad everywhere. **Lever closed — don't re-run on this feature set.** Full finding in CLAUDE.md → Recent Changes → horizon spike.

| horizon | logistic | logistic_weighted | lightgbm |
| --- | --- | --- | --- |
| h5  | `horizon_h5_logistic` | `horizon_h5_logistic_weighted` | `horizon_h5_lightgbm` |
| h10 | `horizon_h10_logistic` | `horizon_h10_logistic_weighted` | `horizon_h10_lightgbm` |
| h20 | `horizon_h20_logistic` | `horizon_h20_logistic_weighted` | `horizon_h20_lightgbm` |
| h60 | `horizon_h60_logistic` | `horizon_h60_logistic_weighted` | `horizon_h60_lightgbm` |
