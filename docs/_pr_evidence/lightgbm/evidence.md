# LightGBM PR — verification evidence

**Run date.** 2026-05-27.
**Config.** `configs/baseline_six_signals_lightgbm.yaml` — six features (sma_crossover,
rsi, macd, bollinger, breakout, volume), 5-day forward-return binary target,
expanding-window walk-forward (5y initial, 12mo test, 5 bps costs).
**Model.** `LightGBMModel` with conservative defaults (n_estimators=200,
learning_rate=0.05, num_leaves=31, min_child_samples=20, random_state=0).
**OOS span.** 2010-12-30 → 2026-04-23, 3,851 days.

## Full-window comparison

LightGBM is **worse** than both logistic configs on every axis.

| config | n_oos | skill_score | accuracy | base_rate | pred_rate | mean P(up) | P(up) range (1–99 %ile) | strat CAGR | bench CAGR | excess CAGR | final equity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_v1 (1 signal, logistic balanced) | 3914 | -0.0380 | 0.5565 | 0.6134 | 0.7534 | 0.5028 | 0.485 – 0.522 | +0.0796 | +0.1454 | -0.0658 | 3.28× vs 8.23× |
| six_signals (logistic balanced) | 3851 | -0.0374 | 0.5157 | 0.6110 | 0.5952 | 0.5052 | 0.434 – 0.566 | +0.0613 | +0.1404 | -0.0791 | 2.48× vs 7.44× |
| six_signals_unweighted (logistic, no reweight) | 3851 | -0.0051 | 0.6100 | 0.6110 | 0.9958 | 0.5896 | 0.522 – 0.650 | +0.1425 | +0.1404 | +0.0020 | 7.65× vs 7.44× |
| six_signals_lightgbm | 3851 | -0.1478 | 0.5378 | 0.6110 | 0.6759 | 0.5832 | 0.134 – 0.937 | +0.0928 | +0.1404 | -0.0476 | 3.88× vs 7.44× |

`skill_score = 1 − log_loss / base_logloss`. Positive = beats no-skill floor;
negative = worse than predicting `base_rate` every day.

## Charts

Equity curves for all three model strategies vs buy-and-hold — same features,
same backtest, model class varies:

![Equity curves](equity_curve.png)

P(up) distributions:

![P(up) histograms](pup_histogram.png)

## Per-period breakdown (2024 / 2025 / Q1 2026)

Strategy return is rebased to 1.0 at each period's first OOS date; trading
costs (5 bps per position change) are baked in.

### Strategy return

| period | Logistic balanced | Logistic unweighted | LightGBM | Buy & hold |
| --- | --- | --- | --- | --- |
| 2024 | +18.72% (252d) | +26.31% (252d) | +14.36% (252d) | +26.31% |
| 2025 | +13.79% (250d) | +16.77% (250d) | +22.07% (250d) | +16.77% |
| Q1 2026 | -4.46% (61d) | -4.46% (61d) | +0.03% (61d) | -4.46% |

### Classification skill within each period

`skill_score` is computed on that period's prediction slice only (not the
full-window log loss). Base rate in parentheses is the realized up-day share
within the period — useful sanity check on accuracy.

| period | metric | Logistic balanced | Logistic unweighted | LightGBM |
| --- | --- | --- | --- | --- |
| 2024 | skill_score | -0.0700 | -0.0053 | -0.1253 |
|  | accuracy (vs base rate) | 0.544 (base 0.659) | 0.659 (base 0.659) | 0.571 (base 0.659) |
|  | pred_rate (long-day share) | 0.583 | 1.000 | 0.643 |
| 2025 | skill_score | -0.0529 | -0.0061 | -0.0520 |
|  | accuracy (vs base rate) | 0.548 (base 0.636) | 0.636 (base 0.636) | 0.604 (base 0.636) |
|  | pred_rate (long-day share) | 0.696 | 1.000 | 0.792 |
| Q1 2026 | skill_score | -0.0729 | -0.1938 | -0.1915 |
|  | accuracy (vs base rate) | 0.393 (base 0.393) | 0.393 (base 0.393) | 0.426 (base 0.393) |
|  | pred_rate (long-day share) | 1.000 | 1.000 | 0.738 |

### Reading this

The per-period split is more interesting than the full-window aggregate. LightGBM's
strategy return is the **worst** in 2024 (+14% vs B&H +26%) and the **best** in 2025
(+22% vs +17%) and Q1 2026 (+0.0% vs -4.5%). The full-window aggregate hides this.

But the skill_score numbers reveal it's **not actually skill** driving the win:

- **Q1 2026** is the cleanest example. Base rate is 0.393 (the market was down
  most days). LightGBM's `skill_score = -0.192` is essentially tied with the
  unweighted logistic's `-0.194` — the probability predictions are equally bad.
  LightGBM "wins" purely because its `pred_rate = 0.738` means it sat in cash
  on 26% of days, some of which happened to be down. The unweighted logistic
  (`pred_rate = 1.000`) took the full B&H drawdown.

- **2025** has a similar story. LightGBM's `skill_score = -0.052` is *worse*
  than the unweighted logistic's `-0.006`, but LightGBM still beat B&H by
  5 pp. Again: the model isn't predicting better, it just sometimes goes to
  cash and got lucky on which days.

- **2024** punishes the same behavior — LightGBM is in cash 36% of the time
  in a strongly-up year, so it lags the always-long logistic config.

**Bottom line.** LightGBM has a structural bias toward sometimes-cash positions.
That helps when the market is choppy/down (2025, Q1 2026) and hurts when it's
up (2024). It's *not* prediction skill — `skill_score` confirms that. So the
full-window finding holds (no model has signal), but the strategy return varies
with regime in a way the full-window aggregate flattens out.


## Reproduce

```
make backtest CONFIG=configs/baseline_six_signals_lightgbm.yaml
python scripts/verify_lightgbm.py
```

The first command appends a row to `artifacts/results_log.csv` and writes a
fresh `reports/baseline_six_signals_lightgbm-<stamp>/report.html`. The second
re-renders the charts in this folder from the committed prediction JSONs.
