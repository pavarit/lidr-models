"""One-off PR-evidence generator for the six-signal baseline checkpoint.

Loads the latest baseline_v1 and baseline_six_signals predictions JSONs,
recomputes the strategy equity curve for each (using the same
``add_strategy_returns`` the live pipeline uses), and renders an overlaid
chart + a markdown evidence file.

Run from repo root:
    python scripts/verify_six_signal_baseline.py

Outputs:
    docs/_pr_evidence/six_signal_baseline/chart.png
    docs/_pr_evidence/six_signal_baseline/evidence.md

Per the PR-evidence convention (see CLAUDE.md), this script and its
outputs are removed in a cleanup commit before squash-merge so ``main``
stays free of review artifacts.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lidr_ml.backtest.engine import add_strategy_returns

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "_pr_evidence" / "six_signal_baseline"
PRICES_PKL = REPO / "data" / "raw" / "SPY_2005-01-01_2026-05-01.pkl"
TRANSACTION_COST_BPS = 5.0


def latest_predictions(config_name: str) -> Path:
    matches = sorted((REPO / "artifacts" / "predictions").glob(f"{config_name}-*.json"))
    if not matches:
        raise FileNotFoundError(f"No predictions JSON for {config_name}")
    return matches[-1]


def load_predictions_df(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    rows = payload["predictions"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"probability_up": "y_proba_1"})
    return df[["y_true", "y_pred", "y_proba_1"]].astype(
        {"y_true": int, "y_pred": int, "y_proba_1": float}
    )


def daily_forward_returns() -> pd.Series:
    with open(PRICES_PKL, "rb") as f:
        prices = pickle.load(f)
    # Loader normalizes columns to lowercase OHLCV; pipeline.py uses "close".
    return prices["close"].pct_change().shift(-1)


def build_equity(preds: pd.DataFrame, fwd: pd.Series) -> pd.DataFrame:
    return add_strategy_returns(preds, fwd, transaction_cost_bps=TRANSACTION_COST_BPS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fwd = daily_forward_returns()

    v1_path = latest_predictions("baseline_v1")
    six_path = latest_predictions("baseline_six_signals")
    v1 = build_equity(load_predictions_df(v1_path), fwd)
    six = build_equity(load_predictions_df(six_path), fwd)

    # Align both strategy curves to the six-signal OOS window so the chart
    # compares apples-to-apples (six_signals starts later due to 252-day breakout warmup).
    start = six.index.min()
    end = six.index.max()
    v1_aligned = v1.loc[(v1.index >= start) & (v1.index <= end)].copy()
    # Re-base both strategy curves to 1.0 at the common start so the y-axis is interpretable.
    v1_aligned["strategy_equity"] /= v1_aligned["strategy_equity"].iloc[0]
    six["strategy_equity"] /= six["strategy_equity"].iloc[0]
    six["buy_hold_equity"] /= six["buy_hold_equity"].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        six.index,
        six["buy_hold_equity"],
        label="Buy & Hold (SPY)",
        color="#444",
        linewidth=1.8,
    )
    ax.plot(
        v1_aligned.index,
        v1_aligned["strategy_equity"],
        label="Logistic, 1 signal (baseline_v1)",
        color="#1f77b4",
        linewidth=1.4,
    )
    ax.plot(
        six.index,
        six["strategy_equity"],
        label="Logistic, 6 signals (baseline_six_signals)",
        color="#d62728",
        linewidth=1.4,
    )
    ax.set_yscale("log")
    ax.set_title(
        f"SPY equity curves, OOS {start.date()} → {end.date()}  "
        "(both strategies vs buy-and-hold)"
    )
    ax.set_ylabel("Equity (log scale, common start = 1.0)")
    ax.set_xlabel("Date")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    chart_path = OUT_DIR / "chart.png"
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)

    # Final-equity snapshot from the aligned series.
    v1_final = float(v1_aligned["strategy_equity"].iloc[-1])
    six_final = float(six["strategy_equity"].iloc[-1])
    bh_final = float(six["buy_hold_equity"].iloc[-1])

    evidence = f"""# Six-signal baseline checkpoint

**Question:** does feeding all six ported signals into the existing
logistic regression beat buy-and-hold on SPY?

**Answer:** no. The six-feature model still loses to buy-and-hold by every
metric except max drawdown, and it slightly underperforms even the
single-signal `baseline_v1` floor on equity terms.

## Setup

Identical to `baseline_v1` except for the feature set:

| | baseline_v1 | baseline_six_signals |
|---|---|---|
| Signals | sma_crossover (50/200) | sma_crossover, rsi, macd, bollinger, breakout, volume |
| Target | 5-day forward return > 0 | same |
| Model | Logistic regression, `C=1.0`, `class_weight=balanced` | same |
| Backtest | Expanding-window walk-forward, 5y initial train, 12m test step | same |
| Costs | 5 bps per trade | same |
| OOS span | 2010-10-18 → 2026-04-23 (3,914 days) | 2010-12-30 → 2026-04-23 (3,862 days) |

The six-signal OOS window starts later because of the 252-day breakout warmup.
Benchmark CAGR therefore differs slightly across the two rows; only `excess_*`
is directly comparable.

## Headline comparison

| Metric | baseline_v1 | baseline_six_signals | Delta |
|---|---|---|---|
| skill_score (= 1 − log_loss / base_logloss) | -0.0380 | -0.0376 | +0.0004 (noise) |
| accuracy | 0.556 | 0.515 | -0.041 |
| strategy CAGR | 7.96% | 6.07% | -1.89 pp |
| strategy Sharpe | 0.666 | 0.582 | -0.08 |
| strategy max drawdown | -33.7% | -22.9% | +10.8 pp (shallower) |
| strategy final equity | 3.28x | 2.47x | -0.81x |
| benchmark final equity | 8.23x | 7.43x | (windows differ) |
| excess CAGR vs B&H | -6.6% | -7.9% | -1.3 pp |

The one bright spot is max drawdown: the six-feature model is in cash more
often during deep selloffs (-22.9% vs the benchmark's -33.7%). This is not
edge, but it is directional information.

## Equity curves (aligned to the six-signal OOS window, both rebased to 1.0)

![Equity curves](chart.png)

- Buy & Hold (SPY): {bh_final:.2f}x
- Logistic, 1 signal: {v1_final:.2f}x
- Logistic, 6 signals: {six_final:.2f}x

## Verdict

Edge gate stays closed. Per the plan in CLAUDE.md, next move is LightGBM
(Next Up #1) — a nonlinear learner over the same six features is the next
plausible place an edge appears. Items past the edge gate (final-model fit,
calibration, 3-class migration, artifact schema, lidr wiring) stay parked.

## Reproduction

```bash
make backtest CONFIG=configs/baseline_six_signals.yaml
python scripts/verify_six_signal_baseline.py
```

Sources:
- `{v1_path.name}`
- `{six_path.name}`
- `artifacts/results_log.csv` (rows `20260526-124439` and `20260527-120203`)
"""
    (OUT_DIR / "evidence.md").write_text(evidence, encoding="utf-8")
    print(f"Wrote {chart_path}")
    print(f"Wrote {OUT_DIR / 'evidence.md'}")


if __name__ == "__main__":
    main()
