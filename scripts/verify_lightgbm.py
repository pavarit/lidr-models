"""Verify-evidence generator for the LightGBM PR (Next Up #1 → edge-gate
checkpoint). One-off script: produces the chart + table embedded in the PR
description so a reviewer can read the result without running the pipeline.
Deleted in the cleanup commit before squash-merge per CLAUDE.md convention.

Outputs to docs/_pr_evidence/lightgbm/:
  - equity_curve.png      LightGBM strategy vs buy-and-hold, 2010-12 → 2026-04
  - pup_histogram.png     P(up) distributions for the three six-signal configs
                          side-by-side (the key diagnostic — does LightGBM
                          spread out vs the ~0.10-wide spikes of the logistic
                          configs, and which side of 0.5 does it lean?)
  - evidence.md           comparison table vs prior runs in results_log.csv

Reads the committed prediction JSONs at artifacts/predictions/ (no yfinance,
no model re-fit) and the latest results_log.csv rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "artifacts" / "predictions"
OUT = ROOT / "docs" / "_pr_evidence" / "lightgbm"
OUT.mkdir(parents=True, exist_ok=True)

# Pinned to the rows generated for this PR (all post dup-date fix).
RUNS = {
    "baseline_v1 (1 signal, logistic balanced)": "baseline_v1-20260526-124439.json",
    "six_signals (logistic balanced)": "baseline_six_signals-20260527-135054.json",
    "six_signals_unweighted (logistic, no reweight)": "baseline_six_signals_unweighted-20260527-134952.json",
    "six_signals_lightgbm": "baseline_six_signals_lightgbm-20260527-143507.json",
}


def load_preds(filename: str) -> tuple[pd.DataFrame, dict]:
    payload = json.loads((PREDS / filename).read_text())
    df = pd.DataFrame(payload["predictions"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df, payload["metrics"]


def equity_curve(preds: pd.DataFrame, daily_fwd: pd.Series, cost_bps: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """Reproduce the pipeline's equity rule: long when y_pred=1, cash otherwise;
    cost charged on position change. See backtest/engine.py::add_strategy_returns.
    """
    position = preds["y_pred"].astype(int)
    ret = daily_fwd.reindex(preds.index).fillna(0.0)
    cost = (position.diff().abs().fillna(position.iloc[0])) * (cost_bps / 10_000)
    strat_ret = position * ret - cost
    strat_equity = (1.0 + strat_ret).cumprod()
    bh_equity = (1.0 + ret).cumprod()
    return strat_equity, bh_equity


def build_daily_fwd_from_predictions() -> pd.Series:
    """The prediction JSON doesn't include the realized return, only y_true (the
    N-day classification). To rebuild the equity curve we need the 1-day-forward
    return — which we don't have without re-loading SPY. Solution: load it from
    the loader cache (no network).
    """
    import pickle

    # Find any SPY cache; date range doesn't have to be exact, we just need the
    # 1-day-forward returns across the prediction span.
    cache_files = sorted((ROOT / "data" / "raw").glob("SPY_*.pkl"))
    if not cache_files:
        raise SystemExit(
            "No SPY cache in data/raw/. Run the LightGBM backtest at least once "
            "(make backtest CONFIG=configs/baseline_six_signals_lightgbm.yaml) "
            "to populate the cache, then re-run this script."
        )
    with cache_files[-1].open("rb") as f:
        prices = pickle.load(f)
    # Loader normalizes columns to lowercase (see CLAUDE.md Recent Changes —
    # aborted bug in the six-signal verify script's first revision).
    return prices["close"].pct_change().shift(-1)


def chart_equity(preds_lgbm: pd.DataFrame) -> None:
    daily_fwd = build_daily_fwd_from_predictions()
    strat, bh = equity_curve(preds_lgbm, daily_fwd)

    fig, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_log.plot(strat.index, strat.values, label="LightGBM strategy", color="C1", linewidth=1.5)
    ax_log.plot(bh.index, bh.values, label="Buy & hold (SPY)", color="C0", linewidth=1.5, alpha=0.8)
    ax_log.set_yscale("log")
    ax_log.set_ylabel("Equity (log scale)")
    ax_log.set_title(
        "Six-signal LightGBM vs buy-and-hold, SPY 2010-12 → 2026-04\n"
        f"Strategy final = {strat.iloc[-1]:.2f}×    B&H final = {bh.iloc[-1]:.2f}×    Excess CAGR = -4.76 pp"
    )
    ax_log.legend(loc="upper left")
    ax_log.grid(True, alpha=0.3)

    recent_mask = strat.index >= "2024-01-01"
    ax_lin.plot(strat.index[recent_mask], strat.values[recent_mask] / strat.values[recent_mask][0],
                label="LightGBM strategy", color="C1", linewidth=1.5)
    ax_lin.plot(bh.index[recent_mask], bh.values[recent_mask] / bh.values[recent_mask][0],
                label="Buy & hold (SPY)", color="C0", linewidth=1.5, alpha=0.8)
    ax_lin.set_ylabel("Equity (rebased to 1.0 at 2024-01-01)")
    ax_lin.set_title("Recent-window zoom (2024 onwards)")
    ax_lin.legend(loc="upper left")
    ax_lin.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "equity_curve.png", dpi=110)
    plt.close()


def chart_pup_histograms() -> None:
    configs = [
        ("six_signals (logistic balanced)", "baseline_six_signals-20260527-135054.json", "C3"),
        ("six_signals_unweighted (logistic none)", "baseline_six_signals_unweighted-20260527-134952.json", "C2"),
        ("six_signals_lightgbm", "baseline_six_signals_lightgbm-20260527-143507.json", "C1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, file, color) in zip(axes, configs, strict=True):
        df, _ = load_preds(file)
        p = df["probability_up"].to_numpy()
        ax.hist(p, bins=50, range=(0.0, 1.0), color=color, edgecolor="black", linewidth=0.3)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(p.mean(), color=color, linestyle="-", linewidth=2,
                   label=f"mean = {p.mean():.3f}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("P(up)")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Days (count)")
    fig.suptitle(
        "P(up) distributions — same window, same features, model class varies\n"
        "Logistic configs collapse to narrow spikes; LightGBM spreads out (capacity used) but skill_score drops further below zero",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT / "pup_histogram.png", dpi=110)
    plt.close()


def evidence_table() -> None:
    rows = []
    for label, file in RUNS.items():
        df, metrics = load_preds(file)
        p = df["probability_up"].to_numpy()
        pred_rate = float(df["y_pred"].mean())
        cls = metrics["classification"]
        strat = metrics["strategy"]
        bench = metrics["benchmark"]
        skill = 1.0 - cls["log_loss"] / cls["base_logloss"]
        rows.append({
            "config": label,
            "n_oos": len(df),
            "skill_score": f"{skill:+.4f}",
            "accuracy": f"{cls['accuracy']:.4f}",
            "base_rate": f"{cls['base_rate']:.4f}",
            "pred_rate": f"{pred_rate:.4f}",
            "mean P(up)": f"{p.mean():.4f}",
            "P(up) range (1–99 %ile)": f"{np.percentile(p, 1):.3f} – {np.percentile(p, 99):.3f}",
            "strat CAGR": f"{strat['cagr']:+.4f}",
            "bench CAGR": f"{bench['cagr']:+.4f}",
            "excess CAGR": f"{strat['cagr'] - bench['cagr']:+.4f}",
            "final equity": f"{strat['final_equity']:.2f}× vs {bench['final_equity']:.2f}×",
        })

    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
    table = "\n".join([header, sep, body])

    md = f"""# LightGBM PR — verification evidence

**Run date.** 2026-05-27.
**Config.** `configs/baseline_six_signals_lightgbm.yaml` — six features (sma_crossover,
rsi, macd, bollinger, breakout, volume), 5-day forward-return binary target,
expanding-window walk-forward (5y initial, 12mo test, 5 bps costs).
**Model.** `LightGBMModel` with conservative defaults (n_estimators=200,
learning_rate=0.05, num_leaves=31, min_child_samples=20, random_state=0).
**OOS span.** 2010-12-30 → 2026-04-23, 3,851 days.

## Headline: edge gate stays closed; LightGBM is worse than the logistic configs.

{table}

`skill_score = 1 − log_loss / base_logloss`. Positive = beats no-skill floor;
negative = worse than predicting `base_rate` every day. All four configs are
negative — none have edge — but LightGBM is the *most* negative by a wide margin.

## What happened

The unweighted logistic config (`six_signals_unweighted`) collapsed to a
near-constant predictor: P(up) sat in a ~0.10-wide spike entirely above 0.5,
pred_rate = 0.996, accuracy ≈ base_rate. Mechanically buy-and-hold minus costs.
That was the working hypothesis going in: the bottleneck was the *linear-model
assumption* — six orthogonal signals can't be combined linearly to add value
beyond the average tendency. Test: hand the same features to LightGBM and see
if nonlinear interactions reveal signal.

**Result: LightGBM uses the capacity, but to fit noise.** Its P(up) distribution
spreads from roughly 0.30 to 0.80 (vs the logistic spikes of width ~0.10), so it
*is* expressing day-to-day variation. But that variation doesn't track outcomes:
log_loss = 0.767 vs the no-skill floor 0.668 (about 5× further from zero than
the balanced logistic, ~30× further than the unweighted logistic). Confidently
wrong on many days. Pred_rate ≈ 0.50, so the strategy spends ~half its days in
cash — explains why strategy CAGR 9.3% trails B&H 14.0% by ~5 pp.

## Charts

![Equity curve](equity_curve.png)

![P(up) histograms](pup_histogram.png)

## Why this is still a useful negative result

1. **It collapses one hypothesis cleanly.** "Maybe a nonlinear learner can
   extract signal from these six features" is now tested with the conservative
   defaults that are most likely to *find* signal without overfitting (modest
   tree count, standard leaf size). It can't — at least not with this target
   formulation and this feature set.
2. **It clarifies what to try next.** The roadmap's next item (stacking) only
   makes sense if at least one base learner has skill. Neither logistic nor
   LightGBM does. So the productive next move is probably **target/feature
   reformulation** rather than more model machinery — e.g., a longer horizon
   (5d → 20d), a return-magnitude regression target instead of sign, or
   regime features (VIX level, yield-curve slope, realized vol).
3. **The P(up) histogram contrast is the most legible single chart.** "Same
   features, three models, here's what each one's probabilities look like" is
   the kind of comparison that previously required scrolling through three HTML
   reports — now it fits on one figure.

## Reproduce

```
make backtest CONFIG=configs/baseline_six_signals_lightgbm.yaml
python scripts/verify_lightgbm.py
```

The first command appends a row to `artifacts/results_log.csv` and writes a
fresh `reports/baseline_six_signals_lightgbm-<stamp>/report.html`. The second
re-renders the charts in this folder from the committed prediction JSONs.
"""
    (OUT / "evidence.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    preds_lgbm, _ = load_preds(RUNS["six_signals_lightgbm"])
    chart_equity(preds_lgbm)
    chart_pup_histograms()
    evidence_table()
    print(f"Wrote {OUT}/equity_curve.png")
    print(f"Wrote {OUT}/pup_histogram.png")
    print(f"Wrote {OUT}/evidence.md")
