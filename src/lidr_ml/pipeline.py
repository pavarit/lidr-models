"""End-to-end pipeline: config → data → signals → target → backtest → report.

Read this module top-to-bottom to understand the system. Every other file in
the package is invoked from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lidr_ml.backtest.engine import add_strategy_returns, expanding_window_backtest
from lidr_ml.data.loaders import DataConfig, load_prices
from lidr_ml.eval.metrics import (
    by_year,
    classification_metrics,
    performance_by_year,
    strategy_metrics,
)
from lidr_ml.eval.report import write_report
from lidr_ml.eval.results_log import append_run
from lidr_ml.models import build_model
from lidr_ml.signals import get_signal

# Project root = parent of `src/`. Resolves correctly when installed editable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PipelineResult:
    report_path: Path
    predictions_path: Path | None


def run_pipeline(config_path: Path) -> PipelineResult:
    config = yaml.safe_load(Path(config_path).read_text())
    name = config["name"]
    print(f"\n=== Running pipeline: {name} ===")

    # 1. Data ----------------------------------------------------------------
    data_cfg = DataConfig.from_dict(config["data"])
    cache_dir = PROJECT_ROOT / "data" / "raw"
    prices_by_ticker = load_prices(data_cfg, cache_dir=cache_dir)

    # Stub: single-ticker pipeline. Multi-ticker comes when we add cross-sectional features.
    if len(prices_by_ticker) != 1:
        raise NotImplementedError(
            "Stub pipeline supports one ticker at a time. Multi-ticker comes later."
        )
    ticker, prices = next(iter(prices_by_ticker.items()))
    print(f"  Loaded {len(prices)} rows for {ticker} ({prices.index.min().date()} → {prices.index.max().date()})")

    # 2. Signals → feature matrix --------------------------------------------
    feature_series: list[pd.Series] = []
    for sig_cfg in config["signals"]:
        fn = get_signal(sig_cfg["name"])
        s = fn(prices, sig_cfg.get("params", {}) or {})
        feature_series.append(s)
    X = pd.concat(feature_series, axis=1)
    print(f"  Computed {X.shape[1]} signal(s) → feature matrix shape {X.shape}")

    # 3. Target --------------------------------------------------------------
    target_cfg = config["target"]
    if target_cfg["type"] != "forward_return_binary":
        raise NotImplementedError(f"Target type {target_cfg['type']!r} not yet supported.")
    horizon = int(target_cfg["horizon_days"])
    threshold = float(target_cfg.get("threshold", 0.0))
    fwd_return = prices["close"].pct_change(horizon).shift(-horizon)
    y = (fwd_return > threshold).astype(int)
    y.name = "y"

    # Align + drop NaNs (warmup from signal lookback + forward-return tail).
    aligned = pd.concat([X, y.rename("__y__"), fwd_return.rename("__fwd__")], axis=1).dropna()
    X_clean = aligned[X.columns]
    y_clean = aligned["__y__"].astype(int)
    fwd_clean = aligned["__fwd__"]
    print(f"  After alignment + dropna: {len(X_clean)} usable rows, base rate {y_clean.mean():.3f}")

    # 4. Backtest ------------------------------------------------------------
    bt_cfg = config["backtest"]
    if bt_cfg.get("cv") != "expanding_window":
        raise NotImplementedError(f"CV {bt_cfg.get('cv')!r} not yet supported.")
    model_spec = config["model"]

    def model_factory():
        return build_model(model_spec)

    result = expanding_window_backtest(
        X_clean,
        y_clean,
        model_factory=model_factory,
        initial_train_years=int(bt_cfg["initial_train_years"]),
        test_period_months=int(bt_cfg["test_period_months"]),
    )
    print(f"  Backtest produced {len(result.predictions)} OOS predictions across {len(result.splits)} splits")

    # The equity curve is marked to market with 1-day-forward returns, NOT the
    # N-day classification target. The model takes a position on day t and earns
    # the return realized from t to t+1. Using the N-day forward return here
    # would compound the same multi-day window once per day — overlapping ~N
    # times and badly inflating the curve. The N-day return stays purely as the
    # classification target (`fwd_clean` above); it must not drive the equity.
    daily_fwd_return = prices["close"].pct_change().shift(-1)
    preds_with_ret = add_strategy_returns(
        result.predictions,
        forward_returns=daily_fwd_return.reindex(result.predictions.index),
        transaction_cost_bps=float(bt_cfg.get("transaction_cost_bps", 5.0)),
    )

    # 5. Metrics + report ----------------------------------------------------
    cls_m = classification_metrics(
        result.predictions["y_true"],
        result.predictions["y_pred"],
        result.predictions["y_proba_1"],
    )
    strat_m = strategy_metrics(preds_with_ret["strategy_equity"])
    bench_m = strategy_metrics(preds_with_ret["buy_hold_equity"])
    yr = by_year(result.predictions)
    perf_yr = performance_by_year(preds_with_ret)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Log this run to the cross-run results CSV so "did this change help?" is
    # answerable without opening individual HTML reports.
    log_path = PROJECT_ROOT / "artifacts" / "results_log.csv"
    try:
        out_dir_for_log = PROJECT_ROOT / "reports" / f"{name}-{stamp}"
        append_run(
            log_path=log_path,
            run_id=stamp,
            config_name=name,
            ticker=ticker,
            predictions=result.predictions,
            cls_m=cls_m,
            strat_m=strat_m,
            bench_m=bench_m,
            by_year_df=yr,
            report_path=out_dir_for_log / "report.html",
            project_root=PROJECT_ROOT,
        )
        skill = cls_m.get("base_logloss", 0) or 0
        skill_str = f"{1.0 - cls_m['log_loss'] / skill:.4f}" if skill > 0 else "n/a"
        excess = (strat_m.get("cagr") or 0.0) - (bench_m.get("cagr") or 0.0)
        print(f"  Results log → {log_path.relative_to(PROJECT_ROOT)}  "
              f"(skill_score={skill_str}, excess_cagr={excess:+.4f})")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not write results log: {exc}")

    out_dir = PROJECT_ROOT / "reports" / f"{name}-{stamp}"
    report_path = write_report(
        out_dir=out_dir,
        config_name=name,
        config_dict=config,
        classification_metrics=cls_m,
        strategy_metrics=strat_m,
        benchmark_metrics=bench_m,
        predictions_with_returns=preds_with_ret,
        by_year_df=yr,
        performance_by_year_df=perf_yr,
    )
    print(f"  Report → {report_path}")

    # 6. Predictions JSON (the artifact lidr's API route will eventually read) -
    predictions_path: Path | None = None
    if config.get("output", {}).get("predictions_json", False):
        artifacts_dir = PROJECT_ROOT / "artifacts" / "predictions"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = artifacts_dir / f"{name}-{stamp}.json"
        payload = {
            "schema_version": 1,
            "config_name": name,
            "ticker": ticker,
            "generated_at": stamp,
            "metrics": {"classification": cls_m, "strategy": strat_m, "benchmark": bench_m},
            "predictions": [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "probability_up": float(row["y_proba_1"]),
                }
                for d, row in result.predictions.iterrows()
            ],
        }
        predictions_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    return PipelineResult(report_path=report_path, predictions_path=predictions_path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Not JSON-serializable: {type(o)!r}")
