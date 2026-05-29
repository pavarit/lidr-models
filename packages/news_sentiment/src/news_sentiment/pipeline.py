"""End-to-end pipeline for news_sentiment.

Mirrors ``ta_ensemble.pipeline.run_pipeline`` in shape so the two models are
strictly comparable through ``results_log`` and the schema-v2 artifact. The
news side replaces the "signals on prices" stage with
"collect → score → features on item streams"; everything from the feature
matrix forward (target, backtest, eval, report, artifact, leaderboard) reuses
lidr_core.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from lidr_core.backtest.engine import add_strategy_returns, expanding_window_backtest
from lidr_core.contract.writer import build_artifact, write_artifact
from lidr_core.data.loaders import DataConfig, load_prices
from lidr_core.eval.leaderboard import write_manifest
from lidr_core.eval.metrics import (
    by_year,
    classification_metrics,
    performance_by_year,
    strategy_metrics,
)
from lidr_core.eval.report import write_report
from lidr_core.eval.results_log import append_run
from lidr_core.models import build_model

from news_sentiment.datasources import build_source
from news_sentiment.features import get_feature
from news_sentiment.ingest.collector import collect
from news_sentiment.scoring import build_scorer

# packages/news_sentiment/src/news_sentiment/pipeline.py → parents[4] is repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class PipelineResult:
    report_path: Path
    predictions_path: Path | None


def run_pipeline(config_path: Path) -> PipelineResult:
    config = yaml.safe_load(Path(config_path).read_text())
    name = config["name"]
    print(f"\n=== Running news_sentiment pipeline: {name} ===")

    # 1. Prices --------------------------------------------------------------
    data_cfg = DataConfig.from_dict(config["data"])
    cache_dir = PROJECT_ROOT / "data" / "raw"
    prices_by_ticker = load_prices(data_cfg, cache_dir=cache_dir)
    if len(prices_by_ticker) != 1:
        raise NotImplementedError(
            "news_sentiment pipeline supports one ticker at a time. "
            "Cross-sectional rollups come later."
        )
    ticker, prices = next(iter(prices_by_ticker.items()))
    print(
        f"  Loaded {len(prices)} rows for {ticker} "
        f"({prices.index.min().date()} → {prices.index.max().date()})"
    )

    # 2. News ingestion ------------------------------------------------------
    sources_cfg = config["news"]["sources"]
    sources = [build_source(s["name"], **(s.get("params") or {})) for s in sources_cfg]
    news_cache = PROJECT_ROOT / "data" / "news" / name
    items = collect(
        sources=sources,
        ticker=ticker,
        start=str(data_cfg.start_date),
        end=str(data_cfg.end_date),
        cache_dir=news_cache,
    )
    print(f"  Collected {len(items)} items from {len(sources)} source(s) → cache {news_cache}")

    # 3. Scoring -------------------------------------------------------------
    scorer_cfg = dict(config["news"]["scorer"])
    scorer_name = scorer_cfg.pop("name")
    # LLM-backed scorers want the spend log + cache + run_id wired through.
    # The hybrid scorer forwards these into its LLM sub-scorer.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if scorer_name in ("llm", "hybrid"):
        scorer_cfg.setdefault(
            "cache_path", PROJECT_ROOT / "data" / "news" / "_llm_cache.jsonl"
        )
        scorer_cfg.setdefault(
            "spend_log_path", PROJECT_ROOT / "artifacts" / "llm_spend.csv"
        )
        scorer_cfg.setdefault("run_id", stamp)
    scorer = build_scorer(scorer_name, **scorer_cfg)
    scored = scorer.score(items)
    if scored:
        sample_sent = sum(s.sentiment for s in scored) / len(scored)
        print(f"  Scored {len(scored)} items via {scorer_name} (mean sentiment {sample_sent:+.3f})")
    else:
        print(f"  Scored 0 items via {scorer_name}")

    # 4. Features ------------------------------------------------------------
    feature_series: list[pd.Series] = []
    for feat_cfg in config["features"]:
        fn = get_feature(feat_cfg["name"])
        s = fn(scored, prices.index, feat_cfg.get("params", {}) or {})
        feature_series.append(s)
    X = pd.concat(feature_series, axis=1)
    print(f"  Computed {X.shape[1]} feature(s) → matrix shape {X.shape}")

    # 5. Target (same shape as ta_ensemble) ----------------------------------
    target_cfg = config["target"]
    if target_cfg["type"] != "forward_return_binary":
        raise NotImplementedError(f"Target type {target_cfg['type']!r} not yet supported.")
    horizon = int(target_cfg["horizon_days"])
    threshold = float(target_cfg.get("threshold", 0.0))
    fwd_return = prices["close"].pct_change(horizon).shift(-horizon)
    y = (fwd_return > threshold).astype(int)
    y.name = "y"

    aligned = pd.concat([X, y.rename("__y__"), fwd_return.rename("__fwd__")], axis=1).dropna()
    X_clean = aligned[X.columns]
    y_clean = aligned["__y__"].astype(int)
    print(
        f"  After alignment + dropna: {len(X_clean)} usable rows, "
        f"base rate {y_clean.mean():.3f}"
    )

    # 6. Backtest ------------------------------------------------------------
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
    print(
        f"  Backtest produced {len(result.predictions)} OOS predictions "
        f"across {len(result.splits)} splits"
    )

    daily_fwd_return = prices["close"].pct_change().shift(-1)
    preds_with_ret = add_strategy_returns(
        result.predictions,
        forward_returns=daily_fwd_return.reindex(result.predictions.index),
        transaction_cost_bps=float(bt_cfg.get("transaction_cost_bps", 5.0)),
    )

    # 7. Metrics + report ----------------------------------------------------
    cls_m = classification_metrics(
        result.predictions["y_true"],
        result.predictions["y_pred"],
        result.predictions["y_proba_1"],
    )
    strat_m = strategy_metrics(preds_with_ret["strategy_equity"])
    bench_m = strategy_metrics(preds_with_ret["buy_hold_equity"])
    yr = by_year(result.predictions)
    perf_yr = performance_by_year(preds_with_ret)

    if config.get("output", {}).get("results_log", True):
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
            skill_floor = cls_m.get("base_logloss", 0) or 0
            skill_str = (
                f"{1.0 - cls_m['log_loss'] / skill_floor:.4f}" if skill_floor > 0 else "n/a"
            )
            excess = (strat_m.get("cagr") or 0.0) - (bench_m.get("cagr") or 0.0)
            print(
                f"  Results log → {log_path.relative_to(PROJECT_ROOT)} "
                f"(skill_score={skill_str}, excess_cagr={excess:+.4f})"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not write results log: {exc}")
    else:
        print("  Results log → skipped (output.results_log: false)")

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

    # 8. Schema-v2 artifact + manifest refresh -------------------------------
    predictions_path: Path | None = None
    if config.get("output", {}).get("predictions_json", False):
        model_id = config.get("model_id", "news_sentiment")
        model_version = config.get("model_version", "0.1.0")
        artifacts_dir = PROJECT_ROOT / "artifacts" / "predictions" / model_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = artifacts_dir / f"{name}-{stamp}.json"
        payload = build_artifact(
            model_id=model_id,
            model_version=model_version,
            config_name=name,
            ticker=ticker,
            generated_at=stamp,
            cls_m=cls_m,
            strat_m=strat_m,
            bench_m=bench_m,
            predictions=result.predictions,
        )
        write_artifact(payload, predictions_path)

        if config.get("output", {}).get("refresh_manifest", True):
            try:
                manifest_path = PROJECT_ROOT / "artifacts" / "manifest.json"
                write_manifest(PROJECT_ROOT / "artifacts" / "predictions", manifest_path)
                print(f"  Manifest → {manifest_path.relative_to(PROJECT_ROOT)}")
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: manifest refresh failed: {exc}")

    return PipelineResult(report_path=report_path, predictions_path=predictions_path)
