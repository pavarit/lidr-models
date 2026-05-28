"""Cross-run results log: append one CSV row per pipeline run.

Usage (from pipeline.py)::

    from lidr_core.eval.results_log import append_run
    append_run(
        log_path=PROJECT_ROOT / "artifacts" / "results_log.csv",
        run_id=stamp,
        config_name=name,
        ticker=ticker,
        predictions=result.predictions,   # OOS DataFrame: y_true, y_pred, y_proba_1
        cls_m=cls_m,
        strat_m=strat_m,
        bench_m=bench_m,
        by_year_df=yr,                    # pd.DataFrame indexed by year
        report_path=report_path,
        project_root=PROJECT_ROOT,
    )

The log lives at ``artifacts/results_log.csv`` and is intentionally git-tracked
so experiment history accumulates across sessions. Edit it directly to remove bad
rows (e.g. a buggy run). No opt-out needed in normal usage — you always want this.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss

# --------------------------------------------------------------------------- #
# Columns — order here matches the written CSV exactly.                        #
# --------------------------------------------------------------------------- #

_COLUMNS = [
    "run_id",
    "config_name",
    "ticker",
    "oos_start",
    "oos_end",
    "n_oos",
    "skill_score",
    "base_logloss",
    "log_loss",
    "accuracy",
    "base_logloss_2025",
    "log_loss_2025",
    "base_logloss_2026q1",
    "log_loss_2026q1",
    "strategy_cagr",
    "strategy_sharpe",
    "strategy_max_dd",
    "strategy_final_equity",
    "bench_cagr",
    "bench_sharpe",
    "bench_max_dd",
    "bench_final_equity",
    "excess_cagr",
    "excess_sharpe",
    "report_path",
]


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #

def _base_logloss(base_rate: float) -> float | None:
    """Entropy of ``base_rate`` — the no-skill floor for log loss."""
    if base_rate <= 0.0 or base_rate >= 1.0:
        return None
    return -(base_rate * math.log(base_rate) + (1 - base_rate) * math.log(1 - base_rate))


def _period_metrics(predictions: pd.DataFrame, year: int, quarter: int | None = None) -> dict:
    """Compute log_loss and base_logloss for a slice of OOS predictions.

    Returns ``{"log_loss": float | None, "base_logloss": float | None}``.
    Both are None if the slice is too small or degenerate (only one class).
    """
    mask = predictions.index.year == year
    if quarter is not None:
        mask &= predictions.index.quarter == quarter
    sub = predictions.loc[mask]

    if len(sub) < 2 or sub["y_true"].nunique() < 2:
        return {"log_loss": None, "base_logloss": None}

    base = float(sub["y_true"].mean())
    proba = np.clip(sub["y_proba_1"].to_numpy(), 1e-6, 1 - 1e-6)
    ll = float(sk_log_loss(sub["y_true"].to_numpy(), proba, labels=[0, 1]))
    bl = _base_logloss(base)
    return {"log_loss": ll, "base_logloss": bl}


def _fmt(value: float | None, decimals: int = 6) -> str:
    """Format a float for CSV, or empty string for None/NaN."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{round(value, decimals)}"


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def append_run(
    *,
    log_path: Path,
    run_id: str,
    config_name: str,
    ticker: str,
    predictions: pd.DataFrame,
    cls_m: dict,
    strat_m: dict,
    bench_m: dict,
    by_year_df: pd.DataFrame,
    report_path: Path,
    project_root: Path,
) -> None:
    """Append one row to the cross-run results CSV at ``log_path``.

    Creates the file with a header row if it does not yet exist.
    The write is non-fatal: callers should wrap in try/except and warn on failure
    so a disk error never aborts a backtest run.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    needs_header = not log_path.exists() or log_path.stat().st_size == 0

    # --- OOS date range --------------------------------------------------------
    oos_start = predictions.index.min().date()
    oos_end = predictions.index.max().date()
    n_oos = len(predictions)

    # --- Skill score -----------------------------------------------------------
    base_ll = cls_m.get("base_logloss")
    ll = cls_m.get("log_loss")
    if base_ll and base_ll > 0.0:
        skill_score: float | None = 1.0 - ll / base_ll
    else:
        skill_score = None

    # --- Per-period log loss ---------------------------------------------------
    # 2025: pull from by_year_df (already computed, consistent with report table)
    if 2025 in by_year_df.index:
        row_2025 = by_year_df.loc[2025]
        bl_2025: float | None = float(row_2025["base_logloss"])
        ll_2025: float | None = float(row_2025["log_loss"])
    else:
        bl_2025 = None
        ll_2025 = None

    # Q1 2026: compute from raw predictions slice
    q1_2026 = _period_metrics(predictions, year=2026, quarter=1)
    bl_2026q1 = q1_2026["base_logloss"]
    ll_2026q1 = q1_2026["log_loss"]

    # --- Relative report path (keeps CSV portable across machines) -------------
    try:
        rel_report = str(report_path.relative_to(project_root))
    except ValueError:
        rel_report = str(report_path)

    # --- Assemble row ----------------------------------------------------------
    row = {
        "run_id": run_id,
        "config_name": config_name,
        "ticker": ticker,
        "oos_start": str(oos_start),
        "oos_end": str(oos_end),
        "n_oos": n_oos,
        "skill_score": _fmt(skill_score),
        "base_logloss": _fmt(base_ll),
        "log_loss": _fmt(ll),
        "accuracy": _fmt(cls_m.get("accuracy")),
        "base_logloss_2025": _fmt(bl_2025),
        "log_loss_2025": _fmt(ll_2025),
        "base_logloss_2026q1": _fmt(bl_2026q1),
        "log_loss_2026q1": _fmt(ll_2026q1),
        "strategy_cagr": _fmt(strat_m.get("cagr")),
        "strategy_sharpe": _fmt(strat_m.get("sharpe")),
        "strategy_max_dd": _fmt(strat_m.get("max_drawdown")),
        "strategy_final_equity": _fmt(strat_m.get("final_equity")),
        "bench_cagr": _fmt(bench_m.get("cagr")),
        "bench_sharpe": _fmt(bench_m.get("sharpe")),
        "bench_max_dd": _fmt(bench_m.get("max_drawdown")),
        "bench_final_equity": _fmt(bench_m.get("final_equity")),
        "excess_cagr": _fmt(
            (strat_m.get("cagr") or 0.0) - (bench_m.get("cagr") or 0.0)
        ),
        "excess_sharpe": _fmt(
            (strat_m.get("sharpe") or 0.0) - (bench_m.get("sharpe") or 0.0)
        ),
        "report_path": rel_report,
    }

    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
