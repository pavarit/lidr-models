"""Evaluation metrics. Kept small and dependency-light; add to it as needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def classification_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba_1: pd.Series) -> dict:
    base = float(y_true.mean())
    # base_logloss = entropy of the base rate = log loss of always predicting the
    # base rate (the no-skill floor). log_loss below this means real information.
    base_logloss = (
        float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
        if 0.0 < base < 1.0
        else 0.0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "base_rate": base,
        "pred_rate": float(y_pred.mean()),
        "base_logloss": base_logloss,
        "log_loss": float(log_loss(y_true, np.clip(y_proba_1, 1e-6, 1 - 1e-6))),
        "n_obs": int(len(y_true)),
    }


def strategy_metrics(equity_curve: pd.Series, periods_per_year: int = 252) -> dict:
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "final_equity": 0.0}
    years = len(returns) / periods_per_year
    cagr = equity_curve.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0
    cummax = equity_curve.cummax()
    drawdown = (equity_curve / cummax) - 1
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(equity_curve.iloc[-1]),
    }


def by_year(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-year classification read: accuracy vs base rate, and log loss vs the
    no-skill floor.

    `base_rate` is the share of up-days (the accuracy a naive "always up" guess
    gets). `base_logloss` is the entropy of that base rate — the log loss of a
    model that just predicts the base rate every day, i.e. the no-skill floor.
    log_loss below base_logloss means the probabilities carried information that
    year; at or above means no edge.
    """
    df = predictions.copy()
    df["year"] = df.index.year
    rows = []
    for year, g in df.groupby("year"):
        base = float(g["y_true"].mean())
        proba = np.clip(g["y_proba_1"].to_numpy(), 1e-6, 1 - 1e-6)
        ll = float(log_loss(g["y_true"].to_numpy(), proba, labels=[0, 1]))
        floor = (
            float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
            if 0.0 < base < 1.0
            else 0.0
        )
        rows.append(
            {
                "year": int(year),
                "n": int(len(g)),
                "accuracy": round(float((g["y_true"] == g["y_pred"]).mean()), 4),
                "base_rate": round(base, 4),
                "pred_rate": round(float(g["y_pred"].mean()), 4),
                "base_logloss": round(floor, 4),
                "log_loss": round(ll, 4),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def performance_by_year(preds_with_ret: pd.DataFrame) -> pd.DataFrame:
    """Per-year realized return: strategy vs buy-and-hold, and the excess.

    Answers "did this work in some years but not others?" — the regime question.
    `strategy_return` is already net of transaction costs; `fwd_return` is the
    1-day market return (the buy-and-hold leg). Each is compounded within the
    calendar year. Requires the columns produced by `add_strategy_returns`.
    """
    df = preds_with_ret.copy()
    df["year"] = df.index.year
    rows = []
    for year, g in df.groupby("year"):
        strat = float((1.0 + g["strategy_return"].fillna(0)).prod() - 1.0)
        bh = float((1.0 + g["fwd_return"].fillna(0)).prod() - 1.0)
        rows.append(
            {
                "year": int(year),
                "n": int(len(g)),
                "buy_hold_return": round(bh, 4),
                "strategy_return": round(strat, 4),
                "excess": round(strat - bh, 4),
            }
        )
    return pd.DataFrame(rows).set_index("year")
