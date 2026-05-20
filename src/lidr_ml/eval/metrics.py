"""Evaluation metrics. Kept small and dependency-light; add to it as needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def classification_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba_1: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba_1, 1e-6, 1 - 1e-6))),
        "base_rate": float(y_true.mean()),
        "pred_rate": float(y_pred.mean()),
        "n_obs": int(len(y_true)),
    }


def strategy_metrics(equity_curve: pd.Series, periods_per_year: int = 252) -> dict:
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
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
    """Per-year accuracy + hit rate."""
    df = predictions.copy()
    df["year"] = df.index.year
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    grouped = df.groupby("year").agg(
        n=("y_true", "size"),
        accuracy=("correct", "mean"),
        base_rate=("y_true", "mean"),
        pred_rate=("y_pred", "mean"),
    )
    return grouped.round(4)
