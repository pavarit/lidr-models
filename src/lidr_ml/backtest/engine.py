"""Walk-forward / expanding-window backtest engine.

For each split, fit on the train slice and predict on the test slice.
The model never sees data from after the test period. Predictions for all
test slices are stitched into one out-of-sample prediction series spanning
(initial_train_years) onwards.

Why custom and not just sklearn.TimeSeriesSplit? We want splits sized in
calendar units (years / months) so behavior is interpretable across
configurations and tickers with different sample counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lidr_ml.models.base import Model


@dataclass
class BacktestResult:
    predictions: pd.DataFrame  # index=date, cols=[y_true, y_pred, y_proba_1]
    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]
    # ^ (train_start, train_end, test_start, test_end) per split


def expanding_window_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,  # callable: () -> Model
    initial_train_years: int = 5,
    test_period_months: int = 12,
) -> BacktestResult:
    """Run expanding-window walk-forward backtest. Returns OOS predictions for every test date."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index.")

    idx = X.index
    if len(idx) == 0:
        raise ValueError("Empty X/y passed to backtest.")

    train_end = idx[0] + pd.DateOffset(years=initial_train_years)
    if train_end >= idx[-1]:
        raise ValueError(
            f"Not enough data for initial_train_years={initial_train_years}: "
            f"would leave nothing to test on."
        )

    preds = []
    splits = []

    while train_end < idx[-1]:
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_period_months)
        if test_end > idx[-1]:
            test_end = idx[-1]

        train_mask = idx < train_end
        test_mask = (idx >= test_start) & (idx <= test_end)

        if train_mask.sum() < 10 or test_mask.sum() < 1:
            train_end = test_end
            continue

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        model: Model = model_factory()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        # Binary problem: take the probability of class 1.
        proba_1 = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
        pred = model.predict(X_test)

        slice_df = pd.DataFrame(
            {
                "y_true": y_test.values,
                "y_pred": pred,
                "y_proba_1": proba_1,
            },
            index=y_test.index,
        )
        preds.append(slice_df)
        splits.append((idx[train_mask][0], idx[train_mask][-1], y_test.index[0], y_test.index[-1]))

        train_end = test_end

    if not preds:
        raise RuntimeError("No splits produced predictions — check config sizing.")

    return BacktestResult(predictions=pd.concat(preds).sort_index(), splits=splits)


def add_strategy_returns(
    predictions: pd.DataFrame,
    forward_returns: pd.Series,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Append columns for the strategy's equity curve.

    Simple long-only rule for the stub: hold the asset on day t if y_pred == 1,
    earn the next-day return otherwise be in cash. Transaction costs charged
    on every position change.
    """
    df = predictions.copy()
    df["fwd_return"] = forward_returns.reindex(df.index)

    position = df["y_pred"].astype(int)
    prev_position = position.shift(1).fillna(0).astype(int)
    turnover = (position != prev_position).astype(int)
    cost = turnover * (transaction_cost_bps / 10_000.0)

    strategy_ret = position * df["fwd_return"] - cost
    df["strategy_return"] = strategy_ret
    df["strategy_equity"] = (1.0 + strategy_ret.fillna(0)).cumprod()
    df["buy_hold_equity"] = (1.0 + df["fwd_return"].fillna(0)).cumprod()
    return df
