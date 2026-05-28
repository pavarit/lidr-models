"""LightGBM base learner. Thin wrapper over the lightgbm sklearn API so the
rest of the pipeline doesn't import lightgbm directly — same shape as
`models/logistic.py`.

Conservative defaults chosen for the walk-forward backtest: modest tree count
and standard leaf size so early splits (smallest train windows ~5 years)
don't overfit. No early stopping — each split fits independently and we want
the same hyperparameters across every split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class LightGBMModel:
    def __init__(self, **params) -> None:
        defaults = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "random_state": 0,
            "verbose": -1,
        }
        defaults.update(params)
        self._model = LGBMClassifier(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
