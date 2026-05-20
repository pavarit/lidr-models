"""Logistic regression base learner. Thin wrapper over sklearn so the rest of
the pipeline doesn't import sklearn directly — makes it easy to swap in other
learners (LightGBM, XGBoost) against the same `Model` protocol later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    def __init__(self, **params) -> None:
        # Standardize features then fit — small but important; logistic regression's
        # regularization assumes features are on similar scales.
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, **params)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._pipeline.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(X)
