"""Model protocol — small enough that base learners and the stacker share it."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # shape (n_samples, n_classes)
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
