"""Assert that every registered news feature is lookahead-safe.

The rule (mechanised by ``features/_common.py::align_to_trading_days``):
the feature value at trading day ``t`` may only depend on items with
``published_at < t.normalize()``. We check it by computing each feature
twice — once over the full scored-item stream, once over only the items
published strictly before a check date ``t`` — and asserting the values at
``t`` match.

When adding a feature, add an entry to ``FEATURE_CASES`` below. The lookahead
test is non-negotiable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from news_sentiment.features import get_feature
from news_sentiment.types import ScoredItem

# Adding a feature? Add a (name, params) entry here.
FEATURE_CASES = [
    ("sentiment_level", {"lookback_days": 5}),
    ("sentiment_momentum", {"fast": 3, "slow": 10}),
    ("abnormal_mention_volume", {"baseline_days": 30}),
]


@pytest.mark.parametrize("name,params", FEATURE_CASES)
def test_news_feature_has_no_lookahead(
    name: str,
    params: dict,
    synthetic_scored_items: list[ScoredItem],
    trading_days: pd.DatetimeIndex,
) -> None:
    fn = get_feature(name)
    full = fn(synthetic_scored_items, trading_days, params)

    check_dates = trading_days[[120, 240, 360, 480]]
    for t in check_dates:
        truncated_items = [it for it in synthetic_scored_items if it.published_at < t]
        partial = fn(truncated_items, trading_days, params)
        if pd.isna(full.loc[t]):
            assert pd.isna(partial.loc[t]), f"{name}: full NaN but truncated populated at {t}"
        else:
            assert np.isclose(full.loc[t], partial.loc[t], equal_nan=False), (
                f"{name} leaks future info at {t}: "
                f"full={full.loc[t]}, truncated={partial.loc[t]}"
            )
