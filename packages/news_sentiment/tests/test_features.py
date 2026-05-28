"""Per-feature element-wise correctness on hand-crafted item streams.

These are the layer-2 spot checks that match the pattern from
``ta_ensemble/tests/test_signal_accuracy.py`` — each test constructs a
synthetic item stream where the expected feature value is hand-derivable,
so the test pins both the math AND the alignment / point-in-time shift in
``features/_common.py``.
"""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd
from news_sentiment.features.abnormal_mention_volume import abnormal_mention_volume
from news_sentiment.features.sentiment_level import sentiment_level
from news_sentiment.features.sentiment_momentum import sentiment_momentum
from news_sentiment.types import NewsItem, ScoredItem


def _item(day: pd.Timestamp, sentiment: float, hour: int = 9) -> ScoredItem:
    return ScoredItem(
        item=NewsItem(
            ticker="X",
            published_at=datetime.combine(day.date(), time(hour, 0, 0)),
            source="t",
            title="t",
        ),
        sentiment=sentiment,
        relevance=1.0,
        confidence=1.0,
        scorer="test",
    )


def test_sentiment_level_picks_up_one_day_shift() -> None:
    days = pd.bdate_range("2024-01-01", periods=8)
    # One +1 item on the first business day; everything else empty.
    items = [_item(days[0], 1.0)]
    s = sentiment_level(items, days, {"lookback_days": 1})
    # The item is published on day[0]; per the PIT shift, it must be visible
    # at day[1], not day[0].
    assert s.iloc[0] == 0.0
    assert s.iloc[1] == 1.0
    # Lookback=1 means it falls out of the window on day[2].
    assert s.iloc[2] == 0.0


def test_sentiment_level_averages_over_lookback() -> None:
    days = pd.bdate_range("2024-01-01", periods=10)
    # Items on days[0], days[1], days[2] with sentiments 1, -1, 1.
    items = [
        _item(days[0], 1.0),
        _item(days[1], -1.0),
        _item(days[2], 1.0),
    ]
    s = sentiment_level(items, days, {"lookback_days": 3})
    # day[3] window covers shifted days[1..3] → sentiments {1, -1, 1} → mean 1/3.
    assert np.isclose(s.iloc[3], 1.0 / 3.0)


def test_sentiment_momentum_sign_matches_recent_minus_old() -> None:
    days = pd.bdate_range("2024-01-01", periods=15)
    items = (
        [_item(days[i], -1.0) for i in range(5)]
        + [_item(days[i], 1.0) for i in range(5, 10)]
    )
    s = sentiment_momentum(items, days, {"fast": 2, "slow": 6})
    # Late in the window fast MA should be positive, slow MA mixed → positive.
    assert s.iloc[10] > 0


def test_abnormal_mention_volume_z_score_on_spike() -> None:
    days = pd.bdate_range("2024-01-01", periods=80)
    # Steady ~1 item/day for 60 days, then a 20-item spike on day 60.
    items: list[ScoredItem] = []
    for i in range(60):
        items.append(_item(days[i], 0.0))
    for _ in range(20):
        items.append(_item(days[60], 0.0))
    s = abnormal_mention_volume(items, days, {"baseline_days": 30})
    # Per the PIT shift, the spike is observable at day[61].
    assert s.iloc[61] > 3.0, f"expected a large z-score on the spike, got {s.iloc[61]}"
    # And the days before the spike should sit near zero.
    assert abs(s.iloc[50]) < 1.0
