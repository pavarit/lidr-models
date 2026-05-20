"""Signal protocol.

A Signal is a pure function:
    (prices: DataFrame, params: dict) -> Series

The returned Series is aligned to the input index. Values represent the
signal's feature value at each timestamp. Signals MUST be lookahead-safe:
the value at time t may only depend on data at times <= t. This is
enforced by tests/test_no_lookahead.py.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class SignalFn(Protocol):
    def __call__(self, prices: pd.DataFrame, params: dict) -> pd.Series: ...
