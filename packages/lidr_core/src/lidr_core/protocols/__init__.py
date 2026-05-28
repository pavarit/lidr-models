"""Protocol interfaces shared across all models in the monorepo.

Today: SignalFn (ported from lidr_ml.signals.base) and Model (ported from
lidr_ml.models.base). New in Task 1: Feature and DataSource — defined here,
not yet exercised; will be used by Task 2's news-sentiment model.
"""

from lidr_core.protocols.datasource import DataSource
from lidr_core.protocols.feature import Feature
from lidr_core.protocols.model import Model
from lidr_core.protocols.signal import SignalFn

__all__ = ["SignalFn", "Model", "Feature", "DataSource"]
