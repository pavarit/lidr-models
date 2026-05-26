"""Signal registry — `name → callable` lookup.

Add a new signal by writing a module under `signals/`, decorating the function
with `@register("name")`, then importing the module (the imports at the bottom
of this file are what makes registration happen at startup).
"""

from __future__ import annotations

from collections.abc import Callable

from lidr_ml.signals.base import SignalFn

REGISTRY: dict[str, SignalFn] = {}


def register(name: str) -> Callable[[SignalFn], SignalFn]:
    """Decorator: register `fn` under `name` in the global signal registry."""

    def _wrap(fn: SignalFn) -> SignalFn:
        if name in REGISTRY:
            raise ValueError(f"Signal {name!r} already registered.")
        REGISTRY[name] = fn
        return fn

    return _wrap


def get_signal(name: str) -> SignalFn:
    if name not in REGISTRY:
        raise KeyError(f"Unknown signal {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


# Import side-effect modules so their @register decorators run.
# Every new signal module needs a line here.
from lidr_ml.signals import sma_crossover  # noqa: E402, F401
