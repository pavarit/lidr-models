"""News-feature registry. Mirrors the signals registry in ta_ensemble.

Adding a feature: write a module under ``features/``, decorate the function
with ``@register("name")``, then import the module at the bottom of
``registry.py`` so the decorator runs at startup.
"""

from __future__ import annotations

from news_sentiment.features.registry import REGISTRY, get_feature, register

__all__ = ["REGISTRY", "get_feature", "register"]
