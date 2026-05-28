"""Artifact writer.

Builds the schema_version: 2 JSON artifact a model produces for lidr to consume,
validates it against ``schema/artifact.schema.json``, and writes it to disk.

Validation is **mandatory** on write — a model that cannot produce a valid
artifact should fail loudly, not write a half-broken file that lidr will then
have to special-case. Validation uses ``jsonschema`` when available; if the
package is not installed the writer falls back to a minimal in-tree check
covering required keys and a small set of structural invariants (schema_version,
date format, recommendation enum, probability range). The in-tree path is
enough to keep tests green in a lean install; production runs should have
``jsonschema`` installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "artifact.schema.json"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RECOMMENDATIONS = {"BUY", "HOLD", "SELL"}


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def build_artifact(
    *,
    model_id: str,
    model_version: str,
    config_name: str,
    ticker: str,
    generated_at: str,
    cls_m: dict,
    strat_m: dict,
    bench_m: dict,
    predictions: pd.DataFrame,
    buy_band: float = 0.55,
    sell_band: float = 0.45,
) -> dict[str, Any]:
    """Assemble the v2 artifact dict from the pieces a pipeline already produces.

    ``predictions`` is the stitched OOS DataFrame from the backtest — index of
    dates, columns ``y_true`` / ``y_pred`` / ``y_proba_1``. The BUY/HOLD/SELL
    recommendation is derived from ``y_proba_1`` using simple bands until the
    3-class target migration replaces this with a true class label.
    """
    payload = {
        "schema_version": 2,
        "model_id": model_id,
        "model_version": model_version,
        "config_name": config_name,
        "ticker": ticker,
        "generated_at": generated_at,
        "metrics": {
            "classification": _coerce(cls_m),
            "strategy": _coerce(strat_m),
            "benchmark": _coerce(bench_m),
        },
        "predictions": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "recommendation": _band_to_recommendation(
                    float(row["y_proba_1"]), buy_band=buy_band, sell_band=sell_band
                ),
                "probability_up": float(row["y_proba_1"]),
                "y_pred": int(row["y_pred"]),
                "y_true": (
                    None
                    if pd.isna(row["y_true"])
                    else int(row["y_true"])
                ),
            }
            for d, row in predictions.iterrows()
        ],
    }
    return payload


def write_artifact(
    payload: dict,
    out_path: Path,
) -> Path:
    """Validate then write the artifact. Returns the resolved output path."""
    validate_artifact(payload)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def validate_artifact(payload: dict) -> None:
    """Raise ``ValueError`` (or jsonschema.ValidationError) if ``payload`` is invalid.

    Tries ``jsonschema`` first for full structural validation; falls back to an
    in-tree check that covers required keys + a few structural invariants
    (schema_version constant, date format, recommendation enum, probability_up
    range). The fallback is intentionally narrow — it is a safety net, not a
    substitute for the JSON Schema.
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        _fallback_validate(payload)
        return

    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.validate(instance=payload, schema=schema)


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #

def _band_to_recommendation(p: float, *, buy_band: float, sell_band: float) -> str:
    if p >= buy_band:
        return "BUY"
    if p <= sell_band:
        return "SELL"
    return "HOLD"


def _coerce(d: dict) -> dict:
    """Convert numpy scalars to plain Python types so json.dumps works."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def _fallback_validate(payload: dict) -> None:
    required_top = {
        "schema_version",
        "model_id",
        "model_version",
        "config_name",
        "ticker",
        "generated_at",
        "metrics",
        "predictions",
    }
    missing = required_top - payload.keys()
    if missing:
        raise ValueError(f"Artifact missing required keys: {sorted(missing)}")

    if payload["schema_version"] != 2:
        raise ValueError(
            f"schema_version must be 2, got {payload['schema_version']!r}"
        )

    if not isinstance(payload["predictions"], list) or not payload["predictions"]:
        raise ValueError("predictions must be a non-empty list")

    for i, p in enumerate(payload["predictions"]):
        for key in ("date", "recommendation", "probability_up", "y_pred"):
            if key not in p:
                raise ValueError(f"prediction[{i}] missing required key {key!r}")
        if not _DATE_RE.match(p["date"]):
            raise ValueError(f"prediction[{i}].date {p['date']!r} not YYYY-MM-DD")
        if p["recommendation"] not in _RECOMMENDATIONS:
            raise ValueError(
                f"prediction[{i}].recommendation {p['recommendation']!r} "
                f"not one of {sorted(_RECOMMENDATIONS)}"
            )
        prob = p["probability_up"]
        if not (isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0):
            raise ValueError(
                f"prediction[{i}].probability_up {prob!r} not in [0, 1]"
            )
