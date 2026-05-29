"""Leaderboard / manifest writer.

Scans ``artifacts/predictions/<model_id>/*.json`` for produced artifacts and
writes a ``manifest.json`` lidr reads to discover the available models and each
model's headline OOS score.

Selection rule (per model_id):

* **Smoke / dev runs are never eligible.** Runs on synthetic data (config_name
  containing ``synthetic`` or ticker ``FAKE``) exist only to exercise the
  pipeline offline; their metrics are meaningless, so they must not headline a
  model. A model whose *only* artifacts are smoke runs is omitted from the
  manifest entirely — it has no real result to advertise yet (e.g.
  ``news_sentiment`` until its first real backtest lands).
* **Among the real runs, the most recent wins**, ordered by the artifact's
  embedded ``generated_at`` timestamp (falling back to file mtime only if that
  can't be parsed). Using the embedded timestamp rather than mtime means a
  ``git checkout``, copy, or restore that rewrites file mtimes can't silently
  reorder which run counts as "latest."

This replaced an earlier rule that picked the most-recently-*touched* file by
mtime alone, which let a freshly-run ``dev_synthetic`` smoke test headline the
model (and did — the committed manifest pointed at a synthetic run).

The manifest is schema_version: 2; bump when adding fields that lidr's reader
treats as required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lidr_core.contract.loader import load_artifact

# Synthetic / offline-smoke runs use this ticker (see configs/dev_synthetic.yaml).
_SMOKE_TICKERS = {"FAKE"}


def build_manifest(predictions_root: Path) -> dict:
    """Return the manifest dict for every model_id under ``predictions_root``.

    ``predictions_root`` is conventionally ``artifacts/predictions/``. Each
    immediate subdirectory is treated as a model_id; the model's headline
    artifact is chosen per the selection rule documented at the top of this
    module (skip smoke runs, then latest real run by embedded timestamp). A
    model with no real run is omitted.
    """
    predictions_root = Path(predictions_root)
    models = []

    if predictions_root.exists():
        for model_dir in sorted(p for p in predictions_root.iterdir() if p.is_dir()):
            chosen = _latest_real_artifact(model_dir)
            if chosen is None:
                continue
            payload, latest = chosen
            models.append(_model_entry(payload, latest, predictions_root))

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "models": models,
    }


def write_manifest(predictions_root: Path, out_path: Path) -> Path:
    """Build the manifest from ``predictions_root`` and write it to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_manifest(predictions_root), indent=2))
    return out_path


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #

def _latest_real_artifact(model_dir: Path) -> tuple[dict, Path] | None:
    """Pick the most recent *real* artifact in ``model_dir``.

    Loads each ``*.json`` (skipping unreadable/invalid ones, so one bad file
    doesn't sink the whole manifest), drops smoke runs, and returns the
    ``(payload, path)`` with the latest embedded ``generated_at`` (mtime breaks
    ties / covers unparseable timestamps). Returns None if the directory has no
    eligible real run.
    """
    candidates: list[tuple[tuple[datetime, float], dict, Path]] = []
    for path in model_dir.glob("*.json"):
        try:
            payload = load_artifact(path)
        except Exception:
            # Skip unreadable / invalid artifacts; lidr re-validates on read.
            continue
        if _is_smoke_artifact(payload):
            continue
        ts = _parse_generated_at(payload.get("generated_at"))
        sort_key = (ts or datetime.min, path.stat().st_mtime)
        candidates.append((sort_key, payload, path))

    if not candidates:
        return None
    _, payload, path = max(candidates, key=lambda c: c[0])
    return payload, path


def _is_smoke_artifact(payload: dict) -> bool:
    """True if this artifact is a synthetic/offline smoke run, not a real backtest.

    Smoke runs use the synthetic data source (ticker ``FAKE``) and a config
    whose name carries ``synthetic`` (``dev_synthetic`` / ``news_dev_synthetic``).
    Either signal is enough — they coincide today, but matching both keeps the
    rule robust if one convention drifts.
    """
    config_name = str(payload.get("config_name", "")).lower()
    ticker = str(payload.get("ticker", "")).upper()
    return "synthetic" in config_name or ticker in _SMOKE_TICKERS


def _parse_generated_at(value: object) -> datetime | None:
    """Parse ``generated_at`` into a tz-naive UTC datetime for ordering.

    Accepts both the legacy ``YYYYMMDD-HHMMSS`` stamp and ISO 8601 — the two
    formats the schema's ``generated_at`` field allows. Returns None if neither
    parses, so the caller falls back to file mtime.
    """
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d-%H%M%S")
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _model_entry(payload: dict, latest: Path, predictions_root: Path) -> dict:
    """Build one manifest ``models[]`` entry from a chosen artifact payload."""
    cls_m = payload.get("metrics", {}).get("classification", {}) or {}
    strat_m = payload.get("metrics", {}).get("strategy", {}) or {}
    bench_m = payload.get("metrics", {}).get("benchmark", {}) or {}

    base_ll = cls_m.get("base_logloss")
    ll = cls_m.get("log_loss")
    skill = (
        1.0 - ll / base_ll
        if isinstance(base_ll, (int, float)) and base_ll
        and isinstance(ll, (int, float))
        else None
    )

    strat_cagr = strat_m.get("cagr")
    bench_cagr = bench_m.get("cagr")
    beats = (
        (strat_cagr > bench_cagr)
        if isinstance(strat_cagr, (int, float))
        and isinstance(bench_cagr, (int, float))
        else None
    )

    return {
        "model_id": payload["model_id"],
        "model_version": payload["model_version"],
        "latest_artifact": str(latest.relative_to(predictions_root.parent)).replace("\\", "/"),
        "oos_skill_score": skill,
        "beats_buy_and_hold": beats,
    }
