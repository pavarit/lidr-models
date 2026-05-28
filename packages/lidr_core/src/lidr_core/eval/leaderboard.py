"""Leaderboard / manifest writer.

Scans ``artifacts/predictions/<model_id>/*.json`` for produced artifacts, picks
the latest per model_id, and writes a ``manifest.json`` lidr can read to
discover the available models and their headline OOS score.

The manifest is also schema_version: 2; bump when adding fields that lidr's
reader treats as required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lidr_core.contract.loader import load_artifact


def build_manifest(predictions_root: Path) -> dict:
    """Return the manifest dict for every model_id under ``predictions_root``.

    ``predictions_root`` is conventionally ``artifacts/predictions/``. Each
    immediate subdirectory is treated as a model_id; within it the file with
    the most recent mtime is used as that model's latest artifact. mtime
    rather than filename order so a recent ``baseline_six_signals_*.json``
    isn't beaten by an older ``dev_synthetic_*.json`` whose config name sorts
    later alphabetically.
    """
    predictions_root = Path(predictions_root)
    models = []

    if predictions_root.exists():
        for model_dir in sorted(p for p in predictions_root.iterdir() if p.is_dir()):
            artifacts = list(model_dir.glob("*.json"))
            if not artifacts:
                continue
            latest = max(artifacts, key=lambda p: p.stat().st_mtime)
            try:
                payload = load_artifact(latest)
            except Exception:
                # Skip unreadable / invalid artifacts so a single bad file
                # doesn't prevent the manifest from being generated for the
                # rest. lidr's reader will validate again on consumption.
                continue

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

            models.append(
                {
                    "model_id": payload["model_id"],
                    "model_version": payload["model_version"],
                    "latest_artifact": str(latest.relative_to(predictions_root.parent)).replace("\\", "/"),
                    "oos_skill_score": skill,
                    "beats_buy_and_hold": beats,
                }
            )

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
