"""Tests for the manifest / leaderboard selection rule.

Pins the fix for the bug where ``build_manifest`` picked each model's headline
artifact by file mtime, so a freshly-run ``dev_synthetic`` smoke test could
headline the model (and did — the committed manifest pointed at a synthetic
run). The rule is now: drop smoke runs, then pick the latest *real* run by the
artifact's embedded ``generated_at`` timestamp.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from lidr_core.eval.leaderboard import build_manifest


def _write_artifact(
    model_dir: Path,
    *,
    config_name: str,
    ticker: str,
    generated_at: str,
    log_loss: float = 0.66,
    base_logloss: float = 0.67,
    strat_cagr: float = 0.10,
    bench_cagr: float = 0.14,
    mtime: float | None = None,
) -> Path:
    """Write a minimal but schema-valid artifact; optionally force its mtime."""
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "model_id": model_dir.name,
        "model_version": "0.1.0",
        "config_name": config_name,
        "ticker": ticker,
        "generated_at": generated_at,
        "metrics": {
            "classification": {"log_loss": log_loss, "base_logloss": base_logloss},
            "strategy": {"cagr": strat_cagr},
            "benchmark": {"cagr": bench_cagr},
        },
        "predictions": [
            {
                "date": "2020-01-02",
                "recommendation": "HOLD",
                "probability_up": 0.5,
                "y_pred": 1,
                "y_true": 1,
            }
        ],
    }
    path = model_dir / f"{config_name}-{generated_at}.json"
    path.write_text(json.dumps(payload))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _entry(manifest: dict, model_id: str) -> dict | None:
    return next((m for m in manifest["models"] if m["model_id"] == model_id), None)


def _predictions_root(tmp_path: Path) -> Path:
    """Mirror the production layout (``artifacts/predictions/``) so the
    ``latest_artifact`` path renders as ``predictions/<model>/<file>``."""
    return tmp_path / "artifacts" / "predictions"


def test_real_run_beats_more_recently_written_smoke_run(tmp_path: Path) -> None:
    """The core regression: a real run must headline even when a dev_synthetic
    smoke run is both newer by timestamp *and* written more recently (newer
    mtime) — the exact situation that produced the bad committed manifest."""
    root = _predictions_root(tmp_path)
    model_dir = root / "ta_ensemble"
    real = _write_artifact(
        model_dir,
        config_name="baseline_six_signals_unweighted",
        ticker="SPY",
        generated_at="20260527-190244",
        mtime=1000.0,  # older on disk
    )
    _write_artifact(
        model_dir,
        config_name="dev_synthetic",
        ticker="FAKE",
        generated_at="20260527-212727",  # newer timestamp
        mtime=2000.0,  # newer on disk — would win under the old mtime rule
    )

    manifest = build_manifest(root)
    entry = _entry(manifest, "ta_ensemble")
    assert entry is not None
    assert entry["latest_artifact"] == f"predictions/ta_ensemble/{real.name}"


def test_model_with_only_smoke_runs_is_omitted(tmp_path: Path) -> None:
    """A model whose only artifacts are synthetic smoke runs has no real result
    to advertise, so it must not appear in the manifest at all (news_sentiment
    today)."""
    root = _predictions_root(tmp_path)
    model_dir = root / "news_sentiment"
    _write_artifact(
        model_dir,
        config_name="news_dev_synthetic",
        ticker="FAKE",
        generated_at="20260528-211727",
    )

    manifest = build_manifest(root)
    assert _entry(manifest, "news_sentiment") is None
    assert manifest["models"] == []


def test_latest_real_run_chosen_by_timestamp_not_mtime(tmp_path: Path) -> None:
    """Among real runs, the later embedded timestamp wins even when the other
    file was touched more recently — so a checkout/copy that rewrites mtimes
    can't reorder the headline."""
    root = _predictions_root(tmp_path)
    model_dir = root / "ta_ensemble"
    newer = _write_artifact(
        model_dir,
        config_name="baseline_six_signals_unweighted",
        ticker="SPY",
        generated_at="20260527-190244",  # later timestamp
        mtime=1000.0,  # but older on disk
    )
    _write_artifact(
        model_dir,
        config_name="baseline_six_signals",
        ticker="SPY",
        generated_at="20260527-120203",  # earlier timestamp
        mtime=9999.0,  # touched most recently
    )

    manifest = build_manifest(root)
    entry = _entry(manifest, "ta_ensemble")
    assert entry is not None
    assert entry["latest_artifact"] == f"predictions/ta_ensemble/{newer.name}"


def test_entry_fields_populated_from_chosen_artifact(tmp_path: Path) -> None:
    """skill_score and beats_buy_and_hold are computed from the chosen real run."""
    root = _predictions_root(tmp_path)
    model_dir = root / "ta_ensemble"
    _write_artifact(
        model_dir,
        config_name="baseline_six_signals_unweighted",
        ticker="SPY",
        generated_at="20260527-190244",
        log_loss=0.66,
        base_logloss=0.67,
        strat_cagr=0.10,
        bench_cagr=0.14,
    )

    entry = _entry(build_manifest(root), "ta_ensemble")
    assert entry is not None
    assert entry["oos_skill_score"] == 1.0 - 0.66 / 0.67
    assert entry["beats_buy_and_hold"] is False  # 0.10 < 0.14
