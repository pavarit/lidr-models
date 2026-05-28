"""End-to-end smoke test for the news_sentiment dev config.

Runs the entire pipeline on synthetic prices + synthetic news. Passes if it
completes, writes a schema-v2 artifact that validates, and does **not**
pollute the tracked ``artifacts/results_log.csv`` (same guard as the
ta_ensemble smoke test).

Doesn't assert anything about model quality — the synthetic news stream is
i.i.d. noise by construction and metrics from it are not meaningful.
"""

from __future__ import annotations

from pathlib import Path

from lidr_core.contract.loader import load_artifact
from news_sentiment.pipeline import run_pipeline

# parents[1] = packages/news_sentiment/, where this package's configs/ live.
# parents[3] = the monorepo root, where artifacts/results_log.csv lives.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dev_runs_end_to_end() -> None:
    config = PACKAGE_ROOT / "configs" / "dev.yaml"

    log_path = REPO_ROOT / "artifacts" / "results_log.csv"
    size_before = log_path.stat().st_size if log_path.exists() else 0

    result = run_pipeline(config)

    assert result.report_path.exists(), "Report file should be written."
    assert result.report_path.stat().st_size > 1000, "Report should not be empty."

    # Predictions artifact must validate against the schema-v2 contract.
    assert result.predictions_path is not None, "dev.yaml asks for the predictions JSON."
    assert result.predictions_path.exists()
    payload = load_artifact(result.predictions_path)
    assert payload["schema_version"] == 2
    assert payload["model_id"] == "news_sentiment"
    assert payload["predictions"], "Should produce at least one OOS prediction."

    size_after = log_path.stat().st_size if log_path.exists() else 0
    assert size_after == size_before, (
        f"Smoke test must not append to {log_path} "
        f"(size {size_before} → {size_after}). "
        "Check that configs/dev.yaml still has `output.results_log: false`."
    )
