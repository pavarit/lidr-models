"""End-to-end smoke test using the offline synthetic config.

Runs the entire pipeline. Passes if it completes and writes a report.
Doesn't assert anything about model quality — that's not what this test is for.
"""

from __future__ import annotations

from pathlib import Path

from lidr_ml.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dev_synthetic_runs_end_to_end() -> None:
    config = REPO_ROOT / "configs" / "dev_synthetic.yaml"
    result = run_pipeline(config)
    assert result.report_path.exists(), "Report file should be written."
    assert result.report_path.stat().st_size > 1000, "Report should not be empty."
    if result.predictions_path is not None:
        assert result.predictions_path.exists()
