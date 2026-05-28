"""End-to-end smoke test using the offline synthetic config.

Runs the entire pipeline. Passes if it completes and writes a report.
Doesn't assert anything about model quality — that's not what this test is for.
"""

from __future__ import annotations

from pathlib import Path

from ta_ensemble.pipeline import run_pipeline

# parents[1] = packages/ta_ensemble/, where this package's configs/ live.
# parents[3] = the monorepo root, where artifacts/results_log.csv lives.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dev_synthetic_runs_end_to_end() -> None:
    config = PACKAGE_ROOT / "configs" / "dev_synthetic.yaml"

    # Guard the tracked artifacts/results_log.csv against pollution: the smoke
    # test runs on every push/PR and its synthetic rows have no analytical
    # value. dev_synthetic.yaml opts out via `output.results_log: false`; this
    # assertion makes sure that opt-out keeps working.
    log_path = REPO_ROOT / "artifacts" / "results_log.csv"
    size_before = log_path.stat().st_size if log_path.exists() else 0

    result = run_pipeline(config)

    assert result.report_path.exists(), "Report file should be written."
    assert result.report_path.stat().st_size > 1000, "Report should not be empty."
    if result.predictions_path is not None:
        assert result.predictions_path.exists()

    size_after = log_path.stat().st_size if log_path.exists() else 0
    assert size_after == size_before, (
        f"Smoke test must not append to {log_path} "
        f"(size {size_before} → {size_after}). "
        "Check that configs/dev_synthetic.yaml still has `output.results_log: false`."
    )
