.PHONY: install backtest test lint format clean clean-reports refresh-sample-report

# Default config if none specified
CONFIG ?= configs/dev_synthetic.yaml

# Override on the command line if needed: make backtest PYTHON=python
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

backtest:
	$(PYTHON) -m lidr_ml backtest $(CONFIG)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache pytest-cache-files-*
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-reports:
	find reports -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Re-runs the SPY baseline and copies the freshest report HTML into
# docs/sample-report/ so README's "Example report" link stays in sync
# with the cited headline numbers. Requires internet (yfinance).
refresh-sample-report:
	$(PYTHON) -m lidr_ml backtest configs/baseline.yaml
	@latest=$$(ls -1td reports/baseline_v1-*/ | head -n 1) && \
		cp $$latest/report.html docs/sample-report/report.html && \
		echo "Refreshed docs/sample-report/report.html from $$latest"
