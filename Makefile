.PHONY: install backtest backtest-news test lint format clean clean-reports clean-predictions refresh-sample-report

# Default config if none specified.
CONFIG ?= packages/ta_ensemble/configs/dev_synthetic.yaml

# Override on the command line if needed: make backtest PYTHON=python
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ./packages/lidr_core
	$(PYTHON) -m pip install -e ./packages/ta_ensemble
	$(PYTHON) -m pip install -e ./packages/news_sentiment
	$(PYTHON) -m pip install -e ".[dev]"

backtest:
	$(PYTHON) -m ta_ensemble backtest $(CONFIG)

# news_sentiment uses a different CLI module; override CONFIG_NEWS if needed.
CONFIG_NEWS ?= packages/news_sentiment/configs/dev.yaml
backtest-news:
	$(PYTHON) -m news_sentiment backtest $(CONFIG_NEWS)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check packages

format:
	$(PYTHON) -m ruff format packages
	$(PYTHON) -m ruff check --fix packages

clean:
	rm -rf build dist *.egg-info packages/*/build packages/*/dist packages/*/*.egg-info
	rm -rf .pytest_cache .ruff_cache pytest-cache-files-*
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-reports:
	find reports -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Clears local prediction artifacts (all gitignored build outputs) + the
# regenerated manifest. Leaves the git-tracked artifacts/results_log.csv and
# the .gitkeep placeholders alone.
clean-predictions:
	find artifacts/predictions -mindepth 1 -not -name .gitkeep -delete
	rm -f artifacts/manifest.json

# Re-runs the SPY baseline and copies the freshest report HTML into
# docs/sample-report/ so README's "Example report" link stays in sync
# with the cited headline numbers. Requires internet (yfinance).
refresh-sample-report:
	$(PYTHON) -m ta_ensemble backtest packages/ta_ensemble/configs/baseline.yaml
	@latest=$$(ls -1td reports/baseline_v1-*/ | head -n 1) && \
		cp $$latest/report.html docs/sample-report/report.html && \
		echo "Refreshed docs/sample-report/report.html from $$latest"
