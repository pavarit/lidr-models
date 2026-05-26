.PHONY: install backtest test lint format clean clean-reports

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
