.PHONY: install backtest test lint format clean clean-reports

# Default config if none specified
CONFIG ?= configs/dev_synthetic.yaml

install:
	pip install -e ".[dev]"

backtest:
	python -m lidr_ml backtest $(CONFIG)

test:
	python3 -m pytest

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache pytest-cache-files-*
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-reports:
	find reports -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
