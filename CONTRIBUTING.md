# Contributing

Thanks for looking. This is a personal-research project right now; PRs and issues are welcome, but expect opinionated review against the rules below.

## Development setup

```bash
# Create a venv and install everything (editable + dev tools)
python -m venv .venv
source .venv/bin/activate          # WSL/macOS/Linux; on Windows PowerShell: .venv\Scripts\Activate.ps1
make install                       # pip install -e ".[dev]"

# Confirm it works end-to-end without internet
make backtest CONFIG=configs/dev_synthetic.yaml
make test
make lint
```

Tested on Python 3.11 (CI). Python 3.10 should work; Python 3.14 works for everything except parquet-based caches (we use pickle on purpose — see `data/loaders.py`).

## What lands together

Each of the rules below has its own non-negotiable test or CI gate. If you can't meet them, open a draft PR and explain why.

### Adding a signal

A signal is a pure function `(prices: DataFrame, params: dict) -> Series` aligned to the input index, conforming to `src/lidr_ml/signals/base.py::SignalFn`. Three things must land in the **same PR**:

1. **The signal itself** — a new module under `src/lidr_ml/signals/` decorated with `@register("name")`, plus an import in `src/lidr_ml/signals/registry.py` so registration happens at startup.
2. **No-lookahead test** — add `(name, params)` to `SIGNAL_CASES` in `tests/test_no_lookahead.py`. The test compares `f(prices[:t])[t]` against `f(prices)[t]` at several points; they must match. Non-negotiable.
3. **Accuracy test** — add an `ACCURACY_CASES` entry in `tests/test_signal_accuracy.py` with an inline reference formula **and** at least two spot checks against hand-derived values on the arithmetic price series `close = [100, 101, ...]`. For RSI/MACD where smoothing is non-trivial, use `pandas-ta` as the reference.

### Adding a model

A model conforms to `src/lidr_ml/models/base.py::Model`: `fit(X, y) -> None`, `predict_proba(X) -> np.ndarray`, `predict(X) -> np.ndarray`. Steps:

1. Add the class under `src/lidr_ml/models/` — see `models/logistic.py` for the reference wrapper pattern (sklearn pipeline with `StandardScaler`).
2. Register it in `src/lidr_ml/models/__init__.py::MODEL_REGISTRY` as `"name": YourModelClass`.
3. Add or update a config in `configs/` that exercises it. Run a synthetic backtest end-to-end to confirm the pipeline wires up.

There's no model-accuracy test analog to the signal one — model behavior is judged by the cross-run `artifacts/results_log.csv` over real configs.

### Adding a config

Copy `configs/baseline.yaml`, change `name`, adjust `data.tickers` / `data.start_date` / `data.end_date`, swap signals or model as needed. See README → **Config schema** for the field reference.

## Backtest invariants

These are enforced by tests and code review:

- **Expanding-window walk-forward only.** Random k-fold leaks across time and is rejected on sight.
- **Transaction costs are modeled in every backtest.** Default 5 bps; configurable, but never zero for a config that's compared to buy-and-hold.
- **Every report shows a benchmark.** Strategy metrics rendered beside buy-and-hold; log loss rendered beside `base_logloss` (no-skill floor).
- **No lookahead.** See "Adding a signal" above.

## Tests and lint

```bash
make test          # python3 -m pytest
make lint          # ruff check src tests
make format        # ruff format + auto-fix
```

CI runs `make test` + `make lint` on Python 3.11 against every push and PR (see `.github/workflows/test.yml`). Don't merge red.

To run a single test:

```bash
python -m pytest tests/test_signal_accuracy.py::test_signal_matches_reference -k sma_crossover
```

## Commit + PR conventions

- Branch off `main`. PRs target `main`.
- Commit messages: short imperative title, then a blank line, then prose. Wrap at ~72 chars.
- Each commit should be self-contained and pass tests + lint.
- For meaningful changes, append a dated entry to `CLAUDE.md` → **Recent Changes** (one paragraph, what + why). The Maintenance Instructions at the bottom of CLAUDE.md spell out the rules.

## Where to start

If you're new to the codebase, read in this order:

1. `README.md` — what + how
2. `CLAUDE.md` → **Conventions** and **Key Decisions** — the rules and the reasoning
3. `src/lidr_ml/pipeline.py::run_pipeline` — every other module is invoked from here, top-to-bottom
4. `CLAUDE.md` → **Next Up** — what's worth working on
