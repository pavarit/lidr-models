# Contributing

Thanks for looking. This is a personal-research project right now; PRs and issues are welcome, but expect opinionated review against the rules below.

## Development setup

```bash
# Create a venv and install everything (editable + dev tools)
python -m venv .venv
source .venv/bin/activate          # WSL/macOS/Linux; on Windows PowerShell: .venv\Scripts\Activate.ps1
make install                       # pip install -e ".[dev]"

# Confirm it works end-to-end without internet
make backtest CONFIG=packages/ta_ensemble/configs/dev_synthetic.yaml
make test
make lint
```

Tested on Python 3.11 (CI). Python 3.10 should work; Python 3.14 works for everything except parquet-based caches (we use pickle on purpose — see `packages/lidr_core/src/lidr_core/data/loaders.py`).

## What lands together

Each of the rules below has its own non-negotiable test or CI gate. If you can't meet them, open a draft PR and explain why.

### Adding a signal

A signal is a pure function `(prices: DataFrame, params: dict) -> Series` aligned to the input index, conforming to `lidr_core.protocols.signal.SignalFn`. Signals live in the owning model's package (today: `ta_ensemble`). Three things must land in the **same PR**:

1. **The signal itself** — a new module under `packages/ta_ensemble/src/ta_ensemble/signals/` decorated with `@register("name")`, plus an import in `packages/ta_ensemble/src/ta_ensemble/signals/registry.py` so registration happens at startup.
2. **No-lookahead test** — add `(name, params)` to `SIGNAL_CASES` in `packages/ta_ensemble/tests/test_no_lookahead.py`. The test compares `f(prices[:t])[t]` against `f(prices)[t]` at several points; they must match. Non-negotiable.
3. **Accuracy test** — add an `ACCURACY_CASES` entry in `packages/ta_ensemble/tests/test_signal_accuracy.py` with a structurally-different reference implementation **and** at least two spot checks against hand-derived values on a price fixture chosen so the expected numbers are derivable without a calculator.

### Adding a model

A model conforms to `lidr_core.protocols.model.Model`: `fit(X, y) -> None`, `predict_proba(X) -> np.ndarray`, `predict(X) -> np.ndarray`. Generic learners (reusable across model packages) live in `lidr_core`; model-specific wiring lives in the model's package. Steps for a generic learner:

1. Add the class under `packages/lidr_core/src/lidr_core/models/` — see `logistic.py` and `lightgbm.py` for the reference wrapper pattern.
2. Register it in `packages/lidr_core/src/lidr_core/models/__init__.py::MODEL_REGISTRY` as `"name": YourModelClass`.
3. Add or update a config in `packages/ta_ensemble/configs/` (or your model's own configs/) that exercises it. Run a synthetic backtest end-to-end to confirm the pipeline wires up.

There's no model-accuracy test analog to the signal one — model behavior is judged by the cross-run `artifacts/results_log.csv` over real configs.

### Adding a config

Copy a reference under `packages/ta_ensemble/configs/`, change `name`, adjust `data.tickers` / `data.start_date` / `data.end_date`, swap signals or model as needed. New configs should set `model_id` + `model_version` so produced artifacts identify their model family. See README → **Config schema** for the field reference.

### Changing report formatting

The README links to a committed sample report at `docs/sample-report/report.html` (rendered via `htmlpreview.github.io`) which is meant to show what a current run actually looks like. If your PR touches **`packages/lidr_core/src/lidr_core/eval/report.py`**, **`packages/lidr_core/src/lidr_core/eval/metrics.py`**, or **`packages/ta_ensemble/configs/baseline.yaml`** — anything that affects what the report displays or what numbers headline it — refresh the sample report in the same commit:

```bash
make refresh-sample-report   # runs SPY baseline + copies report.html to docs/sample-report/
```

Include both the updated `docs/sample-report/report.html` *and* the new `artifacts/results_log.csv` row in the commit. Keeps the link and the cited SPY baseline numbers in sync. Requires internet for the yfinance call.

## Backtest invariants

These are enforced by tests and code review:

- **Expanding-window walk-forward only.** Random k-fold leaks across time and is rejected on sight.
- **Transaction costs are modeled in every backtest.** Default 5 bps; configurable, but never zero for a config that's compared to buy-and-hold.
- **Every report shows a benchmark.** Strategy metrics rendered beside buy-and-hold; log loss rendered beside `base_logloss` (no-skill floor).
- **No lookahead.** See "Adding a signal" above.

## Tests and lint

```bash
make test          # python3 -m pytest
make lint          # ruff check packages
make format        # ruff format + auto-fix
```

CI runs `make test` + `make lint` on Python 3.11 against every push and PR (see `.github/workflows/test.yml`). Don't merge red.

To run a single test:

```bash
python -m pytest packages/ta_ensemble/tests/test_signal_accuracy.py::test_signal_matches_reference -k sma_crossover
```

## Workflow

`main` is protected — direct pushes are blocked, and every change has to go through a pull request with green CI. The flow:

```bash
# 1. Branch off the latest main
git checkout main && git pull
git checkout -b some-descriptive-name

# 2. Make your changes, commit locally
git add ...
git commit -m "..."

# 3. Push the branch
git push -u origin some-descriptive-name

# 4. Open a PR
gh pr create --fill   # uses your last commit message as the PR title/body
```

After the PR is open, **GitHub Actions CI** runs `make test + make lint` (see [`.github/workflows/test.yml`](.github/workflows/test.yml)). The status appears as a check on the PR. Must be green to merge.

When CI passes:

```bash
# 5. Merge with squash (one commit per PR on main)
gh pr merge --squash --delete-branch

# 6. Sync your local main
git checkout main && git pull
```

> Unlike the sibling `lidr` project, lidr-models does not auto-deploy anywhere — there is no Vercel preview to check. CI passing is the only required gate. The artifact `lidr-models` produces (the JSON predictions file) is consumed manually, so there is no continuous deployment to worry about.

> Emergencies: as the repo admin, you can override branch protection and push directly. Don't make a habit of it.

## Commit + PR conventions

- Commit messages: short imperative title, blank line, then prose. Wrap at ~72 chars.
- Each commit should be self-contained and pass tests + lint locally before pushing (`make test && make lint`).
- For meaningful changes, append a dated entry to `CLAUDE.md` → **Recent Changes** (one paragraph, what + why). The Maintenance Instructions at the bottom of CLAUDE.md spell out the rules.

## License of contributions

By opening a pull request, you agree that your contribution is licensed under the same [PolyForm Noncommercial 1.0.0](LICENSE) terms as the rest of the project. If you need different terms for a specific contribution, raise it in the PR description so we can discuss before merging.

## Where to start

If you're new to the codebase, read in this order:

1. `README.md` — what + how
2. `CLAUDE.md` → **Conventions** and **Key Decisions** — the rules and the reasoning
3. `packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline` — every other module is invoked from here, top-to-bottom
4. `CLAUDE.md` → **Next Up** — what's worth working on
