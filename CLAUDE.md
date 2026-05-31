# CLAUDE.md

This file orients any AI assistant (Claude Code, Claude Cowork, etc.) joining this project. Read it before doing anything else. Then keep it current — see "Maintenance Instructions" at the bottom.

**Hosting repo**: https://github.com/pavarit/lidr-models

## Project Goal

`lidr-models` is the Python research monorepo that turns market data into **empirically calibrated BUY / HOLD / SELL recommendations** consumed by the sibling project [`lidr`](https://github.com/pavarit/lidr) (the Next.js front-end). lidr's confidence values today are heuristics (normalized strength scores). This repo's job is to replace them with probabilities learned from historical data — and to host *several competing* models, each scored on the same harness and benchmark, so the best-performing one feeds production.

Models are backtested with walk-forward validation from 2005 to today and emit a versioned JSON artifact (schema_version: 2) that lidr's `/api/signals/[ticker]` route can read directly. No running Python service is needed until lidr's FastAPI-service roadmap item is triggered.

## Architecture

Two-repo setup, deliberately kept separate:

- **`lidr`** (Next.js, `C:\Users\smnk1\Claude\Projects\lidr`) — the website. Computes signals live in TS for display, will eventually overlay calibrated probabilities from the artifacts this repo produces.
- **`lidr-models`** (this repo, Python) — a research monorepo that produces the artifacts. Three packages under `packages/`:
  - `lidr_core` — shared harness: backtest engine, eval/metrics, results_log + leaderboard, the JSON artifact contract (schema + writer + loader), the `SignalFn` / `Model` / `Feature` / `DataSource` protocols, base data loaders, generic learners (logistic / LightGBM). Owned once, reused by every model.
  - `ta_ensemble` — the six technical-analysis signals + their pipeline. Today's only complete model.
  - `news_sentiment` — Task 2, in development. PR-A merged (scaffolding + data adapters + lexicon scorer + features). PR-B ([PR #40](https://github.com/pavarit/lidr-models/pull/40)) in review: rewired to EODHD/Finnhub/Apewisdom, reddit/google_trends converted to permanent stubs, FinBERT + LLM hybrid scoring lit up. PR-C (after PR-B) runs the real `news_v0.yaml` backtest.

Integration is via JSON files written to `artifacts/predictions/<model_id>/<config>-<timestamp>.json`, validated against the contract on write, plus a top-level `artifacts/manifest.json` leaderboard lidr will use to discover models. A FastAPI service is on the lidr roadmap and will be added here when the lidr side is ready to consume live predictions.

## Stack

- **Python** ≥ 3.10
- **pandas / numpy** — data manipulation
- **scikit-learn** — base learners + walk-forward CV (`TimeSeriesSplit`)
- **LightGBM** — second base learner (added 2026-05-27, [PR #20](https://github.com/pavarit/lidr-models/pull/20))
- **yfinance** — free historical price data (with synthetic-data fallback for offline dev)
- **PyYAML** — config files
- **Typer** — CLI
- **jsonschema** — artifact contract validation on write/read
- **Matplotlib** — chart embedded in HTML reports (base64, no internet needed to view)
- **pytest** — tests
- **ruff** — lint + format
- **requests** — HTTP for news adapters; EDGAR, GDELT, Finnhub, EODHD all go through it
- **Optional extras** (`pip install -e ./packages/news_sentiment[scoring,llm]`): `transformers + torch` (FinBERT scoring), `anthropic` (LLM scorer). Reddit and Google Trends adapters are permanent stubs — no extras needed for them.

Workspace: each package under `packages/<name>/` has its own `pyproject.toml`; the root `pyproject.toml` carries only the dev-tool config (`ruff`, `pytest`) and a meta dev install. `make install` installs all three packages editable so cross-package imports resolve.

Planned but not yet added: **MLflow** (experiment tracking — deferred behind the CSV results log), **vectorbt** (faster backtest sweeps — revisit if the custom engine becomes the bottleneck).

## Commands

See [README.md → Quick start](README.md#quick-start) and [README.md → CLI](README.md#cli) for the canonical command reference. Targets: `make install`, `make backtest CONFIG=...` (ta_ensemble), `make backtest-news CONFIG_NEWS=...` (news_sentiment), `make test`, `make lint`, `make clean`, `make clean-reports`. Override the Python interpreter with `PYTHON=` if `python3` isn't right for your environment.

## Pipeline internals

Human-facing end-to-end walkthrough: [README → How the pipeline works](README.md#how-the-pipeline-works). The notes below are the AI-facing per-module reference; per-module detail also lives in each module's docstring, and the TA pipeline is wired together in `packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline`.

1. **Config** (`packages/ta_ensemble/configs/*.yaml`) — declares tickers, date range, data source, signals, model, backtest method, transaction costs. Carries `model_id` + `model_version` so produced artifacts identify their model family.
2. **Data loader** (`packages/lidr_core/src/lidr_core/data/loaders.py`) — pulls OHLCV from yfinance (`auto_adjust=True`, so close prices are total-return-adjusted). Cached by `(ticker, start, end)` in `data/raw/<ticker>_<start>_<end>.pkl`; the cache never expires — `rm data/raw/*.pkl` to force a refresh from yfinance.
3. **Signals** (`packages/ta_ensemble/src/ta_ensemble/signals/`) — pure functions: `DataFrame → Series`, aligned to the price index, lookahead-safe. Tested in `packages/ta_ensemble/tests/test_no_lookahead.py`.
4. **Target** (inline in `packages/ta_ensemble/src/ta_ensemble/pipeline.py::run_pipeline`) — binary: was the *N*-day forward return positive? Will become 3-class (BUY/HOLD/SELL) once we have a model with edge.
5. **Backtest engine** (`packages/lidr_core/src/lidr_core/backtest/engine.py`) — expanding-window walk-forward. Per split: fit on train, predict on test, discard the model. Never trains on post-test data.
6. **Model** (`packages/lidr_core/src/lidr_core/models/`) — pluggable: logistic regression and LightGBM. Neither beats no-skill on the current feature set (see Gotchas, Roadmap).
7. **Eval + report** (`packages/lidr_core/src/lidr_core/eval/`) — `classification_metrics` → `accuracy`, `base_rate`, `pred_rate`, `log_loss`, `base_logloss`, `n_obs`; `strategy_metrics` → `cagr`, `sharpe`, `max_drawdown`, `final_equity`; per-year breakdowns. `base_rate` and `pred_rate` are shown beside `accuracy` — without them, 0.55 accuracy looks fine until you notice the base rate is 0.61. Equity curve built in `engine.py::add_strategy_returns` (long when pred=1, cash otherwise). HTML report written to `reports/<config>-<timestamp>/`. The equity curve uses **1-day-forward returns**, not the N-day classification target — see Gotchas.

## Conventions

- **`main` is protected — all changes land via PR.** Direct pushes are blocked. Workflow: branch → commit → push → `gh pr create` → wait for green CI → `gh pr merge --squash --delete-branch`. Full procedure in [CONTRIBUTING.md → Workflow](CONTRIBUTING.md#workflow). No Vercel preview here — CI green is the only required gate.
- **Python ≥3.10**, src-layout packages under `packages/`; ruff (`line-length = 100`, `target-version = "py310"`). Snake_case modules and functions.
- **`lidr_core` is the harness; per-model packages depend on it. Never the other way around.** Shared logic (eval, contract, backtest engine, base learners, data loaders) lives in `lidr_core`. Model-specific signals/features/configs/pipelines live in the model's package.
- **Signals are pure functions**: `(prices: DataFrame, params: dict) → Series` aligned to the input index. Conform to `lidr_core.protocols.signal.SignalFn`. Must be lookahead-safe — `f(prices[:t])[t] == f(prices)[t]`.
- **Models conform to `lidr_core.protocols.model.Model`**: `fit(X, y) -> None`, `predict_proba(X) -> np.ndarray` (shape `(n_samples, n_classes)`), `predict(X) -> np.ndarray`.
- **Artifacts are validated on write.** Every `predictions/<model_id>/*.json` must validate against `lidr_core/contract/schema/artifact.schema.json`; `write_artifact` enforces this. Breaking schema changes bump `schema_version`; additive optional fields don't. New models must populate `model_id` + `model_version` in config.
- **When adding a signal to `ta_ensemble`, three things must land in the same PR:**
  1. Register in `ta_ensemble/signals/registry.py`.
  2. Add to `SIGNAL_CASES` in `packages/ta_ensemble/tests/test_no_lookahead.py` (non-negotiable).
  3. Add an `ACCURACY_CASES` entry in `packages/ta_ensemble/tests/test_signal_accuracy.py` — a 5-tuple `(name, params, reference_fn, prices_factory, spot_checks)`. Use a structurally different reference impl (e.g., loop-based numpy vs vectorized pandas) so a shared bug between signal and reference is unlikely. ≥2 spot checks required; tolerance `rtol=1e-8`.
- **Outcome-changing PRs need verification evidence** — embedded chart + dated sanity-check table from real SPY data (plus a max-abs-difference parity number for signal ports). See the **`verify-evidence` skill** (`.claude/skills/verify-evidence/`) for the full procedure, guardrails, and runnable scaffold. Refactors, doc-only, and infra/CI PRs don't need this.
- **Transcribe figures, never recall them.** Any number or identifier committed to text (counts, SHAs, token/$ spend, parity numbers) must be copied from same-turn tool output — never recalled from memory. A "PASS"/"verified" status requires an inspectable artifact to already exist (else mark *pending*). If a wrong number slips, correct it in read-only mode: quote every fix figure from a file or command in the same message.
- **Backtests use expanding-window walk-forward only.** Random k-fold leaks information across time and is rejected on sight.
- **Transaction costs are modeled in every backtest.** Default 5 bps; configurable but never zero in a config compared to buy-and-hold.
- **Every report must show a benchmark.** Strategy metrics beside buy-and-hold; log loss beside `base_logloss` (the no-skill floor). New comparison columns appended to the right.
- **Errors at config boundaries are `NotImplementedError`** with a message naming the unsupported value. Internal invariants use `assert` or raise `ValueError`. No silent fallbacks.
- **CI runs `make test` + `make lint` on every push** (`.github/workflows/test.yml`). Don't merge red.
- **Refresh the committed sample report when report formatting changes.** If a PR touches `packages/lidr_core/src/lidr_core/eval/report.py`, `packages/lidr_core/src/lidr_core/eval/metrics.py`, or `packages/ta_ensemble/configs/baseline.yaml`, run `make refresh-sample-report` and include the updated `docs/sample-report/report.html` and new `artifacts/results_log.csv` row in the same commit. Requires internet (yfinance).

## Key Decisions

Strategic forks in the road — *why we chose X over Y*. For procedural rules see Conventions; for things that bit us see Gotchas.

- **One Python project, separate from lidr.** Python venvs + Node modules in one tree gets messy; the two pieces deploy on different cadences (Vercel vs. eventually Railway/Render). The deliberate seam is the JSON artifact.
- **Source of truth for signal logic stays in lidr until cutover.** Each Python signal in `packages/ta_ensemble/signals/` is a *port* of the TS signal in lidr's `lib/signals/`; numerical parity is tested so the two implementations don't drift.
- **Synthetic data fallback is part of the contract.** The `dev_synthetic` config lets the pipeline run with no network — makes CI, tests, and Cowork-sandbox verification trivial. The existence of an offline-runnable config is load-bearing for the test suite.
- **Outputs to lidr are calibrated probabilities, not heuristics.** Today lidr uses normalized strength scores. This project replaces them with `predict_proba` calibrated via Platt or isotonic regression. Current probabilities are *not* calibrated — calibration is required before any artifact goes live.
- **The backtest does NOT produce a deployable model.** Walk-forward fits a fresh model per split and discards it — the deliverable is the stitched OOS prediction series. Nothing is written to `artifacts/models/`. Fitting + serializing a deployable model is edge-gated.

## Gotchas

Non-obvious things that bit us. Each entry earned its place by causing a real problem.

- **The equity curve runs on 1-day-forward returns, not the N-day classification target.** Compounding the N-day return on every daily row counts the same window ~N times and inflates final equity by an order of magnitude. See `pipeline.py::run_pipeline` (`daily_fwd_return` is the equity input; `fwd_clean` is the classifier target only) and `packages/lidr_core/tests/test_strategy_returns.py`, which ensures the bug stays fixed.
- **Python 3.14 has no parquet wheels yet.** `lidr_core/data/loaders.py` caches OHLCV in pickle, not parquet. Don't switch to parquet without verifying `pyarrow`/`fastparquet` wheels exist for the target Python — the comment at `loaders.py:50` documents why.
- **Typer collapses single-command apps into a flat CLI.** Removing `list-signals` from `ta_ensemble/cli.py` would silently break `python -m ta_ensemble backtest <config>`. Don't remove `list-signals` until a third real command lands.
- **`yfinance` only has currently-listed tickers** — survivorship bias. Fine for SPY/QQQ/sector ETFs; suspicious for individual-name backtests. Don't trust individual-stock results without a CRSP-style source.
- **`class_weight="balanced"` is actively harmful on this 60/40 problem, not just "distorts probabilities."** Removing it shrank the distance from no-skill ~7× (`skill_score` -0.0374 → -0.0051) — it was forcing confidently-wrong predictions. Default new configs to `class_weight=None`; revisit only if a future model shows genuine class-balance trouble.
- **`results_log.csv` rows from before 2026-05-27 are slightly off.** The backtester had an inclusive right endpoint on the test slice, so the boundary date between split N and N+1 was predicted twice and double-compounded in the equity curve (~0.3% of rows). Fixed; pre-fix rows weren't re-run. Treat cross-row comparisons straddling 2026-05-27 with this in mind.

## Active Task

Task 2 (`news_sentiment`) sequence:

- **Step 1 — Horizon spike** (done 2026-05-28): Swept `horizon_days ∈ {5,10,20,60}` × 3 model classes. `skill_score` is negative at every horizon and degrades monotonically as horizon lengthens — h5 is least bad in all three model classes. The TA feature set has no directional edge at any horizon. **Keep `horizon_days: 5` for `news_v0.yaml`** (short horizon also matches news's short-lived impact). This direction is closed; full detail in [changelog](docs/changelog.md#2026-05-28--horizon-spike-longer-target-horizons-make-the-ta-model-worse-not-better).
- **Step 2 — Revised PR-B** (in review — [PR #40](https://github.com/pavarit/lidr-models/pull/40)): Rewired data sources (EODHD $19.99/mo, Apewisdom free, Finnhub free backbone); reddit/google_trends converted to permanent stubs; FinBERT + LLM hybrid scoring lit up. Live verification passed 2026-05-30. Full spec: [docs/plans/task-2-news-sentiment-model.md](docs/plans/task-2-news-sentiment-model.md).
- **Step 3 — PR-C** (after PR-B merges): Real `news_v0.yaml` backtest on 5–10 ticker universe; `results_log` row + refreshed manifest + comparison chart + per-period table per the evidence convention. Deletes the plan doc in the cleanup commit.

## Roadmap

No model beats buy-and-hold yet; the bottleneck is features/target, not model class. Full priority list: [docs/roadmap.md](docs/roadmap.md).

**Active directions (before the edge gate):**
1. **Reformulate target/features** — two live directions: (b) magnitude-regression target instead of binary sign; (c) regime features (VIX, yield curve, 60-day realized vol). Direction (a) horizon sweep closed 2026-05-28.
2. **Stacking** — parked until a base learner shows non-zero skill.

*Items past the edge gate (final-model fit/serialize, calibration, 3-class output, lidr wiring, MLflow) stay parked until something beats buy-and-hold.*

## Recent Changes

Full history and earlier entries: [docs/changelog.md](docs/changelog.md).

### 2026-05-29 — Anti-fabrication rule: transcribe figures, never recall them

No number gets committed to text unless copied from same-turn tool output. "PASS"/"verified" status requires a linked, openable artifact (else mark *pending*). Corrections also use read-only mode — quote every fix figure from a command or file in the same message. Added as Conventions bullet; `verify-evidence` skill cross-referenced.

### 2026-05-29 — Revised PR-B: news_sentiment data sources rewired + scoring lit up ([PR #40](https://github.com/pavarit/lidr-models/pull/40))

Deleted `tiingo.py`; reddit/google_trends converted to permanent `NotImplementedError` stubs; added `finnhub.py`, `apewisdom.py`, `eodhd.py`; FinBERT + LLM hybrid scoring lit up. 74 tests, all green. Live verification (EODHD timestamp spot-check + LLM smoke) both PASS 2026-05-30.

### 2026-05-29 — verify-evidence skill created ([PR #39](https://github.com/pavarit/lidr-models/pull/39))

Merged the former Diagnostic Playbook (Stage 1: pressure-test conclusions) and PR-evidence procedure (Stage 2: produce chart + sanity table) into one on-demand skill. Surfaces itself when a performance claim or outcome-changing PR is detected. CLAUDE.md's Diagnostic Playbook section removed; Conventions bullet points at the skill.

### 2026-05-29 — Leaderboard fix: dev_synthetic runs no longer headline manifest.json ([PR #38](https://github.com/pavarit/lidr-models/pull/38))

`build_manifest` now excludes smoke runs and selects the headline artifact by embedded `generated_at`, not mtime. Models with only smoke runs are omitted entirely from the manifest. 44→48 tests.

### 2026-05-29 — Documentation-cleanup arc (4 PRs, [#34](https://github.com/pavarit/lidr-models/pull/34)–[#37](https://github.com/pavarit/lidr-models/pull/37))

Fixed stale facts/dead links; single-sourced status to README; added `docs/README.md` + configs README; added `make clean-predictions` + artifact hygiene docs. No production code touched.

## Maintenance Instructions

If you make meaningful changes, also update this file in the same session.

- **Keep evergreen sections current.** Project Goal, Architecture, Stack, Pipeline internals, Conventions, Key Decisions, Gotchas, Active Task, and the Roadmap summary should reflect reality. If your work invalidates a fact, update it before ending the session.
- **Each fact lives in one section.** Conventions = the rule. Key Decisions = why we chose X over Y. Gotchas = what bit us. Cross-reference instead of duplicating.
- **Durable model-PR diagnostics and reporting lessons go in the `verify-evidence` skill, not a dated entry.** Dated entries fold into `docs/changelog.md`; the skill surfaces itself when the task arises.
- **Revise a Gotcha in place when an experiment contradicts it.** Don't append a caveat elsewhere — rewrite the Gotcha with concrete numbers.
- **Append a dated entry to Recent Changes for each session that produces real changes.** Format: `### YYYY-MM-DD — short title` + 2–4 sentences covering what was done and why. Add the full detail to `docs/changelog.md`.
- **When Recent Changes exceeds 7 entries, move the oldest entries to `docs/changelog.md`**, keeping only the most recent 5 in CLAUDE.md.
- **When a Roadmap item ships, remove it from `docs/roadmap.md`** and document the completion in `docs/changelog.md`. Use names not numbers for cross-references so renumbering doesn't break them.
- **Cross-link to lidr.** If a change affects the integration (artifact format, signal parity, what the website consumes), update lidr's CLAUDE.md in the same session.
- **Session-wrap behavior** — see the "Session-wrap behavior" section in the global `~/.claude/CLAUDE.md`.
