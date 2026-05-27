# CLAUDE.md

This file orients any AI assistant (Claude Code, Claude Cowork, etc.) joining this project. Read it before doing anything else. Then keep it current — see "Maintenance Instructions" at the bottom.

**Hosting repo**: https://github.com/pavarit/lidr-ml

## Project Goal

`lidr-ml` is the Python ML/backtesting pipeline that turns the technical signals from the sibling project [`lidr`](https://github.com/pavarit/lidr) into **empirically calibrated BUY / HOLD / SELL recommendations**. lidr's confidence values today are heuristics (normalized strength scores). This project's job is to replace them with probabilities learned from historical data — by training an ensemble model over the signals, backtested with walk-forward validation from 2005 to today.

The pipeline emits a versioned JSON artifact that lidr's `/api/signals/[ticker]` route can read directly. No running Python service is needed until lidr's FastAPI-service roadmap item is triggered.

## Architecture

Two-project setup, deliberately kept separate:

- **`lidr`** (Next.js, `C:\Users\smnk1\Claude\Projects\lidr`) — the website. Computes signals live in TS for display, will eventually overlay calibrated probabilities from the artifact this project produces.
- **`lidr-ml`** (this project, Python) — produces the model and the artifact. Iterates on its own cadence with its own tooling.

Integration is via a JSON file (`artifacts/predictions/<config>-<timestamp>.json`), not an HTTP service. Cheap, debuggable, version-controllable. A FastAPI service is on the lidr roadmap and will be added here when the lidr side is ready to consume live predictions.

## Stack

- **Python** ≥ 3.10
- **pandas / numpy** — data manipulation
- **scikit-learn** — base learners + walk-forward CV (`TimeSeriesSplit`)
- **yfinance** — free historical price data (with synthetic-data fallback for offline dev)
- **PyYAML** — config files
- **Typer** — CLI
- **Matplotlib** — chart embedded in HTML reports (base64, no internet needed to view)
- **pytest** — tests
- **ruff** — lint + format

Planned but not yet added: **LightGBM** (second base learner — Next Up #4), **MLflow** (experiment tracking — deferred to Next Up #10 behind the CSV results log), **vectorbt** (faster backtest sweeps — not yet on the roadmap; revisit if the custom engine becomes the bottleneck).

## Commands

See [README.md → Quick start](README.md#quick-start) and [README.md → CLI](README.md#cli) for the canonical command reference. Targets: `make install`, `make backtest CONFIG=...`, `make test`, `make lint`, `make clean`, `make clean-reports`. Override the Python interpreter with `PYTHON=` if `python3` isn't right for your environment.

## How the pipeline works

Reading top-down:

1. **Config** (`configs/*.yaml`) — declares what to run: tickers, date range, data source, which signals, which model, backtest method, transaction costs.
2. **Data loader** (`src/lidr_ml/data/loaders.py`) — pulls OHLCV from yfinance (or generates synthetic series for offline dev). yfinance is called with `auto_adjust=True`, so close prices are total-return-adjusted (dividends + splits folded in); a "200-day high" is a 200-day high of the adjusted series, not the raw close. Cached by `(ticker, start, end)` in `data/raw/<ticker>_<start>_<end>.pkl`; the cache never expires, so `rm data/raw/*.pkl` to force a refresh from yfinance.
3. **Signals** (`src/lidr_ml/signals/`) — each signal is a pure function that takes a DataFrame of prices and returns a Series of feature values aligned to the price index. Every signal must be **lookahead-safe** (only uses data up to time *t* to compute the value at time *t*). Tested in `tests/test_no_lookahead.py`.
4. **Target** (computed inline in `src/lidr_ml/pipeline.py::run_pipeline`) — for now, binary: was the *N*-day forward return positive? Will become 3-class (BUY/HOLD/SELL) once we have a useful model.
5. **Backtest engine** (`src/lidr_ml/backtest/engine.py`) — expanding-window walk-forward. For each split: fit model on train slice, predict on test slice, store predictions. Never trains on data after the test period.
6. **Model** (`src/lidr_ml/models/`) — pluggable. Today: logistic regression. Next: LightGBM, then a stacking meta-learner over both.
7. **Eval + report** (`src/lidr_ml/eval/`) — classification metrics (`classification_metrics` → `accuracy`, `base_rate`, `pred_rate`, `log_loss`, `base_logloss`, `n_obs`), strategy metrics (`strategy_metrics` → `cagr`, `sharpe`, `max_drawdown`, `final_equity`), and per-year breakdowns (`by_year` reports the classification fields per calendar year; `performance_by_year` reports strategy vs buy-and-hold returns and the excess). `base_rate` and `pred_rate` are deliberately shown beside `accuracy` — without them, an accuracy of 0.55 looks fine until you notice the base rate is 0.61. The equity curve itself is built in `backtest/engine.py::add_strategy_returns` ("go long when prediction = 1, cash otherwise"). HTML report written to `reports/<config>-<timestamp>/`. The report shows a **Strategy vs Buy & Hold** comparison table (CAGR, Sharpe, max drawdown, final equity for both legs) and a **per-year performance** table (strategy vs buy-hold return + excess) so regime-dependence is visible. A **Summary** section at the top translates the config into English and states the out-of-sample span (derived from the predictions, not the raw data range). Both the top classification metrics and the per-year table show **base_logloss** (the no-skill floor = entropy of the base rate) beside log loss, and the comparison/per-year tables green/red-highlight whichever side wins. The equity curve is marked to market with **1-day-forward returns**, not the N-day classification target — see Gotchas.
8. **Cross-run results log** (`src/lidr_ml/eval/results_log.py`) — `append_run()` appends one row per backtest to `artifacts/results_log.csv` (skill score, full-OOS + per-period log loss with base-rate floors, strategy vs benchmark CAGR/Sharpe/max-drawdown, excess) so "did this change help?" is answerable without opening every HTML report. Opt out per-config with `output.results_log: false`; defaults to true. `configs/dev_synthetic.yaml` sets it false so the smoke test (run on every push/PR) doesn't pollute the tracked CSV with synthetic rows — `tests/test_pipeline_smoke.py` asserts the file size doesn't grow.

Everything is wired together by `src/lidr_ml/pipeline.py::run_pipeline(config_path)`.

## Folder map

```
configs/                  experiment configs (YAML, one per run)
  baseline.yaml           SPY, 2005–today, SMA crossover + logistic regression
  dev_synthetic.yaml      same as baseline but synthetic data — offline-safe smoke test
data/
  raw/                    cached OHLCV pulled from yfinance
src/lidr_ml/
  __init__.py
  __main__.py             entry for `python -m lidr_ml`
  cli.py                  Typer CLI
  pipeline.py             end-to-end orchestrator
  data/
    loaders.py            yfinance + synthetic loader
  signals/
    base.py               Signal protocol
    registry.py           name → signal-callable lookup
    sma_crossover.py      ported from lidr's lib/signals/sma.ts
  models/
    base.py               Model protocol
    logistic.py           sklearn logistic regression wrapper
  backtest/
    engine.py             expanding-window walk-forward backtester + strategy returns
  eval/
    metrics.py            classification_metrics (accuracy / base_rate / pred_rate / log_loss / base_logloss),
                          strategy_metrics (cagr / sharpe / max_drawdown / final_equity),
                          by_year + performance_by_year breakdowns
    report.py             HTML report generator (base64-embedded chart)
    results_log.py        appends one row per run to artifacts/results_log.csv
reports/                  generated HTML reports (gitignored except .gitkeep)
artifacts/
  models/                 (planned) final trained model — NOT yet written; backtest models are throwaway
  predictions/            JSON predictions consumed by lidr
  results_log.csv         cross-run results log (one row per backtest, tracked in git)
tests/
  conftest.py             shared `synthetic_prices` fixture
  test_no_lookahead.py    asserts every registered signal is lookahead-safe
  test_signal_accuracy.py element-wise signal correctness + spot checks on arithmetic series
  test_strategy_returns.py guards the equity-curve return rule (1-day-forward, costs charged)
  test_pipeline_smoke.py  runs dev_synthetic config end-to-end
.github/workflows/
  test.yml                CI: `make test` + `make lint` on every push / PR
```

## Conventions (read before writing code)

- **`main` is protected — all changes land via PR.** Direct pushes are blocked. Workflow: branch → commit → push → `gh pr create` → wait for green CI → `gh pr merge --squash --delete-branch`. Full procedure in [CONTRIBUTING.md → Workflow](CONTRIBUTING.md#workflow). Unlike the sibling lidr project there is no Vercel preview here — CI passing is the only required gate, because lidr-ml doesn't auto-deploy anywhere.
- **Python ≥3.10**, src-layout package; ruff for lint + format (`line-length = 100`, `target-version = "py310"`). Snake_case modules and functions.
- **Signals are pure functions**: `(prices: DataFrame, params: dict) → Series` aligned to the input index. Conform to `signals/base.py::SignalFn`. Must be lookahead-safe — `f(prices[:t])[t] == f(prices)[t]`.
- **Models conform to `models/base.py::Model`**: three methods — `fit(X, y) -> None`, `predict_proba(X) -> np.ndarray` (shape `(n_samples, n_classes)`), and `predict(X) -> np.ndarray`.
- **When adding a signal, three things must land in the same PR**:
  1. Register in `signals/registry.py`.
  2. Add to the `SIGNAL_CASES` table in `tests/test_no_lookahead.py` (the lookahead test is non-negotiable).
  3. Add an `ACCURACY_CASES` entry in `tests/test_signal_accuracy.py` (inline reference formula + ≥2 spot checks against hand-derived values).
- **Backtests use expanding-window walk-forward only.** Random k-fold leaks information across time and is rejected on sight.
- **Transaction costs are modeled in every backtest.** Default 5 bps; configurable but never zero in a config that gets compared to buy-and-hold.
- **Every report must show a benchmark.** Strategy metrics rendered beside buy-and-hold; log loss rendered beside `base_logloss` (the no-skill floor). New comparison columns get appended to the *right* of existing benchmark columns.
- **Errors at config boundaries are `NotImplementedError`** with a message naming the unsupported value (see `pipeline.py` for the pattern). Internal invariants use `assert` or raise `ValueError`. No silent fallbacks.
- **CI runs `make test` + `make lint` on every push** (`.github/workflows/test.yml`). Don't merge red.
- **Refresh the committed sample report when report formatting changes.** If a PR touches `src/lidr_ml/eval/report.py`, `src/lidr_ml/eval/metrics.py`, or `configs/baseline.yaml`, run `make refresh-sample-report` and include the updated `docs/sample-report/report.html` (and the new `artifacts/results_log.csv` row that the run appends) in the same commit. Keeps the README's "Example report" link in sync with the headline SPY baseline numbers. Requires internet (yfinance).

See **Gotchas** below for the past failures that motivated several of these rules.

## Key Decisions

Strategic forks in the road — *why we chose X over Y*. For procedural rules see Conventions; for things that bit us see Gotchas.

- **One Python project, separate from lidr.** Considered as a folder inside lidr; rejected because Python venvs + Node modules in one tree gets messy fast and the two pieces deploy on different cadences (Vercel vs. eventually Railway/Render). The deliberate seam is the JSON artifact.
- **Source of truth for signal logic stays in lidr until cutover.** Each Python signal in `src/lidr_ml/signals/` is a *port* of the corresponding TS signal in lidr's `lib/signals/`; numerical parity is tested so the two implementations don't drift. When the calibrated model goes live, we'll decide whether the TS versions get deleted, kept as fallback, or only used for display.
- **Synthetic data fallback is part of the contract.** The `dev_synthetic` config lets the pipeline run with no network — makes CI, tests, and Cowork-sandbox verification trivial. Not just a dev convenience; the existence of an offline-runnable config is load-bearing for the test suite.
- **Outputs to lidr are calibrated probabilities, not heuristics.** Today lidr's confidence values are normalized strength scores (heuristics). The whole point of this project is to replace them with `predict_proba` calibrated via Platt or isotonic regression. (The current baseline's probabilities are *not* calibrated — see Gotchas re: `class_weight="balanced"` — so the calibration step is required before any artifact goes live.)
- **The backtest does NOT produce a deployable model.** Expanding-window walk-forward fits a fresh model per split, predicts that split's test slice, and discards it — the deliverable is the stitched out-of-sample prediction series, used only for evaluation. There is no single trained model and nothing is written to `artifacts/models/`. Fitting + serializing a deployable model is a separate not-yet-built step (Next Up #7).

## Gotchas

Non-obvious things that bit us. Each entry earned its place by causing a real problem.

- **The equity curve runs on 1-day-forward returns, not the N-day classification target.** Compounding the N-day return on every daily row counts the same window ~N times and inflates final equity by an order of magnitude. See `pipeline.py::run_pipeline` (`daily_fwd_return` is the equity input; `fwd_clean` is the classifier target only) and `tests/test_strategy_returns.py`, which is what makes sure the bug stays fixed.
- **Python 3.14 has no parquet wheels yet.** `data/loaders.py` caches OHLCV in pickle, not parquet. Don't "fix" the loader by switching back to parquet without verifying `pyarrow`/`fastparquet` wheels exist for the target Python — the comment at `loaders.py:50` documents why.
- **Typer collapses single-command apps into a flat CLI.** Removing `list-signals` from `cli.py` would silently break `python -m lidr_ml backtest <config>` (Typer would re-flatten the entry point). Don't remove `list-signals` until a third real command lands.
- **`yfinance` only has currently-listed tickers** — survivorship bias. Fine for SPY/QQQ/sector ETFs; suspicious for individual-name backtests. Don't trust individual-stock results without a CRSP-style source.
- **`class_weight="balanced"` in the baseline distorts `predict_proba` away from true frequencies.** Fine for the up/down decision the backtest evaluates; not OK to ship to lidr as a calibrated probability. Revisit `class_weight` and add an explicit Platt/isotonic calibration step when wiring the prediction artifact (Next Up #7).

## Next Up

Priority order, reset on 2026-05-21 around a single near-term goal: **prove the model has an edge over buy-and-hold** before building any serving/integration plumbing. The SPY baseline currently shows no edge (see Active Task), so the path below is instrument → add features → improve the model. Everything past #6 is explicitly gated on an edge actually appearing.

1. ~~**Add a lightweight cross-run results log.**~~ ✅ Done 2026-05-22. `artifacts/results_log.csv` — appended after every run; columns include skill score, full-OOS log loss, per-period log loss for 2025 and Q1 2026 (with base_logloss floors), strategy vs benchmark CAGR/Sharpe/drawdown, and excess. See `src/lidr_ml/eval/results_log.py`.
2. ~~**Build the signal accuracy test harness.**~~ ✅ Done 2026-05-22. `tests/test_signal_accuracy.py` — two-layer validation: (1) element-wise comparison against an inline reference formula for each signal; (2) spot checks against hand-derived values on an arithmetic price series. No cross-repo coordination needed. GitHub Actions CI added (`.github/workflows/test.yml`). `tests/conftest.py` introduced to share the `synthetic_prices` fixture. Pre-existing lint errors in `src/` cleaned up; `Makefile` updated to use `python3 -m pytest` for correct interpreter resolution.
3. **Port the remaining five lidr signals to Python**: `rsi`, `macd`, `bollinger`, `breakout`, `volume`. Each registered, added to `tests/test_no_lookahead.py`, and given an `ACCURACY_CASES` entry in `tests/test_signal_accuracy.py` (inline reference formula + ≥2 spot checks; use `pandas-ta` as reference for RSI/MACD where the smoothing algorithm is non-trivial). *Biggest single lever on edge — takes the model from one feature to six. Re-run the logistic baseline on all six as a cheap checkpoint: if six features still can't beat the benchmark, the bottleneck is the model, not the features.*
4. **Add LightGBM as a second base learner.** Drop-in: implement `models/lightgbm.py` against the same `Model` protocol, add a config that uses it. *A nonlinear learner over six signals is the most likely place an edge first appears.*
5. **Introduce stacking.** Once two base learners exist, add a `StackedModel` whose `fit` trains the base learners via out-of-fold predictions and then trains a meta-learner (logistic regression) on top.
6. **Add regime features.** VIX level, yield-curve slope, 60-day realized vol. Fed into the meta-learner so it can lean on different base models in different environments.

— Edge gate: items below stay parked until something above beats buy-and-hold. —

7. **Add a final-model fit + serialize step.** The backtest only evaluates (throwaway per-split models); nothing fits a model on all data or writes `artifacts/models/`. Before serving live predictions, fit one model on all available history, serialize it, and write the prediction artifact. Trigger: once a model beats buy-and-hold.
8. **Calibrate `predict_proba` via Platt or isotonic regression.** The baseline's `class_weight="balanced"` distorts probabilities away from true frequencies (see Gotchas) — fine for the up/down decision the backtest evaluates, not OK to ship to lidr as a calibrated probability. Switch to `class_weight=None` plus a calibration wrapper before any prediction artifact is consumed by lidr.
9. **Migrate the output from binary to 3-class (BUY / HOLD / SELL).** Today's target is binary (`fwd_return > 0`). The project's stated deliverable is 3-class. Two paths: (a) post-hoc bucketing of the calibrated probability into BUY / HOLD / SELL bands, (b) reformulate the target itself (e.g., three quantile bins of `fwd_return`). Decision punt until a binary model with edge exists; the 3-class formulation changes how "beats buy-and-hold" is measured.
10. **Evolve the artifact JSON schema as lidr's needs firm up.** Schema is already implemented at `schema_version: 1` (see `pipeline.py:174-189`: `config_name`, `ticker`, `generated_at`, `metrics.{classification,strategy,benchmark}`, `predictions[].{date,y_true,y_pred,probability_up}`). Bump the version when lidr's `/api/signals/[ticker]` is ready to consume it and any fields need to change (per-signal contributions, BUY/HOLD/SELL class labels from #9, etc.).
11. **Wire lidr's `/api/signals/[ticker]` to read the artifact.** That's the bridge moment. Coordinate with the lidr CLAUDE.md.
12. **Wire up MLflow for experiment tracking.** Replace the timestamped-folder report + CSV log with proper logged runs and a comparison UI. *Deferred: the CSV results log (#1) covers the "did this help?" need until the run count is large enough to justify the heavier tooling.*

## Active Task

_Nothing currently in-flight._

Signal accuracy test harness shipped (2026-05-22, see Recent Changes). Next: port the five remaining signals starting with RSI (Next Up #3).

<!-- Update this section when work is in progress. Replace with `_Nothing currently in-flight._`
     when paused. Keep it short: what's being built, where it was left off, mid-flight decisions. -->

## Recent Changes

### 2026-05-26 — Restructure: README is the public face, CLAUDE.md is Claude-facing (Batch 6 of 6)

Final batch in the drift-fix series. Public-facing material now lives in README; CLAUDE.md is leaner and Claude-focused.

- **README** gained an **Architecture** section with an ASCII pipeline diagram (configs → pipeline orchestrator → data/signals/models/backtest → eval → three output artifacts), a one-paragraph **Stack** description, a sentence describing what the HTML report contains, and a real **Project layout** block (no longer just "see CLAUDE.md").
- **CLAUDE.md** "Commands" section trimmed to a one-liner pointer at README's Quick Start + CLI sections — the canonical home for "how to run it" is now README. Everything else (Project Goal, Stack, How the pipeline works, Folder map, Conventions, Key Decisions, Gotchas, Next Up, Active Task) stays as the deep-dive reference Claude needs while working on the code.
- **Screenshot deferred.** A PNG of a rendered report would round out the README but requires actually rendering and capturing one — out of band for this session. The architecture diagram + one-paragraph description of report contents covers most of the orientation need until a screenshot is added (one-line PR when convenient).

That closes a 6-batch arc driven by a drift report (read README, CLAUDE.md, and the codebase; produce findings in three buckets — documented-but-false / true-but-undocumented / ambiguous; then implement). Net effect across the six commits: every claim in CLAUDE.md now matches code; every public-facing contract (CLI, YAML schema, JSON artifact, results_log columns, license, contributing rules) is documented in README/CONTRIBUTING; the SPY baseline number traces to a real CSV row; and README is self-sufficient as the entry point.

### 2026-05-26 — Drift-fix pass (Batch 1 of 6)

Drift audit against the code surfaced several stale facts in CLAUDE.md and one dead config field. Fixes in this commit, no behavior change:

- **CLAUDE.md Conventions**: protocol names corrected — `Signal` → `SignalFn` (real name at `signals/base.py:19`); Model protocol now lists all three methods (`fit`, `predict_proba`, `predict`) with correct `predict_proba -> np.ndarray` return type, not a DataFrame; signal test table is `SIGNAL_CASES`, not `SIGNALS`.
- **Pipeline walkthrough step 7 + folder map for `metrics.py`**: dropped the phantom "hit rate" function; replaced with the actual exports (`classification_metrics`, `strategy_metrics`, `by_year`, `performance_by_year`) and noted that the equity curve itself is built in `backtest/engine.py::add_strategy_returns`, not `metrics.py`.
- **Project Goal**: backtest range corrected from "pre-2008 to today" to "2005 to today" (matches `configs/baseline.yaml` `start_date: 2005-01-01`).
- **Gotchas**: dropped the two Cowork-specific bullets (cloud-review truncating CLAUDE.md, and Cowork sandbox stale-mount) — those are environment-specific and unhelpful to contributors not on Anthropic's cloud dev stack.
- **Next Up**: rewrote #8 (artifact JSON schema is already implemented at `schema_version: 1`; remaining work is "evolve as lidr's API needs firm up"). Promoted two previously-buried items to standalone roadmap entries: **calibrate `predict_proba` via Platt/isotonic** (was a parenthetical under #7), and **migrate binary target to 3-class BUY/HOLD/SELL** (was an offhand line in the pipeline walkthrough only). Both stay edge-gated. Renumbered downstream items.
- **Configs**: removed dead `output.report_html: true` field from both YAMLs — `pipeline.py` always writes the HTML report unconditionally and never read this flag. Kept `output.predictions_json` which is honored.
- **`strategy_metrics` empty-path**: return dict now includes `final_equity: 0.0` so callers don't need to branch on shape (the non-empty path already returned it).
- **README**: "synthetic-data fallback" → "synthetic-data alternative" — `source: synthetic` is an explicit config switch, not an automatic fallback when yfinance fails.

This is the first of six batches mapped out from a full drift report. Remaining: LICENSE + CONTRIBUTING + badges; README YAML/JSON schema reference; smaller operational notes (yfinance auto-adjust, cache invalidation); re-run SPY baseline so README numbers trace to `results_log.csv`; README ↔ CLAUDE.md restructure with architecture diagram + report screenshot.

### 2026-05-26 — Adopt protected-main PR workflow

Following the same workflow change applied to the sibling `lidr` project today: `main` is now protected by GitHub branch protection rules, all changes land via PR, required green CI (`make test` + `make lint`) before merge. No code-review requirement (sole developer); the gate is just "CI green."

Unlike `lidr`, lidr-ml has no Vercel deploy step — the pipeline produces a JSON artifact that's consumed manually. So CI passing is the only required gate; there is no preview-deploy step.

Doc-only changes in this commit:

- `CONTRIBUTING.md`: new top-level **Workflow** section with the full branch → PR → CI → merge → cleanup sequence, including the admin-override-for-emergencies note.
- `CLAUDE.md` Conventions: new bullet at the top of the section flagging that `main` is protected and pointing at CONTRIBUTING.md for the procedure.

Branch protection settings themselves live in GitHub repo settings, not in git — applied separately in the GitHub UI alongside the same setup on `lidr`.

### 2026-05-26 — Relicense MIT → PolyForm Noncommercial 1.0.0

Relicensed both lidr-ml and lidr from MIT to PolyForm Noncommercial 1.0.0 after explicitly working through the question "is MIT the right license?" The honest answer: no. The desired stance is "anyone can read and learn from this; non-commercial / research / educational use is welcome without asking; commercial use requires permission" — which is not MIT (MIT allows unrestricted commercial use including closed-source SaaS clones) and is not any OSI-approved open-source license. It's a **source-available** stance, and PolyForm Noncommercial is the lawyer-drafted license that fits it.

Mechanics, both repos: replaced `LICENSE` with the PolyForm Noncommercial 1.0.0 text (copyright header naming Boon Boonyasirichok); updated `pyproject.toml` `license` field (lidr-ml) and added `"license": "SEE LICENSE IN LICENSE"` to `package.json` (lidr — PolyForm-Noncommercial-1.0.0 is on SPDX but npm prefers the file pointer when also using `private: true`); swapped the README MIT badge for a PolyForm Noncommercial badge; rewrote the README "License" section in plain English (what's allowed, what isn't, how to ask for a commercial license); added a "License of contributions" section to CONTRIBUTING.md so a contributor knows their PR also lands under PolyForm Noncommercial.

Worth knowing for future Claude: this is a one-way change in the sense that anyone who grabbed either repo under MIT before 2026-05-26 keeps MIT rights to *that snapshot* — but both repos are recent and have effectively zero external adoption, so the practical exposure is nil. Future contributors agree to PolyForm at the time of contribution. The author can always commercialize freely (the license restricts other people, not the licensor) and can dual-license to specific parties as needed. If a future decision is to relax back to a permissive license (MIT, Apache), that requires the consent of any external contributor whose code is still in the repo at that point — so the sooner-the-better window for moving back the other way is now, while there are zero external contributors. Keep this in mind before merging the first community PR.

### 2026-05-26 — Adopt Conventions + Gotchas; partition for one-fact-one-place

Two-part pass on this file. (1) Adopted the missing two sections from the standard CLAUDE.md template: **Conventions** (slotted before Key Decisions — the procedural rules: signal/model protocols, the three-things-must-land-together rule for new signals, expanding-window-only, transaction costs always modeled, benchmark always shown, CI requirement) and **Gotchas** (slotted after Key Decisions — non-obvious things that bit us: the N-day-vs-1-day equity-curve overlap bug, cloud-review iterations silently truncating CLAUDE.md, the Cowork stale-mount issue, Python 3.14 parquet wheels, Typer single-command flattening, yfinance survivorship bias, `class_weight="balanced"` distorting `predict_proba`). The other template sections (What this is / Architecture / Key files / Commands / Current focus) were already covered by Project Goal / Architecture / Folder map / Commands / Active Task and were not duplicated. (2) **Repartitioned to remove duplication.** The first cut left the same fact in two or three sections (lookahead-safety, transaction costs, benchmark-in-reports, the 1-day-vs-N-day rule, survivorship bias, `class_weight=balanced`). Tightened the rule: **Conventions = the rule**, **Key Decisions = why we chose X over Y**, **Gotchas = what bit us**. Each fact now appears once; cross-references replace the duplicates. Trimmed Key Decisions from 10 items to 5 (one Python project / source of truth in lidr / synthetic-data fallback / outputs are calibrated probabilities / backtest doesn't produce a deployable model); rules and warnings moved out to the section that owns them. Dropped the "Anti-patterns we have hit before" sub-list inside Conventions in favor of a pointer to Gotchas. Added the one-fact-one-place rule to Maintenance Instructions.

### 2026-05-26 — Documentation & tech-debt cleanup pass

Hygiene pass with no behavior changes. (1) **README.md refreshed**: "What's in the box" now lists the benchmark-vs-buy-hold report, the `artifacts/results_log.csv` cross-run log, and the signal accuracy + CI harness — i.e. the work that landed 2026-05-20 through 2026-05-22. "What's next" rewritten around the **edge gate** framing (prove the model beats buy-and-hold before any serving/integration), with the gated items (final-model fit/serialize, artifact schema, lidr wiring, MLflow) called out explicitly. Added the **SPY baseline status** line so the README states the current bar without needing to open CLAUDE.md. Also documented the `lidr-ml` console-script entry point as an alternative to `python -m lidr_ml`. (2) **CLAUDE.md drift fixes**: Stack footnote for "Planned but not yet added" now cross-references roadmap numbers (LightGBM = #4, MLflow = #10/deferred, vectorbt = off-roadmap until the engine becomes a bottleneck); the "How the pipeline works" list grew a step 8 for the cross-run results log; the **Folder map** was reconciled with reality — added `eval/results_log.py`, `artifacts/results_log.csv`, `tests/conftest.py`, `tests/test_signal_accuracy.py`, `tests/test_strategy_returns.py`, and `.github/workflows/test.yml`; removed the never-used `data/processed/` cache layer (no code reads or writes it). (3) **De-stub code language**: `pipeline.py` (single-ticker comment + NotImplementedError message) and `backtest/engine.py::add_strategy_returns` docstring no longer call themselves "stubs" — these have been the real implementation since the 2026-05-20 equity-curve fix. (4) **Repo hygiene**: deleted `data/processed/` and its `.gitignore` rule; deleted the stray `pytest-cache-files-o4434ewm/` directory at the repo root (one-off pytest cache from somewhere); added `pytest-cache-files-*/` to `.gitignore`; extended `make clean` to sweep that pattern; added a new `make clean-reports` target so accumulated timestamped `reports/<config>-<ts>/` dirs can be swept without nuking the parent. No source-of-truth or test changes; tests + lint still green.

### 2026-05-22 — Signal accuracy test harness + CI (Next Up #2)

New file `tests/test_signal_accuracy.py` — two-layer accuracy validation for every registered signal. Layer 1: element-wise comparison against a simple inline reference formula (catches wrong `min_periods`, wrong normalization, wrong formula). Layer 2: spot checks against hand-derived values on an arithmetic price series (`close = [100, 101, ...]`), where SMAs reduce to sums of consecutive integers and can be verified without running code. `ACCURACY_CASES` table is the extension point: add one entry per signal when porting.

New file `tests/conftest.py` — shared `synthetic_prices` fixture (600-day log-normal series, seed=0) extracted from `test_no_lookahead.py` so both test modules use the same data without duplication.

New file `.github/workflows/test.yml` — CI runs `make test` + `make lint` on every push and pull request (Python 3.11, `ubuntu-latest`).

`Makefile` updated: `make test` now calls `python3 -m pytest` instead of bare `pytest` to ensure the correct interpreter (and therefore the installed packages) is used.

Pre-existing lint errors in `src/` cleaned up: unused imports in `engine.py`, `registry.py`; unsorted import blocks in `results_log.py`, `signals/__init__.py`; unused variable in `pipeline.py`; `typer.Argument` call moved to `Annotated` type in `cli.py`. All tests pass; lint is now clean.

### 2026-05-22 — Cross-run results log (Next Up #1)

New file `src/lidr_ml/eval/results_log.py` — `append_run()` appends one row to `artifacts/results_log.csv` after every backtest run. Columns: `run_id`, `config_name`, `ticker`, `oos_start/end`, `n_oos`, `skill_score` (= 1 − log_loss/base_logloss), full-OOS `base_logloss` + `log_loss` + `accuracy`, period breakdowns `base_logloss_2025` / `log_loss_2025` / `base_logloss_2026q1` / `log_loss_2026q1`, strategy and benchmark CAGR/Sharpe/max_dd/final_equity, `excess_cagr`, `excess_sharpe`, `report_path`. The file lives at the `artifacts/` root (not gitignored) and accumulates experiment history across sessions. Pipeline prints a one-liner summary after each append. `pipeline.py` updated to import and call `append_run` (non-fatal; wrapped in try/except). All existing tests pass.

### 2026-05-21 — Roadmap reprioritization around proving the edge

Reset the Next Up order to serve one near-term goal Boon confirmed: prove the model beats buy-and-hold before building any serving/integration plumbing. No code changed — this is a planning pass. Three substantive moves: (1) promoted the **cross-run results log** from #3 to **#1**, on the logic that it's the cheapest item and it makes every later experiment measurable, so it should land before the experiments rather than after; (2) kept the feature/model-building sequence (parity harness → port five signals → LightGBM → stacking → regime features) as the core edge-hunt, with an explicit note that re-running the logistic baseline on all six ported signals is a cheap checkpoint to tell whether the bottleneck is features or model; (3) added an explicit **edge gate** line — the final-model fit/serialize, artifact schema, and lidr wiring stay parked until something beats buy-and-hold — and **deferred MLflow to #10**, since the CSV log covers the "did this help?" need until run volume justifies heavier tooling. Updated the Active Task next-step pointer to match the new numbering. Mirrored the framing in lidr's CLAUDE.md (website items parked behind this work).

### 2026-05-21 — Report readability pass + SPY baseline review

Made the backtest report self-explanatory and ran/reviewed the SPY baseline end to end. Report changes (`eval/report.py` + `eval/metrics.py`): added a **Summary** section translating the config into plain English plus a quick-reference table, stating the out-of-sample span (derived from predictions, not the raw data range); added **base_logloss** (no-skill floor = entropy of the base rate) to the top classification metrics and to the per-year table so log loss is interpretable against its floor; reordered **Buy & Hold before Strategy** in both the comparison and per-year tables (new comparison columns now append to the right); and added **green/red highlighting** — the Strategy column in the comparison table (green = beats buy-and-hold; higher = better, including negative-stored max drawdown), the `strategy_return`/`excess` cells in the per-year table (green = beat buy-and-hold that year by *relative* excess, even if the absolute return was negative), and the `log_loss` cell in the per-year table (green = below the no-skill floor). Verified by rendering the synthetic config and by extracted-function unit tests on the real source.

SPY baseline finding (OOS 2010-10 → 2026-04, 3,914 days): the single-signal logistic model loses to buy-and-hold on every axis — final equity 3.28x vs 8.23x, CAGR ~8.0% vs ~14.5%, Sharpe 0.67 vs 0.89, essentially identical max drawdown (~−33.7%); accuracy (0.556) is below the base rate (0.613) and log loss (~0.693) sits at the no-skill floor. No edge — exactly what one trend feature should produce — and this is the floor every real model must clear. Also corrected a documentation point discovered in review: the pipeline never fits or saves a final model (the `artifacts/models/` folder is aspirational); the backtest's per-split models are throwaway. Added "fit + serialize a final model" as a Next Up item.

Tooling note: the Cowork sandbox was reset mid-session and its mount intermittently served stale/truncated copies of freshly-edited files; verification was done against sandbox-local copies and extracted-function tests. The committed source on disk is authoritative.

### 2026-05-20 — Iteration-readiness: equity-curve fix + strategy-vs-SPY reporting

Made the backtest output trustworthy and able to answer the core question ("is this model better than just holding the index?") before any model work begins. Three changes. (1) **Fixed the overlapping-returns bug** in the equity curve: it was compounding the N-day-forward classification target (`horizon_days`, e.g. 5) on every daily row, counting the same multi-day window ~N times and badly inflating returns. The curve is now marked to market with 1-day-forward returns (`daily_fwd_return` in `pipeline.py`); the N-day return stays purely as the classification target. On the synthetic config this dropped final equity from a nonsensical ~9.4x to a realistic ~1.30x. (2) **Added a Strategy vs Buy & Hold comparison**: `strategy_metrics` is now also computed for the buy-and-hold leg, rendered as a side-by-side table (CAGR, Sharpe, max drawdown, final equity) in the HTML report, and added to the JSON artifact under `metrics.benchmark` (additive — `schema_version` stays 1, existing consumers unaffected). (3) **Added `performance_by_year()`** (`eval/metrics.py`) and a report section showing per-year strategy return vs buy-hold return and excess, so regime-dependence ("worked in the past, not recently") is visible at a glance. Added `tests/test_strategy_returns.py` with three regression tests (compounds each return once, cash days earn nothing, costs charged on position change); full suite is 5 tests, all green, plus a clean synthetic backtest. Also hardened `_json_default` to handle `np.bool_`.

Note on tooling: Cowork's sandbox mount served a stale/truncated view of freshly-edited files during this session (file mtimes weren't advancing, so cached bytecode masked the edits), so verification was run against a sandbox-local copy of the tree. The committed source on disk is correct; if a future run sees "old" behavior after an edit, clear `__pycache__` and confirm the file mtime advanced.

### 2026-05-19 — Initial scaffold + first end-to-end run

Stood up the project structure described in this file: src-layout package, config-driven pipeline, expanding-window walk-forward backtest, HTML report with embedded equity curve, yfinance loader with synthetic-data fallback, one ported signal (SMA crossover), one base model (logistic regression), pytest smoke test that runs the pipeline end-to-end on the synthetic config. No real model performance to speak of yet — the point of this commit is working plumbing, not a working strategy. Verified the synthetic config runs cleanly inside the Cowork sandbox, then Boon ran both `dev_synthetic` and `baseline` configs on his WSL machine (Python 3.14 venv) — both produced HTML reports and JSON prediction artifacts as expected.

Two small fixes shook out during verification that are worth knowing about: (1) Typer collapses single-command apps into a flat CLI, which broke `python -m lidr_ml backtest ...`; added a second command (`list-signals`) so Typer keeps `backtest` as a real subcommand — do not remove `list-signals` until a third real command exists. (2) The data cache originally used parquet, which requires `pyarrow` or `fastparquet`; on Python 3.14 neither has wheels yet. Switched to pickle — zero extra deps, fine for a local-only regeneratable cache. Inline comment in `loaders.py` explains why; don't switch it back without a real reason.

## Maintenance Instructions

If you (a future AI assistant joining this project) make meaningful changes, also update this file in the same session.

- **Keep evergreen sections current.** Project Goal, Architecture, Stack, How the pipeline works, Folder map, Conventions, Key Decisions, Gotchas, Next Up should reflect reality. If your work invalidates a fact in any of these sections, update it before ending the session.
- **Each fact lives in one section.** Conventions = the rule. Key Decisions = why we chose X over Y. Gotchas = what bit us. If you find yourself writing the same fact in two places, pick the canonical home and cross-reference from the other.
- **Append a dated entry to Recent Changes for each session that produces real changes.** Use a `### YYYY-MM-DD — short title` header followed by a paragraph describing what was done and why. Include decisions and rationale future-Claude would benefit from knowing.
- **Cross-link to lidr.** If a change here affects the integration with lidr (artifact format, signal parity, what the website should consume), update lidr's CLAUDE.md too in the same session.
- **Archive when Recent Changes exceeds 10 entries.** Fold the oldest 5 into a `## Archived Summary` section at the bottom. Preserve decisions and rationale; compress narratives, not insights.
