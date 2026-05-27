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

Planned but not yet added: **LightGBM** (next base learner on the roadmap), **MLflow** (experiment tracking — on the roadmap but deferred behind the CSV results log), **vectorbt** (faster backtest sweeps — not yet on the roadmap; revisit if the custom engine becomes the bottleneck).

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
  baseline_six_signals.yaml  same as baseline but all six signals — the edge-gate checkpoint (still loses)
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
    rsi.py                ported from lidr's lib/signals/rsi.ts (Wilder smoothing)
    macd.py               ported from lidr's lib/signals/macd.ts (normalized histogram)
    bollinger.py          ported from lidr's lib/signals/bollinger.ts (z-score)
    breakout.py           ported from lidr's lib/signals/breakout.ts (position in 252-day range)
    volume.py             ported from lidr's lib/signals/volume.ts (volume ratio vs 50-day avg)
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
docs/
  signals.md              first-time-reader explainer for all six registered signals
  signals/                per-signal PNG charts embedded by signals.md (SPY, full history)
  sample-report/report.html committed sample of the HTML backtest report
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
  3. Add an `ACCURACY_CASES` entry in `tests/test_signal_accuracy.py` — a 5-tuple `(name, params, reference_fn, prices_factory, spot_checks)`. The `reference_fn` should be a *structurally different* implementation of the same algorithm (e.g., loop-based numpy reference vs the signal's vectorized pandas path) so a shared bug between signal and reference is unlikely. The `prices_factory` returns the price fixture used for the layer-2 spot checks (typically chosen so the expected values are derivable by hand without running code; ≥2 spot checks required). The layer-1 element-wise tolerance is `rtol=1e-8`.
- **Outcome-changing PRs need verification evidence in the PR description.** PRs that affect what the pipeline outputs (signal ports, model updates, scoring/calibration changes, JSON-artifact schema changes) need a `## Summary` section explaining the change in plain English plus an embedded chart and dated sanity-check table from real SPY data. For signal ports, also cite a max-absolute-difference number vs lidr's TS implementation. Refactors, doc-only changes, and infra/CI changes don't need this. **Mechanics:** `scripts/verify_<thing>.py` generates `docs/_pr_evidence/<thing>/chart.png` + `evidence.md`, committed to the branch then **removed in a final commit before squash-merge** so `main` stays free of review artifacts; the chart URL in the PR description pins to a commit SHA via `raw.githubusercontent.com` so it survives branch deletion. See PRs #5 / #7 / #8 / #9 / #10 (the five signal ports of 2026-05-27) for the template.
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
- **The backtest does NOT produce a deployable model.** Expanding-window walk-forward fits a fresh model per split, predicts that split's test slice, and discards it — the deliverable is the stitched out-of-sample prediction series, used only for evaluation. There is no single trained model and nothing is written to `artifacts/models/`. Fitting + serializing a deployable model is a separate not-yet-built step (on the roadmap, edge-gated).

## Gotchas

Non-obvious things that bit us. Each entry earned its place by causing a real problem.

- **The equity curve runs on 1-day-forward returns, not the N-day classification target.** Compounding the N-day return on every daily row counts the same window ~N times and inflates final equity by an order of magnitude. See `pipeline.py::run_pipeline` (`daily_fwd_return` is the equity input; `fwd_clean` is the classifier target only) and `tests/test_strategy_returns.py`, which is what makes sure the bug stays fixed.
- **Python 3.14 has no parquet wheels yet.** `data/loaders.py` caches OHLCV in pickle, not parquet. Don't "fix" the loader by switching back to parquet without verifying `pyarrow`/`fastparquet` wheels exist for the target Python — the comment at `loaders.py:50` documents why.
- **Typer collapses single-command apps into a flat CLI.** Removing `list-signals` from `cli.py` would silently break `python -m lidr_ml backtest <config>` (Typer would re-flatten the entry point). Don't remove `list-signals` until a third real command lands.
- **`yfinance` only has currently-listed tickers** — survivorship bias. Fine for SPY/QQQ/sector ETFs; suspicious for individual-name backtests. Don't trust individual-stock results without a CRSP-style source.
- **`class_weight="balanced"` in the baseline distorts `predict_proba` away from true frequencies.** Fine for the up/down decision the backtest evaluates; not OK to ship to lidr as a calibrated probability. Revisit `class_weight` and add an explicit Platt/isotonic calibration step before any prediction artifact is consumed by lidr (on the roadmap as the calibration item).
- **`artifacts/results_log.csv` rows from before 2026-05-27 are slightly off.** The expanding-window backtester used an inclusive right endpoint on the test slice, so the boundary date between split N and split N+1 (`test_end` of N == `test_start` of N+1) was predicted twice and double-compounded in the equity curve. ~0.3% of rows affected; tiny effect on every metric (`accuracy`, `log_loss`, `skill_score`, `cagr`, `sharpe`, `n_oos`). Fixed by making the right endpoint exclusive except on the final split — see `backtest/engine.py::expanding_window_backtest` and `tests/test_backtest_engine.py`. Pre-fix rows weren't re-run; treat any cross-row comparison that straddles 2026-05-27 with this in mind.

## Next Up

Priority order, framed around a single near-term goal: **prove the model has an edge over buy-and-hold** before building any serving/integration plumbing. The SPY baseline currently shows no edge (see Active Task), so the path below is improve-the-model first. Everything past #3 (Add regime features) is explicitly gated on an edge actually appearing.

Cross-references to Next Up items use names, not numbers — see Maintenance Instructions for why. Completed items are removed from this section and live in Recent Changes instead.

1. **Add LightGBM as a second base learner.** Drop-in: implement `models/lightgbm.py` against the same `Model` protocol, add a config that uses it. *A nonlinear learner over six signals is the most likely place an edge first appears.*
2. **Introduce stacking.** Once two base learners exist, add a `StackedModel` whose `fit` trains the base learners via out-of-fold predictions and then trains a meta-learner (logistic regression) on top.
3. **Add regime features.** VIX level, yield-curve slope, 60-day realized vol. Fed into the meta-learner so it can lean on different base models in different environments.

— Edge gate: items below stay parked until something above beats buy-and-hold. —

4. **Add a final-model fit + serialize step.** The backtest only evaluates (throwaway per-split models); nothing fits a model on all data or writes `artifacts/models/`. Before serving live predictions, fit one model on all available history, serialize it, and write the prediction artifact. Trigger: once a model beats buy-and-hold.
5. **Calibrate `predict_proba` via Platt or isotonic regression.** The baseline's `class_weight="balanced"` distorts probabilities away from true frequencies (see Gotchas) — fine for the up/down decision the backtest evaluates, not OK to ship to lidr as a calibrated probability. Switch to `class_weight=None` plus a calibration wrapper before any prediction artifact is consumed by lidr.
6. **Migrate the output from binary to 3-class (BUY / HOLD / SELL).** Today's target is binary (`fwd_return > 0`). The project's stated deliverable is 3-class. Two paths: (a) post-hoc bucketing of the calibrated probability into BUY / HOLD / SELL bands, (b) reformulate the target itself (e.g., three quantile bins of `fwd_return`). Decision punt until a binary model with edge exists; the 3-class formulation changes how "beats buy-and-hold" is measured.
7. **Evolve the artifact JSON schema as lidr's needs firm up.** Schema is already implemented at `schema_version: 1` (see `pipeline.py:174-189`: `config_name`, `ticker`, `generated_at`, `metrics.{classification,strategy,benchmark}`, `predictions[].{date,y_true,y_pred,probability_up}`). Bump the version when lidr's `/api/signals/[ticker]` is ready to consume it and any fields need to change (per-signal contributions, BUY/HOLD/SELL class labels from the 3-class migration, etc.).
8. **Wire lidr's `/api/signals/[ticker]` to read the artifact.** That's the bridge moment. Coordinate with the lidr CLAUDE.md.
9. **Wire up MLflow for experiment tracking.** Replace the timestamped-folder report + CSV log with proper logged runs and a comparison UI. *Deferred: the existing `artifacts/results_log.csv` covers the "did this help?" need until the run count is large enough to justify the heavier tooling.*

## Active Task

**LightGBM as second base learner (Next Up: LightGBM).** The six-feature logistic checkpoint ran 2026-05-27 (see Recent Changes) and finished essentially **tied** with the single-signal baseline in probability space — both score `skill_score ≈ -0.038` (within noise of each other; both anti-informative vs the no-skill floor). Neither beats buy-and-hold on the full 15-year OOS window. The full-window equity chart looks bad for six but the gap is **exposure (pred_rate), not skill**; recent-period numbers (2025 calendar + 2026 Q1) actually favor the six-signal model. The bottleneck is the linear-model assumption, not the feature count. Next move per the roadmap: implement `src/lidr_ml/models/lightgbm.py` against the `Model` protocol, add a config that uses it over the same six signals, see whether a nonlinear learner finds anything the logistic regression can't.

<!-- Update this section when work is in progress. Replace with `_Nothing currently in-flight._`
     when paused. Keep it short: what's being built, where it was left off, mid-flight decisions. -->

## Recent Changes

### 2026-05-27 — Six-signal logistic baseline checkpoint (edge gate stays closed)

Ran `baseline_six_signals.yaml` — same target/model/backtest/costs as `baseline_v1`, only the feature set changed (added rsi, macd, bollinger, breakout, volume to sma_crossover). New row appended to `artifacts/results_log.csv` at `run_id=20260527-120203`. OOS: 2010-12-30 → 2026-04-23, 3,862 days (later start than baseline_v1 because of the 252-day breakout warmup, so only `excess_*` is directly comparable across the two full-window rows). The row was generated *before* the dup-date fix landed (same day, PR #16) so it carries the same ~0.3% double-counting bias as other pre-fix rows — see Gotchas.

**Verdict: edge gate stays closed.** Neither model beats buy-and-hold and neither has positive skill.

**Important reframe.** The initial read was "six features is worse than one feature" (lower accuracy, lower CAGR, lower final equity). That's mechanically true on the full window but **misleading on what the models actually do**:

| | baseline_v1 | baseline_six_signals |
|---|---|---|
| `skill_score` (log-loss vs no-skill floor) | -0.0378 | -0.0374 |
| `mean P(up)` | 0.503 | 0.505 |
| `pred_rate` (model says up) | 0.753 | 0.595 |
| `base_rate` (truth = up) | 0.613 | 0.611 |

In probability space the two models are essentially **equally non-skilled** — skill_score differs by 0.0004 over 3,800+ observations (sampling noise). Both are 0.5-spitters; 99% of six's probabilities fall in [0.40, 0.60]. The headline equity gap is **exposure (pred_rate), not skill**: baseline_v1 is long 75% of days vs six's 60%, and in a 14%-CAGR market more exposure compounds harder. The accuracy gap (-4.1 pp) is the same effect — in a market with `base_rate` ≈ 0.61, predicting "up" 75% of the time accidentally matches truth more often than predicting "up" 60% of the time.

**Recent windows favor six.** On 2025 calendar (n=250, same window for both), six **beats v1 decisively** — accuracy 0.548 vs 0.496, skill_score -0.053 vs -0.057, total return +15.2% vs +6.4% (excess vs B&H -3.0 pp vs -11.8 pp). On 2026 Q1 (n=61), six predicts up every day and **exactly matches B&H** (0% excess); v1 takes a few wrong cash days and loses 3.15 pp. baseline_v1's equity curve has long flatlines (cash for months at a time, including the entire post-April-2025 rally) that baseline_six does not.

**Conclusion: the bottleneck is the linear-model assumption, not the feature count.** Adding five orthogonal signals didn't move skill_score and made accuracy *worse* on the full window — but modestly improved recent-period behavior. Next move per the roadmap is LightGBM (now Active Task).

**Workflow lessons worth keeping.**
- **Don't conflate accuracy with skill when `base_rate` is far from 0.5.** Accuracy of 0.557 sounds OK until you notice base_rate is 0.613 (predict-always-up gets 0.613). `skill_score` (= 1 − log_loss / base_logloss) controls for this; report it next to accuracy whenever both are visible.
- **When a result is surprising or contradicts a prior, dig before publishing.** Initial PR framing led with "six is worse"; the user pushed back ("Are you sure?"), the diagnostic exposed the pred_rate/skill_score story, the PR was reframed before merge. The reframe was the more interesting finding.
- **Two charts beat one when the data spans years.** The full-window log chart compresses recent behavior into pixels; a 2024-onwards linear zoom (`docs/_pr_evidence/six_signal_baseline/chart_recent.png` at the time of merge, then removed in cleanup) is where the actual recent behavior is legible.
- **Aborted bug worth flagging.** The verify script's first version had `prices["Close"]` (capital C). The yfinance loader normalizes columns to lowercase (`loaders.py:77`); the script's fallback `prices.iloc[:, 0]` silently returned the `open` column, producing an equity curve that ended at ~0.6× when results_log said 2.47×. Caught by sanity-checking the chart against the logged final equity. Lesson: chart-vs-log cross-check is a fast first-pass test for any one-off verification script that re-derives metrics.

### 2026-05-27 — Fix duplicate-boundary-date bug in expanding-window backtest

Caught while diagnosing the six-signal checkpoint: `backtest/engine.py::expanding_window_backtest` had `test_mask = (idx >= test_start) & (idx <= test_end)` and set `train_end = test_end` at the bottom of the loop. So the boundary date between split N and split N+1 (the actual idx date equal to `test_end_N == test_start_{N+1}`) was predicted in **both** splits' output slices. Visible in the committed prediction JSONs as duplicate dates at ~annual cadence — baseline_v1 had 12 dups out of 3,914, baseline_six had 11 out of 3,862 (~0.3%).

**Fix.** Right endpoint of the test slice is now exclusive except on the final split (so the last data point still gets predicted). New regression tests in `tests/test_backtest_engine.py`: `test_predictions_index_is_unique` and `test_consecutive_splits_do_not_overlap`. Both fail against the pre-fix engine — verified by stashing the fix and re-running. Engine also now raises `AssertionError` if `pd.concat(preds)` ends up with a non-unique index, so the bug can't silently reappear.

**Caveat noted in Gotchas.** All `artifacts/results_log.csv` rows from before this commit are very slightly off (accuracy, log_loss, skill_score, CAGR, Sharpe, n_oos all touched by the ~0.3% double-counted days). Didn't re-run the headline configs in this PR — the effect is below the noise floor for any conclusion drawn from those rows so far, and re-running requires yfinance. Future cross-row comparisons that straddle 2026-05-27 should be aware of this.

### 2026-05-27 — Signals explainer doc (PR #13)

Shipped a standalone first-time-reader explainer for the six implemented signals at [`docs/signals.md`](docs/signals.md), with per-signal charts in `docs/signals/`. Closes the gap that the in-code docstrings explained *how* each signal is computed but nothing told a casual reader *why it exists* or *how to read it*. README's "What's in the box" line — which had drifted to "One ported signal: SMA crossover" — also updated to reflect all six and link the new doc.

**Per-signal template** (apply this when adding a future signal — keeps the doc internally consistent): *What it watches* (plain English) → *What the number means* (directional reading: high=BUY or high=SELL, with the trend-following / mean-reversion / conviction family called out) → *How it's calculated* (math table) → *Chart on SPY* → *What you'd have seen in recent history* (3–4 named events) → *When this signal works well, and when it doesn't* (two concrete failure modes) → *Parameters used here* (and *why* — convention vs. real justification).

**Format choices worth remembering.**
- **One combined markdown file**, not per-signal files. Lets a reader compare signals side-by-side and Ctrl-F across all of them; a TOC at the top handles navigation. README would have bloated past usability if these landed there instead.
- **Chart-generation pattern**: write a throwaway top-level `_gen_signal_docs_charts.py` that calls each signal on the cached `data/raw/SPY_*.pkl`, writes PNGs into `docs/signals/`, then delete the script. Commit the outputs, not the generator. Same spirit as the PR-evidence pattern from PRs #5-#10 — review/regeneration artifacts don't live in `main`.
- **Charts use the full 2005–2026 history on a log-scale price panel**, so 2008 GFC, 2020 COVID, 2022 bear, and 2025 tariff plunge are all simultaneously visible. Use HTML `<img>` tags with `width="100%"` (not bare `![]()`) so charts embed inline in every markdown renderer, not just GitHub.
- **Trend-following vs mean-reversion vs conviction framing** lives in the intro and is repeated per-signal. A first-time reader's biggest landmine is assuming "high feature value = bullish" universally — false for RSI and Bollinger, undefined for volume. The grouping table at the top neutralises that misconception before they hit any signal section.

**Cross-signal April 2025 callout.** The doc's closing section reuses the April 2025 plunge as a sanity check: five oscillator signals all flagged the same week as their most extreme reading; SMA crossover (a long-window trend indicator) didn't register the move for another five weeks. The contrast is intuition for why a model benefits from multiple signals at all. Computed signal readings on the plunge day (April 4/7/8, 2025) are quoted from a one-off run of the chart-generator script and are not stored anywhere in the repo — if the doc is ever updated, recompute rather than trusting the embedded numbers.

### 2026-05-27 — Five lidr signals ported + PR-evidence convention (Next Up #3 done)

Shipped all five remaining signal ports in PRs #5 (RSI), #7 (MACD), #8 (Bollinger), #9 (breakout), #10 (volume). Each one lands the standard three things — registry entry, `SIGNAL_CASES` row, `ACCURACY_CASES` entry — plus a fourth: review-time **verification evidence** committed to the branch and removed in a cleanup commit before squash-merge. This is now an explicit convention for any outcome-changing PR (see Conventions). The Python pipeline now exposes six features (`sma_crossover` + the five new ones).

**The five signals.** All match lidr's TS implementation when run on SPY 2023–2026 (783–1,086 days each, depending on warmup). Features emitted are the raw continuous quantities (not the TS's discrete BUY/SELL action or clipped confidence) so the model can learn its own thresholds:

| Signal | Feature emitted | Parity vs lidr TS |
|---|---|---|
| `rsi` (period=14) | Wilder-smoothed RSI value, 0–100 | exact bit-match (0 over 820 dates) |
| `macd` (12/26/9) | `(macd_line − signal_line) / slow_ema` (normalized histogram) | exact bit-match (0 over 795 dates) |
| `bollinger` (period=20) | `(close − sma_N) / std_N` z-score | 1.5e-11 over 815 dates (streaming vs direct std) |
| `breakout` (period=252) | `(close − low_N) / (high_N − low_N)` position in 52-week range | exact bit-match (0 over 1,086 dates) |
| `volume` (period=50) | `volume / volume.rolling(N).mean()` ratio | exact bit-match (0 over 785 dates) |

**Cross-signal validation: April 2025 SPY plunge agreed on by all five signals.** A useful side-effect of running all five against the same SPY series: each independently flagged early April 2025 as the most extreme bearish event in the 2023–2026 window. RSI = 21.59 (deepest oversold), Bollinger z = −3.61σ (deepest stretched-down), MACD histogram = −6.36 (deepest bearish momentum-of-momentum), breakout feature ≈ 0.07 (deepest dip toward the 52-week low), volume ratio = 3.82× on 256.6M shares (highest-conviction selling day in 3+ years). Five independent measurements agreeing on the same day is a strong correctness signal for the port effort as a whole.

**PR-evidence pattern formalized.** Each port followed the same review-time flow: `scripts/verify_<signal>.py` runs the Python signal **and** a literal JS transcription of lidr's TS on the same SPY closes, computes a parity number, renders a chart, writes `docs/_pr_evidence/<signal>/{chart.png, evidence.md}`. The chart is embedded in the PR description via `raw.githubusercontent.com/<owner>/<repo>/<commit-sha>/...` pinned to a specific SHA. A final cleanup commit removes the script + evidence before squash-merge — git's squash collapses the add+delete pair, so `main` stays free of review artifacts while the PR comment's image URL keeps working (commit-SHA persistence outlasts branch deletion). PR description top section is `## Summary` (no "for reviewers" suffix) and includes a plain-English explanation of what the signal is *before* the technical sections, so a non-technical reviewer can approve the output without code expertise.

**Accuracy-test schema extended (PR #5).** The `ACCURACY_CASES` table now uses a 5-tuple `(name, params, reference_fn, prices_factory, spot_checks)` — adds a per-case `prices_factory` callable so each signal can pick the price fixture that makes its spot checks hand-derivable. Examples: RSI uses a `zigzag_prices` fixture (deltas alternating +2/−1, yields clean 200/3 and 3000/43 rationals at seed and first-recursion step); MACD uses `constant_prices` (every EMA = 100, normalized feature = 0 post-warmup, exercises the warmup boundary); Bollinger reuses `arithmetic_prices` (z-score is constant 19/√133 across all valid indices); breakout uses a new `step_prices` fixture engineered to land at positions 0.0, 0.5, 1.0 at three specific indices; volume uses `volume_spike_prices` (single spike yields 20/11, 10/11, 1.0 ratios).

**Test-suite tolerance loosened, `rtol=1e-12` → `rtol=1e-8` (PR #8).** RSI and MACD pass at exact bit-match because they're recursive (one float op per step) and the loop-based numpy reference computes in the same order as the signal. Rolling-window signals don't: pandas' `rolling(N).std()` uses a streaming Welford-style accumulator that disagrees with naive `numpy.std`-per-window at ~1e-11 even though both are correct population std. The new tolerance is still ~1e5 tighter than any real-bug threshold (real bugs show 1e-3 or larger).

**Spawned-task follow-up shipped in PR #6 (parallel session).** Noticed during PR #5 that the `dev_synthetic` smoke test was appending a row to the tracked `artifacts/results_log.csv` on every test run (2 rows per local pytest, plus 2 per CI run). PR #6 added an `output.results_log` config flag (default `true`; `dev_synthetic.yaml` sets it `false`) plus an assertion in `tests/test_pipeline_smoke.py` that the log file size doesn't grow during the smoke test.

Test count went from 7 → 22 across this session (3 added per signal × 5 signals). All passing; lint clean across all PRs.

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

## Archived Summary

Older entries folded down per the Maintenance Instructions rule (Recent Changes exceeds 10 → fold oldest 5). Decisions and rationale preserved; narratives compressed. Sources: PRs and full entries in git history before commit `<this PR>`.

### Initial development phase (2026-05-19 → 2026-05-22)

**Bootstrap (2026-05-19).** Stood up the src-layout package, config-driven pipeline, expanding-window walk-forward backtest, HTML report, yfinance + synthetic data loaders, logistic-regression baseline, and an end-to-end smoke test on the synthetic config. Two early gotchas surfaced and remain documented in **Gotchas**: Typer collapses single-command apps into a flat CLI (the `list-signals` command exists to keep `backtest` as a subcommand — don't remove it until a third real command exists); and Python 3.14 lacks parquet wheels, so the OHLCV cache uses pickle.

**Equity curve made trustworthy (2026-05-20).** The original equity curve was compounding the N-day forward classification target on every daily row, which counts overlapping windows ~N times and inflated final equity by an order of magnitude. Fixed to mark to market with **1-day-forward returns**; the N-day return stays purely as the classification target. `tests/test_strategy_returns.py` was added at the same time and guards the rule (one regression test per failure mode: compounds each return once / cash days earn nothing / costs charged on position change). The same change added a Strategy-vs-Buy-and-Hold comparison to every report (additive to `metrics.benchmark` in the JSON artifact, `schema_version` unchanged) and a per-year breakdown table so regime-dependence is visible. Also hardened the JSON serializer to handle `np.bool_`.

**Report readability + first real SPY baseline (2026-05-21, AM).** The HTML report gained a Summary section translating the config into plain English plus a quick-reference table; `base_logloss` (the no-skill floor) was added next to log loss everywhere; comparison and per-year tables got green/red winner-highlighting (Strategy column, excess cells, per-year log-loss cells). End-to-end SPY baseline ran for the first time on real data: OOS 2010-10 → 2026-04, 3,914 days. Single-signal logistic loses to buy-and-hold on every axis — final equity 3.28× vs 8.23×, CAGR ~8.0% vs ~14.5%, Sharpe 0.67 vs 0.89, near-identical max drawdown (~−33.7%); accuracy 0.556 is below the base rate 0.613, log loss sits at the no-skill floor. **This is the floor every real model must clear** — the framing for the current "edge gate" in Next Up.

**Roadmap reset around the edge gate (2026-05-21, PM).** Reordered Next Up to serve one near-term goal: prove edge before building serving/integration plumbing. Promoted the cross-run results log to #1 (cheapest item, makes every later experiment measurable). Kept the feature → model build-out sequence (parity harness → port five signals → LightGBM → stacking → regime features) as the core edge-hunt, with an explicit note that re-running the logistic baseline on all six ported signals is a cheap checkpoint to tell whether the bottleneck is features or model. Added the **edge gate** line — items past it (final-model fit/serialize, artifact schema evolution, lidr wiring) stay parked until something beats buy-and-hold. Deferred MLflow to #10 since the CSV log covers the "did this help?" need until run volume justifies heavier tooling. Mirrored the framing in lidr's CLAUDE.md.

**Cross-run results log (Next Up #1 shipped, 2026-05-22).** New file `src/lidr_ml/eval/results_log.py` — `append_run()` appends one row to `artifacts/results_log.csv` after every backtest run. Columns include `run_id`, `config_name`, `ticker`, OOS span, `n_oos`, `skill_score` (= 1 − log_loss/base_logloss), full-OOS and per-period log losses with base-rate floors, strategy and benchmark CAGR/Sharpe/max-drawdown/final-equity, and excess. The file accumulates across sessions; `pipeline.py` calls `append_run` non-fatally (wrapped in try/except). Later PR #6 (2026-05-26) added the `output.results_log` opt-out so the dev_synthetic smoke test stops polluting the tracked CSV.

## Maintenance Instructions

If you (a future AI assistant joining this project) make meaningful changes, also update this file in the same session.

- **Keep evergreen sections current.** Project Goal, Architecture, Stack, How the pipeline works, Folder map, Conventions, Key Decisions, Gotchas, Next Up should reflect reality. If your work invalidates a fact in any of these sections, update it before ending the session.
- **Each fact lives in one section.** Conventions = the rule. Key Decisions = why we chose X over Y. Gotchas = what bit us. If you find yourself writing the same fact in two places, pick the canonical home and cross-reference from the other.
- **Append a dated entry to Recent Changes for each session that produces real changes.** Use a `### YYYY-MM-DD — short title` header followed by a paragraph describing what was done and why. Include decisions and rationale future-Claude would benefit from knowing.
- **When a Next Up item ships, remove it (don't strike it through) and renumber the rest.** Document the completion in Recent Changes — that's the single source of truth for "what's been done." Keeping completed items in Next Up duplicates information and makes the section grow monotonically. Cross-references to Next Up items elsewhere in this file should use **names** ("see roadmap: LightGBM"), not numbers, so renumbering doesn't break references. Historical references in Recent Changes (e.g., "Next Up #3 done") are fine to leave as-is — they describe state at the time.
- **Cross-link to lidr.** If a change here affects the integration with lidr (artifact format, signal parity, what the website should consume), update lidr's CLAUDE.md too in the same session.
- **Archive when Recent Changes exceeds 10 entries.** Fold the oldest 5 into a `## Archived Summary` section at the bottom. Preserve decisions and rationale; compress narratives, not insights.
