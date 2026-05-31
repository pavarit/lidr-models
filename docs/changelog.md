# Changelog

Full history of meaningful changes. Entries are added here when they age out of CLAUDE.md's Recent Changes section (which keeps only the 5 most recent). Sources: PR descriptions and git history.

---

## 2026-05-28 — Fix: backtests crash on a stock Windows console (cp1252 → UTF-8)

**`make backtest` / `python -m ta_ensemble backtest <config>` now run on a stock Windows shell without `PYTHONIOENCODING=utf-8`.** Both pipelines print progress lines containing `→`; Windows' default console codec is cp1252, which can't encode it, so the run died with `UnicodeEncodeError` at `pipeline.py:57` *before any backtest work* — the env-var workaround noted in the horizon-spike entry below is no longer needed.

Fixed at the CLI layer rather than by de-Unicode-ing print statements (whack-a-mole, and the `→` is intentional in the output): new shared helper `lidr_core.console.ensure_utf8_stdout()` reconfigures stdout/stderr to UTF-8, called once at the top of both `ta_ensemble/cli.py` and `news_sentiment/cli.py`. It's a no-op when the stream is already UTF-8 or can't be reconfigured (e.g. a captured buffer under pytest), so it's safe everywhere. Lives in `lidr_core` because both model packages need it (the harness-reuse convention). Residual edge case: a script that calls `run_pipeline` directly (bypassing the CLI) under cp1252 would still crash — call `ensure_utf8_stdout()` first if you write one.

Infra fix, no behavior/output change → no verification chart needed. Verified by reproducing the crash in a cp1252 PowerShell with `PYTHONIOENCODING` unset (negative control: bare `print('→')` raises `UnicodeEncodeError`), then running the six-signal backtest in the same shell to completion. 44/44 tests pass, `ruff check` clean.

---

## 2026-05-28 — Horizon spike: longer target horizons make the TA model worse, not better

**Plain result: lengthening the prediction horizon doesn't help — it hurts.** Swept `target.horizon_days ∈ {5, 10, 20, 60}` crossed with three model classes — unweighted logistic, `class_weight=balanced` logistic, and LightGBM — on the six-signal TA model, SPY 2005→2026-05. 12 new configs (`horizon_h{N}_{logistic,logistic_weighted,lightgbm}.yaml`), 12 new `results_log.csv` rows.

`skill_score` (= 1 − log_loss/base_logloss) is **negative at every horizon and degrades monotonically as the horizon lengthens**:

| horizon | logistic | weighted | LightGBM |
|---|---|---|---|
| 5  | −0.0051 | −0.0374 | −0.1478 |
| 10 | −0.0114 | −0.0687 | −0.2075 |
| 20 | −0.0211 | −0.1149 | −0.3169 |
| 60 | −0.0580 | −0.2739 | −0.4833 |

So **h5 is the least-bad horizon in all three classes**, the opposite of the "5-day target is too noisy, go monthly" hypothesis. Why: the unweighted logistic just predicts the majority class (`pred_rate` ≈ 1.0, accuracy ≈ base rate at every horizon); as the horizon lengthens the base rate climbs 0.61 → 0.77, which *lowers* the no-skill floor (`base_logloss` 0.668 → 0.544), so the model's fixed probability miscalibration becomes a larger fraction of a smaller floor. The six TA signals carry no usable directional information at any horizon — consistent with the LightGBM checkpoint's "features/target is the bottleneck" conclusion, now extended along the horizon axis. Strategy CAGR is **informational only**: a high-base-rate config that's ~always-long just tracks buy-and-hold — its excess is exposure, not skill.

**Verification:** `scripts/verify_horizon_sweep.py` recomputes every metric from the prediction-artifact JSONs and gates on four checks, all PASS: (1) parity — each h5 run reproduces its committed source-config row; (2) chart-vs-log — recomputed `skill_score` matches `results_log.csv` within 5e-4; (3) n_oos per config matches; (4) base_rate surfaced beside accuracy. Chart + table in `docs/_pr_evidence/horizon_sweep/` (removed in the cleanup commit; chart URL in the PR pinned to the full commit SHA).

**Implication for Task 2:** `news_v0.yaml` keeps `target.horizon_days: 5`. Closed roadmap direction (a); live directions are now (b) magnitude-regression target and (c) regime features. Deleted `docs/plans/horizon-spike.md` in the cleanup commit per the disposable-plan-doc convention.

---

## 2026-05-28 — CLAUDE.md context-bleed trim, batch 3 of 3: Folder map condensation (docs only)

Final batch of the three-batch context-bleed arc ([batch 1 = PR #29](https://github.com/pavarit/lidr-models/pull/29), [batch 2 = PR #30](https://github.com/pavarit/lidr-models/pull/30)). Targeted the `## Folder map` section — ~100 lines enumerating nearly every `.py` file with a one-line annotation.

Chose Path B (delete annotations, rely on module docstrings) over Path A (move them into docstrings): coverage was already 90–100% across all three packages — every logic-bearing module already had a docstring that said what its folder-map annotation said. Per-file annotations were a one-fact-one-place violation. Path B avoided ~30-file churn and didn't touch `eval/report.py`/`metrics.py` (which would have triggered the refresh-sample-report rule). The folder map became a high-level 3-package overview with a pointer to module docstrings.

Sole code touch: one-line module docstring added to `packages/lidr_core/src/lidr_core/models/__init__.py` — the only annotated logic module that lacked one.

---

## Archived summary

Older entries folded down per the maintenance rule. Decisions and rationale preserved; narratives compressed. Sources: PRs and full entries in git history.

### CLAUDE.md context-bleed trims (batches 1–2) + Task 2 PR-A & plan revision (2026-05-28)

Five same-day entries folded together. **Context-bleed batch 1** ([PR #29](https://github.com/pavarit/lidr-models/pull/29)): four mechanical trims — horizon-spike kickoff prompt moved to its own plan doc; the PR-evidence SHA anecdote compressed to rule + git-history pointer; the duplicated session-wrap bullet in Maintenance Instructions replaced with a pointer to the global CLAUDE.md; the hosting-repo rename parenthetical dropped. **Context-bleed batch 2** ([PR #30](https://github.com/pavarit/lidr-models/pull/30)): extracted the buried model-PR diagnostics + reporting lessons into a top-level **Diagnostic Playbook**, choosing a section over an external doc because the bleed problem was decay (out-of-sight), not location. *(Superseded 2026-05-29: that Playbook moved into the `verify-evidence` skill.)* **Maintenance fold** ([PR #28](https://github.com/pavarit/lidr-models/pull/28)): folded the prior oldest-5 into the "Six-signal TA model build-out…" sub-section and restored the missing Task 1 (PR #23) header.

**Task 2 PR-A** ([commit `b9ce76a`](https://github.com/pavarit/lidr-models/commit/b9ce76a)): filled the `news_sentiment` shell with the structural harness PR-B/PR-C bolt logic onto — no model-edge claim. ~25 files mirroring `ta_ensemble`; six data adapters (five real, one stub); a collector that dedups by `content_hash` and persists raw items keyed by their true `published_at`; a working `LexiconScorer` with `FinBertScorer`/`LlmScorer` stubbed but their cache + per-run budget cap + spend log already functioning; three lookahead-safe features built on a one-trading-day forward shift. 20 tests, suite 44, all green. **Plan revision** (post-PR-A): credential setup found Reddit blocked by its Responsible Builder Policy, Tiingo actually $30/mo with 3-month history, and pytrends archived 2025-04-17 — so the original PR-B data stack was invalid. Chose **Path 4**: drop Reddit/Tiingo/Google Trends; add EODHD ($19.99/mo), Apewisdom (free), Finnhub (free). Net $19.99/mo.

### Multi-model restructure + edge-gate model checkpoints (2026-05-27)

**Planning: multi-model architecture (docs only).** Decided to reorganize around the JSON-artifact *contract* rather than around models: rename `lidr-ml` → `lidr-models` and split into a `lidr_core` shared harness plus per-model packages. Rationale + alternatives live in `docs/adr/0001-multi-model-repo-architecture.md`. Doc-hygiene rule set here: ADR + `docs/research/data-sources.md` are durable and stay; `docs/plans/` docs are disposable and self-delete on their task's merge.

**Task 1: monorepo restructure + schema v2 ([PR #23](https://github.com/pavarit/lidr-models/pull/23)).** `src/lidr_ml/` → three packages under `packages/`; each owns its `pyproject.toml`; root is dev-tool-only; `make install` installs all three editable; CI updated. Configs gained `model_id` + `model_version`. **Artifact contract formalized at `schema_version: 2`**: `build_artifact` + `write_artifact` validate against `artifact.schema.json`; predictions land under `artifacts/predictions/<model_id>/`. Parity gate: re-ran `baseline_six_signals_unweighted.yaml` before/after → `skill_score -0.005104`, `cagr 0.142454`, `n_oos 3851` bit-identical. 24/24 tests, lint clean.

**LightGBM checkpoint: still no edge ([PR #20](https://github.com/pavarit/lidr-models/pull/20)).** Added LightGBM as the second base learner. Headline: **LightGBM is worse than logistic** — `skill_score -0.148` vs -0.005. But wrapping it in `CalibratedClassifierCV(isotonic, cv=3)` moved it to -0.004 → raw probabilities are *miscalibrated*, not anti-informative. Three well-behaved configs all cluster at the no-skill floor → **the bottleneck is the features/target, not the model class.** Stacking parked under the edge gate.

**`class_weight=None` sanity check.** Removing `class_weight=balanced` shrank the distance from no-skill ~7× (`skill_score -0.0374 → -0.0051`): balanced was actively harmful — forcing confidently-wrong predictions on a 60/40 problem. Still not skill: `pred_rate 0.996`, predicts ≈base_rate every day.

**Six-signal logistic baseline checkpoint.** Added rsi/macd/bollinger/breakout/volume. Six features vs one are equally non-skilled in probability space (skill_score differs by 0.0004 — sampling noise); the equity gap is exposure (`pred_rate` ~0.60 vs ~0.75), not skill. Bottleneck is the linear-model assumption → motivated the LightGBM checkpoint.

### Six-signal TA model build-out + CLAUDE.md drift-fix arc (2026-05-26 → 2026-05-27)

**Drift-fix arc (batches 1–6, 2026-05-26).** Drift audit between CLAUDE.md and code fixed stale facts: protocol name `Signal` → `SignalFn`; Model protocol's three methods; signal test table `SIGNALS` → `SIGNAL_CASES`; phantom `hit rate` removed; backtest range "pre-2008" → "2005"; dead `output.report_html` removed. README gained Architecture section (ASCII pipeline diagram), Stack paragraph, real Project layout block. CLAUDE.md Commands trimmed to a one-line pointer at README.

**Five lidr signals ported + PR-evidence convention established (2026-05-27, PRs #5/#7/#8/#9/#10).** Shipped RSI, MACD, Bollinger, breakout, volume. Each port matched lidr's TS implementation: RSI/MACD/breakout/volume exact bit-match; Bollinger 1.5e-11. **PR-evidence convention formalized**: `scripts/verify_<thing>.py` → `docs/_pr_evidence/<thing>/{chart.png, evidence.md}`, embedded chart pinned to full-40-char commit SHA via `raw.githubusercontent.com`, removed in cleanup commit so `main` stays free of review artifacts. `ACCURACY_CASES` schema extended to 5-tuple with per-case `prices_factory`. Tolerance loosened `rtol=1e-12` → `rtol=1e-8`. Test count 7 → 22.

**Signals explainer doc (2026-05-27, [PR #13](https://github.com/pavarit/lidr-models/pull/13)).** [`docs/signals.md`](signals.md) — standalone first-time-reader explainer for all six signals with per-signal SPY charts. Per-signal template: what it watches → what the number means → math table → SPY chart → recent history → failure modes → parameters.

**Dup-date fix in expanding-window backtest (2026-05-27, [PR #16](https://github.com/pavarit/lidr-models/pull/16)).** `expanding_window_backtest` had inclusive right endpoint on the test slice, so the boundary date between split N and N+1 was predicted in both splits' output (~0.3% of rows). Fix: right endpoint exclusive except on the final split. New regression tests added; engine raises on non-unique index. Pre-fix `results_log.csv` rows are very slightly off — see Gotchas in CLAUDE.md.

### Repo hygiene + workflow setup (2026-05-22 → 2026-05-26)

**Signal accuracy test harness + CI (2026-05-22).** Added `tests/test_signal_accuracy.py` — two-layer validation per signal: (1) element-wise comparison against an inline reference formula; (2) hand-derived spot checks. `ACCURACY_CASES` table is the extension point when porting new signals. Added `.github/workflows/test.yml` (Python 3.11, `make test` + `make lint` on every push and PR).

**Adopted Conventions + Gotchas sections; partitioned to one-fact-one-place (2026-05-26).** CLAUDE.md: Conventions = the rule, Key Decisions = why we chose X over Y, Gotchas = what bit us. Cross-references replace duplicates.

**MIT → PolyForm Noncommercial 1.0.0 relicense (2026-05-26).** Desired stance: non-commercial use welcome without asking, commercial use requires permission. Replaced LICENSE files, updated `pyproject.toml` / `package.json` metadata, swapped README badges, added "License of contributions" to CONTRIBUTING.md.

**Adopted protected-main PR workflow (2026-05-26).** GitHub branch protection on `main`, all changes via PR, CI gate. CONTRIBUTING.md gained a Workflow section.

### Initial development phase (2026-05-19 → 2026-05-22)

**Bootstrap (2026-05-19).** Stood up the src-layout package, config-driven pipeline, expanding-window walk-forward backtest, HTML report, yfinance + synthetic data loaders, logistic-regression baseline, and an end-to-end smoke test on the synthetic config. Two early gotchas: Typer collapses single-command apps (the `list-signals` command exists to keep `backtest` as a subcommand); Python 3.14 lacks parquet wheels (OHLCV cache uses pickle).

**Equity curve made trustworthy (2026-05-20).** Fixed compounding the N-day forward target on every daily row (inflated equity ~N×). Now marks to market with 1-day-forward returns; N-day return stays as classifier target only. `tests/test_strategy_returns.py` guards the rule. Strategy-vs-Buy-and-Hold comparison added to every report.

**Report readability + first real SPY baseline (2026-05-21).** HTML report gained Summary section; `base_logloss` added next to log loss everywhere; comparison and per-year tables got green/red winner-highlighting. First real SPY baseline: OOS 2010-10 → 2026-04, 3,914 days. Single-signal logistic loses to buy-and-hold on every axis — this is the floor every real model must clear.

**Roadmap reset around the edge gate (2026-05-21).** Reordered Next Up to serve one near-term goal: prove edge before building serving/integration plumbing. Added the **edge gate** line — items past it stay parked until something beats buy-and-hold.

**Cross-run results log (2026-05-22).** New `eval/results_log.py` — `append_run()` appends one row to `artifacts/results_log.csv` after every backtest. Columns: `run_id`, `config_name`, `ticker`, OOS span, `n_oos`, `skill_score`, log losses, strategy and benchmark metrics, excess.
