# Task 1 — Repo restructure: `lidr-ml` → `lidr-models`

> **For:** a Claude Code session. **Type:** mechanical refactor, **no behavior change.** **Blocks:** Task 2.
> **Read first:** the repo `CLAUDE.md` and [`../adr/0001-multi-model-repo-architecture.md`](../adr/0001-multi-model-repo-architecture.md) (the artifact contract design is the "The artifact contract (schema v2)" section there).

## Goal

Turn the single-package `lidr-ml` into the `lidr-models` research monorepo: a shared `lidr_core` package + per-model packages, with the artifact contract formalized. **The headline acceptance test is parity:** the existing `ta_ensemble` (today's six-signal pipeline) must produce the same backtest numbers after the move as before. This is a relocation, not a rewrite — do not change any signal, model, or backtest logic.

## Guiding principles

- **No numerical change.** Re-run `baseline_six_signals_unweighted.yaml` before and after; `skill_score`, `cagr`, `n_oos` must match to full precision. If they don't, a move broke something.
- **One PR, reviewable.** Follow the repo's protected-`main` PR workflow. CI (`make test` + `make lint`) green is the gate.
- **Move, don't rewrite.** Prefer `git mv` so history is preserved. Update imports mechanically.
- **Tests travel with their code.**

## Target structure

```
lidr-models/
  packages/
    lidr_core/
      src/lidr_core/
        backtest/engine.py
        eval/{metrics.py, report.py, results_log.py, leaderboard.py(new)}
        contract/{schema/artifact.schema.json(new), writer.py(new), loader.py(new)}
        protocols/{signal.py, model.py, feature.py(new), datasource.py(new)}
        data/loaders.py
      pyproject.toml  tests/
    ta_ensemble/
      src/ta_ensemble/
        signals/{sma_crossover,rsi,macd,bollinger,breakout,volume}.py + base.py + registry.py
        models/{logistic.py, lightgbm.py}     # or keep generic models in lidr_core — see note
        pipeline.py        # the TA-specific wiring; calls lidr_core for backtest/eval/contract
        cli.py
      configs/  pyproject.toml  tests/
    news_sentiment/        # empty shell created here; filled in Task 2
      src/news_sentiment/  pyproject.toml  README.md (points to Task 2 plan)
  artifacts/
    manifest.json
    predictions/<model_id>/...
  pyproject.toml           # workspace root (uv workspace or equivalent)
  Makefile  README.md  CLAUDE.md  CONTRIBUTING.md  docs/
```

**Note on `models/`.** `logistic.py` and `lightgbm.py` are model-agnostic learners — they could live in `lidr_core/models/` (shared) rather than `ta_ensemble`. Recommended: **move generic learners + the `Model` protocol to `lidr_core`**, since the news model will reuse them. Keep only model-*specific* wiring in each package.

## Migration map (from → to)

| Current (`lidr-ml`) | Target | Notes |
|---|---|---|
| `src/lidr_ml/backtest/engine.py` | `lidr_core/backtest/engine.py` | model-agnostic |
| `src/lidr_ml/eval/metrics.py` | `lidr_core/eval/metrics.py` | |
| `src/lidr_ml/eval/report.py` | `lidr_core/eval/report.py` | |
| `src/lidr_ml/eval/results_log.py` | `lidr_core/eval/results_log.py` | extend schema with `model_id`, `model_version` |
| `src/lidr_ml/data/loaders.py` | `lidr_core/data/loaders.py` | the yfinance/synthetic price loader is shared |
| `src/lidr_ml/signals/base.py` | `lidr_core/protocols/signal.py` | the `SignalFn` protocol is shared |
| `src/lidr_ml/models/base.py` | `lidr_core/protocols/model.py` | the `Model` protocol is shared |
| `src/lidr_ml/models/{logistic,lightgbm}.py` | `lidr_core/models/` | generic learners, reused by both models |
| `src/lidr_ml/signals/{sma_crossover,rsi,macd,bollinger,breakout,volume,registry}.py` | `ta_ensemble/signals/` | TA-specific |
| `src/lidr_ml/pipeline.py` | `ta_ensemble/pipeline.py` | TA wiring; the artifact-writing block moves to `lidr_core/contract/writer.py` and is *called* from here |
| `src/lidr_ml/cli.py`, `__main__.py` | `ta_ensemble/` (+ a root CLI) | keep `list-signals` (Typer flattening gotcha) |
| `configs/*.yaml` | `ta_ensemble/configs/` | add `model_id: ta_ensemble` + `model_version` to each |
| `tests/test_backtest_engine.py`, `test_strategy_returns.py` | `lidr_core/tests/` | guard shared engine |
| `tests/test_no_lookahead.py`, `test_signal_accuracy.py` | `ta_ensemble/tests/` | guard TA signals |
| `tests/test_pipeline_smoke.py`, `conftest.py` | split as appropriate | smoke test exercises the full path |

## New files to create (scaffolding only — minimal logic)

- `lidr_core/contract/schema/artifact.schema.json` — formalize `schema_version: 2` per the ADR's "The artifact contract (schema v2)" section.
- `lidr_core/contract/writer.py` — builds + **validates** the artifact against the schema before writing. Move the dict-building currently in `pipeline.py` here.
- `lidr_core/contract/loader.py` — read + validate (used by tooling/tests; `lidr` has its own TS reader).
- `lidr_core/eval/leaderboard.py` — read all model artifacts → write `artifacts/manifest.json`.
- `lidr_core/protocols/{feature.py, datasource.py}` — interfaces for Task 2 (define the protocol; no implementations yet).
- `news_sentiment/` shell — package skeleton + `README.md` pointing at the Task 2 plan.

## Packaging

- Set up a workspace (recommend `uv` workspaces) so each package has its own `pyproject.toml` and isolated dependency set. `ta_ensemble` and `lidr_core` keep today's deps (pandas, numpy, sklearn, lightgbm, yfinance). `news_sentiment` will add `transformers`/FinBERT etc. in Task 2 without touching the others.
- Update the root `Makefile`: `make backtest CONFIG=...` should still work (resolve config → owning package).

## Verification checklist (definition of done)

1. `make install` succeeds at the workspace root.
2. `make test` + `make lint` green across all packages.
3. **Parity:** `baseline_six_signals_unweighted.yaml` reproduces the pre-move `results_log` row (`skill_score`, `cagr`, `n_oos`) to full precision. Record the before/after numbers in the PR description.
4. The smoke test (`dev_synthetic`) runs end-to-end and still does **not** append to the tracked `results_log.csv`.
5. A produced artifact validates against `artifact.schema.json`, and `leaderboard.py` writes a valid `manifest.json`.
6. `git mv` used so history is preserved; imports updated repo-wide (no stale `lidr_ml.` references).
7. **Delete this plan doc** (`docs/plans/task-1-repo-restructure.md`) as the final step — once Task 1 merges it's stale, and the PR description plus the CLAUDE.md Recent Changes entry capture what was done. This mirrors the repo's existing PR-evidence cleanup habit. The durable docs (the ADR and `docs/research/data-sources.md`) stay.

## Repo rename

- Rename the GitHub repo `lidr-ml` → `lidr-models` (GitHub auto-redirects the old URL, but update remotes).
- Update cross-references: this repo's `CLAUDE.md` + `README.md`, and `lidr`'s `CLAUDE.md` (the sibling-project pointer + GitHub URL).
- Update the local working directory name if desired (`C:\Users\smnk1\Claude\Projects\lidr-models`).

## Out of scope for Task 1

- Any `lidr` (Next.js) code changes beyond updating doc cross-references. Wiring `/api/signals` to read the artifact + generating the TS type stays a later bridge task, gated as in the existing roadmap.
- Building the news model (that's Task 2).

---

## Claude Code kickoff prompt (Task 1)

```
Read CLAUDE.md, then docs/adr/0001-multi-model-repo-architecture.md
(including its "The artifact contract (schema v2)" section), and
docs/plans/task-1-repo-restructure.md.

Execute Task 1: restructure this repo from the single-package lidr-ml into the
lidr-models research monorepo (lidr_core shared harness + ta_ensemble package +
empty news_sentiment shell), and formalize the artifact contract as a validated
schema_version: 2 JSON Schema. This is a mechanical, no-behavior-change refactor —
follow the migration map exactly, use `git mv` to preserve history, and do NOT
change any signal/model/backtest logic.

The acceptance gate is PARITY: baseline_six_signals_unweighted.yaml must reproduce
its pre-move results_log row (skill_score, cagr, n_oos) to full precision. Capture
before/after numbers in the PR. Follow the protected-main PR workflow; CI green is
required. Do not start Task 2.
```
