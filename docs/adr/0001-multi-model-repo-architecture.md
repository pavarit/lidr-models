# ADR 0001 — Multi-model repository architecture

- **Status:** Proposed (planning only — no code moved yet)
- **Date:** 2026-05-27
- **Deciders:** Boon
- **Supersedes:** the implicit "one model = one repo (`lidr-ml`)" assumption

## Context

The project shape has become clear. There are three logical pieces:

1. **`lidr`** — the Next.js front-end. Displays BUY / HOLD / SELL recommendations. Already its own repo, deliberately separate from Python (Node + Python in one tree was rejected — see `lidr`'s CLAUDE.md Key Decisions).
2. **One or more competing models** — each ingests data, builds features, trains, backtests, and emits a recommendation. Today there is exactly one (`lidr-ml`: the six technical-analysis signals + logistic / LightGBM ensemble). A **news-sentiment model** is coming, and others may follow.
3. **The artifact contract** — the versioned JSON file a model produces and `lidr` consumes. Today it is `schema_version: 1`, defined implicitly inside `lidr-ml`'s `pipeline.py`.

The current single-model layout will not age well once models *compete*. Two failure modes are predictable:

- **Schema drift.** With the contract defined implicitly per model, each new model invents slightly different fields and `lidr` accumulates special-case handling. The contract rots.
- **Eval drift.** The backtest engine, walk-forward CV, metrics, and `results_log` are the "rules of the competition." If each model repo keeps its own copy, the copies diverge and model-vs-model comparisons quietly become invalid — which is fatal, because the project's entire near-term goal is comparing models to prove an edge.

## Decision

Reorganize around **the contract**, not around the models. Concretely:

### 1. Rename `lidr-ml` → `lidr-models` and make it a research monorepo

`lidr-ml` reads as "the ML model," which is confusing once several exist. `lidr-models` signals "the place where models compete." Inside it, split the **shared harness** from the **per-model code**:

```
lidr-models/                       (was lidr-ml)
  packages/
    lidr_core/                     THE SHARED HARNESS — owned once, reused by every model
      backtest/                    expanding-window walk-forward engine + strategy returns
      eval/                        metrics, skill_score, results_log, cross-model leaderboard
      contract/                    artifact JSON Schema + writer (validates on write) + loader
      protocols/                   Signal, Feature, Model, DataSource interfaces
      data/                        base loaders (yfinance, on-disk cache)
    ta_ensemble/                   the six TA signals + configs — "lidr-ml as it is today"
      signals/  configs/  pyproject.toml   (depends on lidr_core)
    news_sentiment/                the new model (built in Task 2)
      datasources/ ingest/ scoring/ features/ configs/  pyproject.toml  (depends on lidr_core)
  artifacts/
    manifest.json                  leaderboard: every model + its latest artifact + OOS skill
    predictions/<model_id>/...     each model writes here, all to the same schema
```

`lidr` stays its own Node repo, unchanged in spirit. It gains only a generated TypeScript type for the artifact and validation on read (future bridge work, not now).

### 2. The split that matters

> **Shared (lives in `lidr_core`):** backtest engine, eval/metrics, `results_log`, the artifact writer + schema, the protocol interfaces, base data loaders.
> **Per-model (lives in each package):** data ingestion, features, the model itself, and configs.

This is what lets a new model inherit the entire battle-tested evaluation stack for free, and guarantees every model is scored by the *same* harness against the *same* buy-and-hold benchmark.

### 3. Promote the artifact to a formal, versioned, validated contract

Write today's implicit schema down as a JSON Schema in `lidr_core/contract`, evolve it to `schema_version: 2` to carry `model_id` / `model_version` and a `manifest.json` leaderboard, validate on write (model side) and on read (`lidr` side), and generate `lidr`'s TS type from it. Full design in [The artifact contract](#the-artifact-contract-schema-v2) below.

## The artifact contract (schema v2)

The JSON file a model produces and `lidr` consumes is the *contract* that decouples the two. Today it is `schema_version: 1`, defined implicitly by `pipeline.py`. A **JSON Schema** is a language-agnostic description of "what a valid artifact looks like" — the same idea as a Pydantic model, but readable by both Python (producer) and TypeScript (consumer). Formalizing it buys three things: a model **cannot emit a malformed artifact** (validate on write), `lidr` **fails loudly instead of rendering garbage** on a bad file (validate on read), and `lidr`'s TS type is **generated from** the schema so the two sides can't drift.

**v1 → v2 changes.** v1 carried `config_name`, `ticker`, `generated_at`, `metrics.{classification,strategy,benchmark}`, `predictions[].{date, y_true, y_pred, probability_up}`. v2 adds the identity + discovery fields needed once more than one model produces artifacts:

- `model_id` — stable id for the producing model (`"ta_ensemble"`, `"news_sentiment"`).
- `model_version` — version of that model's logic/config, so two runs are distinguishable.
- `schema_version` — bumped to `2`.
- a top-level `manifest.json` (separate file) — the leaderboard `lidr` reads to discover which models exist and how each scores.
- `recommendation` on each prediction — the human-facing BUY/HOLD/SELL label `lidr` displays (derived from `probability_up` bands until the 3-class migration; the field exists now so the schema is stable when 3-class arrives).

`config_name` is kept (the specific experiment) alongside `model_id` (the model family). `y_true` must allow `null` for not-yet-resolved recent dates.

**Example artifact** (`predictions/<model_id>/<config>-<timestamp>.json`):

```json
{
  "schema_version": 2,
  "model_id": "ta_ensemble",
  "model_version": "2.1.0",
  "config_name": "baseline_six_signals_unweighted",
  "ticker": "SPY",
  "generated_at": "2026-05-27T22:00:00Z",
  "metrics": {
    "classification": { "accuracy": 0.610, "base_rate": 0.611, "skill_score": -0.005, "log_loss": 0.671, "base_logloss": 0.668, "n_obs": 3851 },
    "strategy":  { "cagr": 0.1425, "sharpe": 0.71, "max_drawdown": -0.337, "final_equity": 7.9 },
    "benchmark": { "cagr": 0.1404, "sharpe": 0.89, "max_drawdown": -0.337, "final_equity": 8.2 }
  },
  "predictions": [
    { "date": "2026-05-26", "recommendation": "HOLD", "probability_up": 0.58, "y_true": null, "y_pred": 1 }
  ]
}
```

**Example `manifest.json`** (the leaderboard — what makes "competing models feeding `lidr`" concrete):

```json
{
  "schema_version": 2,
  "generated_at": "2026-05-27T22:05:00Z",
  "models": [
    { "model_id": "ta_ensemble",   "model_version": "2.1.0", "latest_artifact": "predictions/ta_ensemble/baseline_six_signals_unweighted-20260527-220000.json", "oos_skill_score": -0.005, "beats_buy_and_hold": false },
    { "model_id": "news_sentiment", "model_version": "0.1.0", "latest_artifact": "predictions/news_sentiment/dev-20260527-203000.json",                         "oos_skill_score": null,   "beats_buy_and_hold": null }
  ]
}
```

`lidr` reads `manifest.json`, then each model's latest artifact, and can render "Model A: HOLD (skill −0.01) · Model B: BUY (skill +0.03)".

**Versioning rule.** Bump `schema_version` only on a *breaking* change (removing/renaming a field, changing a type). Additive optional fields don't need a bump. `lidr` validates the major version it understands and rejects unknown majors loudly. Mirrors the existing "new comparison columns get appended to the right" discipline.

**Open questions (decide when relevant, not now):** whether to expose per-signal/feature contributions for `lidr` explainability (lean: defer to an additive v2.x field when `lidr` wants it); and once 3-class lands, whether `lidr` reads `recommendation` directly or re-derives bands client-side.

## Designed for change

These are explicit requirements, not nice-to-haves. Each maps to a structural choice:

| Anticipated change | Structural answer |
|---|---|
| **Data sources will be added / swapped** | A `DataSource` protocol in `lidr_core/protocols`; each source (e.g. EDGAR, GDELT, Finnhub, Apewisdom, EODHD) is a pluggable adapter under a model's `datasources/`. Configs select which sources are active. Adding a source = one new adapter file + one config line; re-training is a config change, not a code change. The 2026-05-28 data-source revision (see `docs/research/data-sources.md`) is the protocol's first real stress test — swapping Reddit / Tiingo / pytrends out for Finnhub / Apewisdom / EODHD is a config + adapter swap, not a model rewrite. |
| **Models will iterate many times** | Every backtest run already appends a row to `results_log.csv` with a `run_id`. Extend it with `model_id` + `model_version` and surface a cross-model **leaderboard** so "did this version beat the last one, and does it beat the other model?" is answerable without opening reports. Builds directly on what `lidr-ml` already has. |
| **Features and the model will change** | Keep the existing pluggable `Signal` / `Model` protocols; add a parallel `Feature` registry for the news model. A model is assembled from a config that names its sources, features, scorer, and learner — so swapping any layer is a config edit. |

## Consequences

**Positive**
- The second model is cheap: `news_sentiment` starts as a shell that imports `lidr_core` and only implements ingestion + features.
- Model comparisons are valid by construction (one harness, one benchmark).
- The contract can't silently drift; both sides validate against one schema.
- The data-source and feature/model layers are swappable via config, matching the stated flexibility goals.

**Negative / costs**
- A one-time restructuring effort (Task 1) — mostly file moves, low logic risk, but it touches every import.
- The repo rename costs a GitHub URL change and cross-reference updates in both CLAUDE.md files.
- Slightly more indirection (a model now depends on a `lidr_core` package) — accepted, because it's what prevents eval drift.

## Alternatives considered

- **One repo per model (polyrepo).** Rejected as the primary structure: it forces the shared eval harness to be copied (→ drift), and every schema change becomes N coordinated PRs — heavy for a solo developer. Dependency isolation, its main advantage, is handled inside a monorepo by workspace tooling (`uv` workspaces / per-package extras) so the news model can pull `transformers`/FinBERT without burdening the lean TA model.
- **Full monorepo (lidr + models together).** Rejected — reverses the deliberate Node/Python separation for good reasons (tooling, deploy cadence). Note: the chosen design does **not** reverse that decision; `lidr` stays separate. This ADR only governs the *Python* side's internal shape.
- **Leave the contract implicit.** Rejected — it's the single highest-leverage thing to formalize before a second producer exists.

## Execution

Split into two handoff tasks for Claude Code:

- **Task 1 — restructure** (mechanical, no behavior change): [`../plans/task-1-repo-restructure.md`](../plans/task-1-repo-restructure.md)
- **Task 2 — build the news-sentiment model** (new logic): [`../plans/task-2-news-sentiment-model.md`](../plans/task-2-news-sentiment-model.md)

Task 2 is blocked by Task 1.
