# docs/ — documentation index

Orientation for everything under `docs/`. Two kinds of docs live here:

- **Durable** — kept long-term; the record of *why* the project is shaped the way it is.
- **Disposable** — task plans that self-delete when their task merges (the disposable-plan-doc convention; see [`../CLAUDE.md`](../CLAUDE.md)).

| Doc | Kind | What it is |
| --- | --- | --- |
| [`adr/0001-multi-model-repo-architecture.md`](adr/0001-multi-model-repo-architecture.md) | durable | Why the repo is a `lidr_core` + per-model monorepo, and the schema-v2 artifact-contract design. Accepted, implemented in Task 1. |
| [`research/data-sources.md`](research/data-sources.md) | durable | News / sentiment data-source comparison + the point-in-time caveat. Drives the `news_sentiment` data layer. |
| [`signals.md`](signals.md) | durable | First-time-reader explainer for the six TA signals, with charts on real SPY data (PNGs in `signals/`). |
| [`sample-report/report.html`](sample-report/report.html) | durable | A committed sample of the generated HTML backtest report (the SPY baseline run). |
| [`reports/`](reports/README.md) | durable | Per-major-version backtesting report archive + the procedure every backtest follows going forward. Self-contained, reproducible, immutable folders. |
| [`plans/`](plans/) | disposable | Task plans handed to an AI assistant; each self-deletes on its task's merge. Currently: `task-2-news-sentiment-model.md`, `pr-c-news-v0-backtest.md` (both delete on PR-C merge). |

## Start here

New to the project? Read in this order:

1. [`../README.md`](../README.md) — what the project does, how to run it, the [pipeline walkthrough](../README.md#how-the-pipeline-works), and [current status](../README.md#current-status-at-a-glance).
2. [`adr/0001-multi-model-repo-architecture.md`](adr/0001-multi-model-repo-architecture.md) — why the repo is shaped this way.
3. [`signals.md`](signals.md) — what the six TA signals measure.
4. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — the rules for landing a change.

[`../CLAUDE.md`](../CLAUDE.md) is the deep, AI-facing playbook (full history, conventions, gotchas, roadmap). It's the most complete doc but also the longest — humans usually want the README + this index first.
