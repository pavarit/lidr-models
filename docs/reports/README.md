# docs/reports/ — backtesting report archive

Durable, per-major-version backtesting reports. Every **major model version** (the kind that lands via an outcome-changing PR) gets one **immutable folder** here containing everything needed to *read* and *reproduce* its backtest.

This is the human-facing companion to `artifacts/manifest.json` (the machine leaderboard). The manifest answers *"which artifact is best right now"*; this archive answers *"what did we run, why, what happened, and how do I reproduce it exactly."*

## Why this exists / how it's used going forward

`reports/<config>-<timestamp>/` (the auto-generated HTML) and `artifacts/predictions/*` (the JSON artifacts) are both **gitignored** — they're per-machine build outputs that don't survive on `main`. So a backtest's evidence used to live only in the PR description and then vanish. From now on, **every major model version publishes a self-contained folder here**, and the PR points to the folder instead of inlining the analysis. A folder must contain everything a future reader needs — nothing it relies on may be gitignored elsewhere.

## Relationship to the `verify-evidence` skill

This procedure **extends** the skill, it doesn't replace it.

- **Stage 1 (pressure-test the conclusion)** still runs in full before any verdict is written — column-order check, in-sample fit, headline `skill_score` (not accuracy), leakage/exposure checks, generate-first-write-second. The report's "pressure-test checklist" section records the outcome.
- **The recompute-from-artifact mechanics** still apply: every number in the report is recomputed from the prediction-artifact JSON and cross-checked against the `results_log.csv` row within tolerance (≈5e-4). Use `scripts/verify_<thing>.py` (the skill's scaffold) to produce the chart + table.
- **What changes is Stage 2's destination.** For a **major version**, the chart/table/evidence are *promoted into this durable folder and kept*, and the PR links here — instead of going to `docs/_pr_evidence/<thing>/` and being deleted before squash-merge. The transient `_pr_evidence` flow stays the right tool for **experiments that don't ship a version**: sweeps, signal ports, schema-only PRs, robustness probes.

This split is stated from the skill's side too (`verify-evidence` SKILL.md, Stage 2) so the two docs can't drift apart: the skill owns *how to validate a conclusion and generate the evidence*; this file owns *where the durable record lives and how it's packaged*.

> Rule of thumb: *Will someone want to reproduce this run in six months?* Yes → durable folder here. No (it's a one-off probe) → transient `_pr_evidence`, deleted on merge.

## Folder naming

```
docs/reports/<YYYY-MM-DD>-<model_id>-v<major.minor>/
```

**Date first** so the folders sort chronologically — a plain `ls` shows the order of development at a glance. `<YYYY-MM-DD>` is the **merge date of the PR** that produced the version. Examples: `2026-06-01-news_sentiment-v0/`, `2026-06-15-ta_ensemble-v1/`.

## Folder contents

Everything self-contained. A reader with only this folder + the repo at the pinned commit can reproduce the analysis.

| File / dir | What it is | Why it's here |
| --- | --- | --- |
| `report.md` | The narrative report (skeleton below). The primary deliverable. | The analysis itself. |
| `chart.png` (+ `chart_recent.png`) | Comparison equity curves / skill chart. Add a recent-window linear zoom when the data spans years (per the skill — a full-window log curve hides the last year). | Reviewer-facing visual evidence. |
| `configs/` | **Exact copy** of every config used — the model config(s) **and** the comparison-baseline config(s). | `packages/*/configs/*.yaml` may change after the run; the copy freezes what actually ran. |
| `artifacts/` | **Frozen copy** of the prediction JSON(s) produced by the run, plus the matching `results_log.csv` row(s). | `artifacts/predictions/*` is gitignored — without a copy the folder can't be re-verified from the artifact. |
| `report.html` *(optional)* | The generated HTML report(s) copied out of the ephemeral `reports/`. | `reports/*` is gitignored; keep it if the rendered view adds value. |
| `REPRODUCE.md` | Exact reproduction recipe (skeleton below). | "How to reproduce" — PR, commit, commands, environment, data provenance, seeds, cost. |
| `env.lock` *(or `requirements-freeze.txt`)* | `pip freeze` (or equivalent) snapshot at run time. | ML results move with `scikit-learn` / `lightgbm` / `transformers` / `torch` versions; pin them. |

## `report.md` skeleton

```markdown
# <model_id> v<X.Y> backtest — <one-line title>

**Verdict:** <edge / no edge / inconclusive>, <one sentence>. **Gates:** <what this
result unblocks or keeps parked — e.g. "edge gate stays closed; calibration deferred">.
**PR:** #NN · **Commit:** <full 40-char SHA> · **Run date:** YYYY-MM-DD

## 1. Intent & hypothesis
What question this run answers and the hypothesis being tested.

## 2. Methodology
Model/learner + params, target (type, horizon, threshold), backtest method
(expanding-window walk-forward, fold params), transaction costs, run topology
(e.g. per-ticker loop).

## 3. Inputs & outputs
Feature set (each feature + the source stream it draws from); the universe;
the artifact(s) and results_log row(s) produced.

## 4. Data sources & why
Each source, its role, its usable history window, cost. Why sources were
included or excluded (e.g. "Finnhub excluded — 1yr history too short for a
multi-year window"). The point-in-time / backfill caveat and how it was checked.

## 5. Key choices & rationale
The decisions that shaped the run and why each was made. Link the signed-off
decision record if one exists.

## 6. Outcomes analysis
Headline **skill_score** (not accuracy) with `base_rate` / `pred_rate` / `n_obs`
beside it; per-year / sub-period table; the comparison (this model vs the
relevant benchmark vs buy-and-hold). Reference chart.png. Read the result in
plain English first, then the numbers.

## 7. Limitations & caveats
Survivorship bias, small-sample (`n_obs`), backfill/point-in-time risk,
exposure-vs-skill, anything that bounds how far the result generalizes.

## 8. Stage-1 pressure-test checklist
Which verify-evidence Stage-1 checks were run and their PASS/FAIL outcome
(column-order, in-sample fit, leakage/lookahead, exposure-vs-skill, costs
modeled, within-noise, cherry-picking). Cross-check vs results_log: PASS/FAIL.

## 9. Reproduction
Pointer to REPRODUCE.md.
```

## `REPRODUCE.md` skeleton

```markdown
# Reproduce: <model_id> v<X.Y>

- **PR:** #NN — <url>
- **Commit:** <full 40-char SHA> (from `git rev-parse HEAD`, pasted verbatim)
- **Python:** <version> · **OS:** <where it ran>
- **Install:** `make install` · environment frozen in `env.lock`

## Commands
Exact commands, in order. E.g.:
    make backtest-news-v0          # runs all configs in configs/
    python scripts/verify_<thing>.py   # regenerates chart.png + the metrics table

## Data provenance
- Prices: yfinance pull date <date> (survivorship snapshot — yfinance only has
  currently-listed tickers as of this date).
- News cache: sources <…>, window <start>–<end>, pulled <date>. Cheap-tier
  timestamps aren't guaranteed point-in-time — validated by <spot-check>.
- Any API/data caveats that would change a re-run.

## Determinism
Seeds; note any non-deterministic component (e.g. LLM scoring is not
reproducible bit-for-bit — record the cache that pins it).

## Cost ledger
EODHD API calls (5/request), LLM $ spend, etc. — so cost-per-iteration is tracked.

## Expected outputs
Artifact filenames + results_log run_ids a successful re-run should produce,
so the reproducer can confirm a match.
```

## Immutability

A published report folder is **immutable**. Found an error? Append a dated `## Erratum` to `report.md` (quote the corrected figure from a tool run, per the transcribe-don't-recall convention) or supersede it with a new version folder — never silently rewrite a shipped report.

## Index

Newest first. One row per published version.

| Version | Date | Model | Verdict | Report |
| --- | --- | --- | --- | --- |
| _none yet — `news_sentiment-v0` lands with PR-C_ | | | | |
