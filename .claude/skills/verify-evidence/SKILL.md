---
name: verify-evidence
description: >-
  Pressure-test and then document model-performance conclusions before committing them in
  lidr-models. Use this in TWO situations. (1) Whenever you are about to conclude something
  about a model's performance — that a new model / feature / target BEATS the benchmark, OR
  that it has NO edge / is worse — run the Stage 1 self-checks that keep you from shipping a
  false result. (2) Whenever you open or write the description for an outcome-changing PR
  (signal ports, model/learner changes, scoring or calibration changes, target/feature
  reformulations, JSON-artifact schema changes) run Stage 2 to generate the reviewer-facing
  evidence. Trigger it on phrases like "the model beats buy-and-hold", "it has an edge",
  "no skill / no edge", "is this result real", verification evidence, an embedded
  before/after chart, a dated sanity-check table, a parity number, the `docs/_pr_evidence/`
  directory, or a `scripts/verify_*.py` script. It produces
  `docs/_pr_evidence/<thing>/{chart.png, evidence.md}` recomputed from the prediction
  artifacts and gated by PASS/FAIL gut checks, then reminds you to pin the chart URL to a
  full 40-char commit SHA and delete the evidence dir in a cleanup commit before
  squash-merge. Do NOT use for refactors, doc-only changes, or infra/CI PRs.
---

# verify-evidence

Two things must be true before a model result lands here: the conclusion has to be
**real** (not an artifact of leakage, a misread probability column, or a lucky config),
and the PR has to carry **trustworthy evidence** a reviewer can check without re-running
anything. Those are two stages of one job — establish the truth, then prove it — and this
skill covers both, because each stage has bitten us in the same few ways.

- **Stage 1 — pressure-test the conclusion.** Run when the PR (or the writeup you're about
  to draft) makes a claim about model performance, in *either* direction.
- **Stage 2 — produce the evidence artifact.** Run when packaging any outcome-changing PR.

A typical model PR does both. A pure signal-port or schema-only PR skips Stage 1 (no
performance claim to validate) and does Stage 2. Refactors, doc-only, and infra/CI PRs do
neither — a parity test, smoke run, or green CI is the right evidence there; don't
manufacture a chart.

---

## Stage 1 — Pressure-test the conclusion (don't fool yourself)

A model conclusion is easy to get wrong in ways that look fine on the surface. Run the
checks below *before* you write the "reading this" narrative. The direction of the claim
changes which failure mode is most likely, so the list splits accordingly.

### Universal — run for any model conclusion, either direction

- **Column-order spot check.** Confirm `classes_ == [0, 1]` and that in-sample
  P(class=1) is higher on up-days than down-days. Five lines; rules out silently reading
  P(down) as P(up), which inverts *every* conclusion. Trivially passes for sklearn-conforming
  models, but it's cheap insurance against the one bug that flips everything.
- **In-sample fit check.** Training-set accuracy must be meaningfully above the base rate.
  If it isn't, the model isn't fitting at all — and an out-of-sample collapse is a fit
  failure, not a generalization failure, which is a completely different story to tell.
- **Headline `skill_score`, not accuracy.** Report `base_rate` and `pred_rate` beside
  accuracy. Accuracy 0.557 looks fine until you notice the base rate is 0.613 — predicting
  "up" every day scores 0.613. `skill_score` (= 1 − log_loss/base_logloss) normalizes to the
  base rate, so it's the only metric comparable across configs whose base rates differ.
- **Generate first, write second.** Never draft the interpretation from a remembered prior
  before the numbers exist — doing so has contradicted the actual numbers more than once
  here. Produce the table, then write from it. When the result is surprising or contradicts a
  prior, dig: build in one "what would convince me this is wrong?" pass before declaring done.
  The same discipline covers *operational* facts — article/row counts, commit SHAs, token
  counts, $ spend, test totals — in any PR, comment, or commit: transcribe each from same-turn
  tool output, never recall it from memory (CLAUDE.md Conventions → "Transcribe figures").

### If you're claiming the model is BETTER (has an edge / beats the benchmark)

This is the **dangerous direction** — it's the result you *want*, so confirmation bias and
subtle leakage are exactly what manufacture a false positive. Be your own harshest skeptic:

- **Leakage / lookahead.** Is the "edge" coming from peeking at the future? Confirm the
  features are lookahead-safe and the backtest is expanding-window walk-forward (never
  k-fold). This is the first suspect for any positive result.
- **Exposure vs. skill.** Is the higher CAGR just *more market exposure*, not skill? A
  high-base-rate, near-always-long config (`pred_rate` ≈ 1) tracks buy-and-hold, and the
  equity curve marks every position to market on 1-day-forward returns — so its "excess" is
  exposure, not edge. A "better" claim must show a higher **skill_score**, not just a fatter
  equity curve. (A prior +0.20pp CAGR "excess" here turned out to be ~15 lucky cash days.)
- **Costs modeled.** Transaction costs must be non-zero (5 bps default). A positive result
  with zero or under-modeled costs is suspect by construction.
- **Within noise?** Is the excess a handful of lucky days, or does it hold across years /
  sub-periods? Look at the per-year table, not just the aggregate.
- **Cherry-picking / robustness.** Is this one lucky config out of a sweep? Does it survive
  different seeds, hyperparameters, and sub-periods? If you ran many configs, the best one is
  expected to look good by chance — say how many you tried.

### If you're claiming the model has NO edge (or is worse)

Here the risk is the opposite: a **broken pipeline masking real signal**, so you wrongly
declare defeat. Rule that out:

- **Hyperparameter sensitivity.** For a low-SNR problem, expect monotonic behavior in
  capacity: tiny config → no-skill floor; default → confidently wrong; large → more
  confidently wrong. Non-monotonic behavior ⇒ the result is hyperparameter noise, not a real
  read on the signal.
- **Calibration wrapper.** For any tree ensemble emitting `predict_proba`, run with and
  without `CalibratedClassifierCV`. A large delta (LightGBM moved skill_score −0.148 → −0.004)
  means the probabilities are *miscalibrated*, not anti-informative — a very different
  conclusion. Don't conflate "the model fits noise" with "the model is miscalibrated."
- **Seed stability.** Only informative once subsampling is enabled (`feature_fraction` /
  `bagging_fraction` < 1). With no randomness consuming `random_state`, all seeds are
  bit-identical and the sweep proves nothing — the hyperparameter sweep above is the stronger
  robustness check.

Once the conclusion survives Stage 1, write it up — leading with the plain-English result —
and move to Stage 2 to produce the evidence.

---

## Stage 2 — Produce the reviewer-facing evidence

> **Destination depends on whether this PR ships a major model version.**
> - **Major version** (a new model or a version bump that becomes part of the record): the evidence is *promoted into a durable, self-contained report folder* `docs/reports/<YYYY-MM-DD>-<model_id>-v<X.Y>/` and **kept on `main`**; the PR description **points to that folder** instead of inlining the analysis. Follow [`docs/reports/README.md`](../../../docs/reports/README.md) for the folder convention (naming, contents, `report.md`/`REPRODUCE.md` skeletons, immutability, index). The transient `docs/_pr_evidence/` steps below do **not** apply — do not delete the durable folder.
> - **Transient experiment** (sweep, signal port, schema-only PR, robustness probe — nothing that ships a version): use the `docs/_pr_evidence/<thing>/` flow below and delete it in the cleanup commit before squash-merge.
>
> Either way, Stage 1 and the recompute-from-artifact mechanics in this section are identical — only the destination and lifecycle differ. This split is mirrored in `docs/reports/README.md` so the two docs stay consistent: this skill owns *how to validate and generate the evidence*; that file owns *where the durable record lives and how it's packaged*.

Outcome-changing PRs need an embedded before/after chart plus a dated sanity-check table
from **real** data (SPY, not synthetic) in the description, so a non-technical reviewer can
approve without re-running anything. The procedure below is the transient `_pr_evidence` flow;
for a major version, produce the same artifacts but publish them per `docs/reports/README.md`:

1. **Pick a `<thing>` slug** for this PR (e.g. `rsi_port`, `calibration`, `regime_features`).
   Everything lands under `docs/_pr_evidence/<thing>/`.
2. **Copy the scaffold** `scripts/verify_evidence_template.py` (bundled with this skill) to
   `scripts/verify_<thing>.py`, then fill in the `# EDIT:` markers — the configs to include,
   the parity anchors, what the chart plots. Keep the recompute-from-artifact and gut-check
   skeleton intact; that's the part that's earned its place.
3. **Run it** — `python scripts/verify_<thing>.py`. It writes
   `docs/_pr_evidence/<thing>/{chart.png, evidence.md}` and exits non-zero if any gut check
   fails. Don't write the chart narrative until it passes (see "generate first, write second").
4. **Commit the script + evidence dir to the branch** so the chart is reachable at a raw URL.
5. **In the PR description**, embed the chart by its raw URL pinned to the full 40-char commit
   SHA, and paste the sanity table from `evidence.md`. Lead with the plain-English result.
6. **Before squash-merge, delete the evidence dir and the verify script** in a final cleanup
   commit so `main` stays free of review artifacts. The chart URL still resolves because it's
   pinned to the SHA, not the branch.

### Why each Stage 2 guardrail exists

- **Recompute every number from the prediction-artifact JSON, not the pipeline's logged
  output.** Read `y_true` / `probability_up` / `y_pred` from the artifact and redo the metrics
  yourself. A verify script that re-derives metrics is itself unverified code: an early draft
  read `prices["Close"]`, but the loader lowercases columns, so the fallback silently grabbed
  `open` and drew an equity curve ending at 0.6× while results_log said 2.47×.
- **Cross-check the recomputed `skill_score` against the `results_log.csv` row within a small
  tolerance (≈5e-4), and fail loudly on drift.** This is the single check that catches a broken
  verify script. The scaffold does it as gut check #2.
- **For a port, anchor parity to the prior implementation** — cite a max-absolute-difference
  vs lidr's TS signal (or vs the committed `results_log` row for the source config).
- **Report `base_rate` / `pred_rate` beside `accuracy`, headline `skill_score`** (same reason
  as Stage 1 — accuracy alone misleads when the base rate is far from 0.5).
- **Two charts when the data spans years.** A full-window log-scale equity curve compresses the
  last year into a few pixels; add a recent-window linear zoom so recent behavior stays
  legible.
- **Pin the embedded chart URL to the full 40-char SHA from `git rev-parse <short>`, pasted
  verbatim.** The `raw.githubusercontent.com` URL is exact-match or 404 — never type SHA hex
  from memory or hand-extend a short SHA. A PR once shipped with a 404'd chart for exactly this.
- **Live-verification evidence uses the same mechanics as the chart.** API responses, LLM
  smoke output, and spend logs that back a verification claim get committed under
  `docs/_pr_evidence/<thing>/`, linked in the PR pinned to a full 40-char SHA, and removed in
  the cleanup commit before squash-merge — exactly like the chart/table evidence above. A
  "PASS" claim doesn't merge without that linked, openable artifact (see CLAUDE.md Conventions
  → "Transcribe figures": PASS requires an inspectable artifact to already exist).

### The script scaffold

`scripts/verify_evidence_template.py` is a runnable starting point adapted from the real
horizon-sweep verifier. It already wires up: loading the latest artifact per config from
`artifacts/predictions/<model_id>/`; recomputing `skill_score`, `accuracy`, `base_rate`,
`pred_rate`, `log_loss`, `n_oos` from the artifact predictions; the cross-check against
`results_log.csv` and the parity anchors; per-year strategy returns reconstructed from cached
prices via `lidr_core`'s own `add_strategy_returns` / `performance_by_year` (so the equity
table is independent of the run that produced the artifact); a chart and a Markdown
`evidence.md` with a PASS/FAIL gut-check block; and a non-zero exit on any failure. Fill in the
`# EDIT:` markers and delete what you don't need. Read it top-to-bottom once before adapting —
the docstring explains what each gut check guards.
