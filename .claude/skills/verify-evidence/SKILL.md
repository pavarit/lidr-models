---
name: verify-evidence
description: >-
  Scaffold and run the verification-evidence workflow for outcome-changing PRs in
  lidr-models — PRs that change what the pipeline OUTPUTS: signal ports, model/learner
  changes, scoring or calibration changes, target/feature reformulations, and
  JSON-artifact schema changes. Use this whenever you are about to open (or write the
  description for) such a PR, or whenever someone mentions verification evidence, an
  embedded before/after chart, a dated sanity-check table, a parity number, the
  `docs/_pr_evidence/` directory, or a `scripts/verify_*.py` script. It produces
  `docs/_pr_evidence/<thing>/{chart.png, evidence.md}` recomputed from the prediction
  artifacts (never the logged numbers) and gated by PASS/FAIL gut checks, then reminds
  you to pin the chart URL to a full 40-char commit SHA and delete the evidence dir in a
  cleanup commit before squash-merge. Do NOT use for refactors, doc-only changes, or
  infra/CI PRs — those don't need evidence.
---

# verify-evidence

Outcome-changing PRs in this repo need verification evidence in the description: an
embedded before/after chart plus a dated sanity-check table from **real** data (SPY,
not synthetic), so a non-technical reviewer can approve without re-running anything.
This skill turns that convention into a repeatable procedure with a script scaffold,
because every time we've done it by hand it has bitten us in the same few ways.

## When this applies

Use it when the PR changes **what the pipeline outputs**:

- a signal port (Python signal mirroring lidr's TS) — also needs a max-abs-diff parity number
- a model/learner change (new model class, hyperparameters, calibration wrapper)
- a scoring or calibration change
- a target or feature reformulation (e.g. horizon sweep, magnitude-regression target, regime features)
- a JSON-artifact schema change

**Skip it** for refactors, doc-only changes, and infra/CI PRs. Those don't change
outputs, so a parity test / smoke run / green CI is the right evidence instead — don't
manufacture a chart for them.

## The workflow

1. **Pick a `<thing>` slug** for this PR (e.g. `horizon_sweep`, `rsi_port`, `calibration`).
   Everything lands under `docs/_pr_evidence/<thing>/`.
2. **Copy the scaffold** `scripts/verify_evidence_template.py` (bundled with this skill)
   to `scripts/verify_<thing>.py` in the repo, then fill in the `# EDIT:` markers — the
   configs to include, the parity anchors, and what the chart plots. Keep the
   recompute-from-artifact and gut-check skeleton intact; that's the part that's earned
   its place.
3. **Run it** — `python scripts/verify_<thing>.py`. It writes
   `docs/_pr_evidence/<thing>/{chart.png, evidence.md}` and exits non-zero if any gut
   check fails. Do not write the PR narrative until this passes (see "Generate first,
   write second" below).
4. **Commit the script + evidence dir to the branch** so the chart is reachable at a raw
   GitHub URL.
5. **In the PR description**, embed the chart by its raw URL pinned to the full 40-char
   commit SHA, and paste the sanity table from `evidence.md`. Lead with the plain-English
   result, then the evidence.
6. **Before squash-merge, delete the evidence dir and the verify script** in a final
   cleanup commit so `main` stays free of review artifacts. The chart URL still resolves
   because it's pinned to the SHA, not the branch.

## Why each guardrail exists

These aren't ceremony — each one is a bug that actually shipped here.

- **Recompute every number from the prediction-artifact JSON, not the pipeline's logged
  output.** Read `y_true` / `probability_up` / `y_pred` out of the artifact and redo the
  metrics yourself. A verify script that re-derives metrics is itself unverified code: an
  early draft read `prices["Close"]`, but the loader lowercases columns, so the fallback
  silently grabbed `open` and drew an equity curve ending at 0.6× while results_log said
  2.47×. Recomputing from the artifact, then cross-checking, catches that.
- **Cross-check the recomputed `skill_score` against the `results_log.csv` row within a
  small tolerance (≈5e-4), and fail loudly if it drifts.** This is the single check that
  catches a broken verify script. The scaffold does it as gut check #2.
- **For a port, anchor parity to the prior implementation.** Cite a max-absolute-difference
  vs lidr's TS signal (or vs the committed `results_log` row for the source config). The
  scaffold has a `PARITY` dict for exactly this.
- **Report `base_rate` and `pred_rate` beside `accuracy`, and headline `skill_score`, not
  accuracy.** Accuracy 0.557 looks fine until you notice the base rate is 0.613 — predicting
  "up" every day scores 0.613. `skill_score` (= 1 − log_loss/base_logloss) normalizes to the
  base rate, so it's the only metric comparable across configs whose base rates differ.
- **Two charts when the data spans years.** A full-window log-scale equity curve compresses
  the last year into a few pixels; add a recent-window linear zoom so recent behavior stays
  legible. (For non-equity charts, the same spirit applies — make the regime that matters
  readable.)
- **Pin the embedded chart URL to the full 40-char SHA from `git rev-parse <short>`, pasted
  verbatim.** The `raw.githubusercontent.com` URL is exact-match or 404 — never type SHA hex
  from memory or hand-extend a short SHA. A PR once shipped with a 404'd chart for exactly
  this reason.

## Generate first, write second

Never draft the "reading this chart" narrative from memory before the numbers exist.
Drafting interpretation from a remembered prior has contradicted the actual numbers more
than once here. Run the script, read `evidence.md`, then write the PR description from the
table in front of you. When the result is surprising or contradicts a prior, dig — build in
one "what would convince me this is wrong?" pass before declaring it done.

## The script scaffold

`scripts/verify_evidence_template.py` is a runnable starting point adapted from the real
horizon-sweep verifier. It already wires up:

- loading the latest artifact per config from `artifacts/predictions/<model_id>/`
- recomputing `skill_score`, `accuracy`, `base_rate`, `pred_rate`, `log_loss`, `n_oos`
  from the artifact predictions
- the cross-check against `results_log.csv` and the parity anchors
- per-year strategy returns reconstructed from cached prices via `lidr_core`'s own
  `add_strategy_returns` / `performance_by_year` (so the equity table is independent of
  the run that produced the artifact)
- a two-panel chart and a Markdown `evidence.md` with a PASS/FAIL gut-check block
- a non-zero exit on any failure

Fill in the `# EDIT:` markers and delete the parts you don't need. Read it top-to-bottom
once before adapting — the docstring explains what each gut check guards.
