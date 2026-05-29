# Horizon spike — TA model target-horizon sweep

**Disposable plan doc.** Delete this file in the cleanup commit before squash-merge of the horizon-spike PR, per the disposable-plan-doc convention (mirrors how `docs/plans/task-1-repo-restructure.md` was handled — see Recent Changes for the precedent). The durable record of the finding goes in CLAUDE.md's Active Task section (which horizon won, by how much, what it implies for `news_v0.yaml`) and in `artifacts/results_log.csv` (10 new rows).

## Why this exists

CLAUDE.md → Next Up #1: target / feature reformulation, direction (a) — "longer prediction horizon (5d → 20d)." Cheapest of the three directions in Next Up #1, and the answer directly shapes `news_v0.yaml`'s `target.horizon_days` for PR-C of Task 2.

As of 2026-05-27, neither logistic nor LightGBM beats the no-skill baseline on the six-TA-signal → 5d-forward-return-sign target. The LightGBM diagnostics ruled out model class as the bottleneck — three independent well-behaved configs (unweighted logistic, tiny LightGBM, calibrated LightGBM) all cluster at the no-skill floor. The 5-day-sign target is extremely noisy; a monthly horizon should have higher signal-to-noise, and the existing TA signals (RSI, MACD, Bollinger) are arguably better matched to weekly/monthly motion than daily.

Free, no-credentials, runs entirely on existing TA infrastructure.

## Claude Code kickoff prompt

```
Read CLAUDE.md, then read docs/plans/horizon-spike.md.

Execute the horizon spike (Next Up: target/feature reformulation, direction (a)).
This is a free, no-credentials task that runs entirely on existing TA infrastructure.

Steps:
1. In a new branch, clone baseline_six_signals_unweighted.yaml and the LightGBM
   config to variants with target.horizon_days in {1, 5, 10, 20, 60} — 10 configs total.
2. Run each. Each run appends a row to artifacts/results_log.csv with the existing
   per-period breakdown.
3. Build the evidence chart: skill_score vs horizon for both model classes (10 points),
   plus a per-period strategy returns table for the most interesting horizons.
4. PR with the chart + table as PR-evidence per the outcome-changing-PR convention
   (scripts/verify_*.py → docs/_pr_evidence/horizon_sweep/ → removed in cleanup commit).
5. Document the finding in CLAUDE.md's Active Task section (which horizon won, by how
   much, what it implies for news_v0.yaml). Delete this plan doc in the same cleanup
   commit per the disposable-plan-doc convention. Do NOT start the revised PR-B.

Follow protected-main PR workflow; CI green is required.
```

## Definition of done

- 10 new rows in `artifacts/results_log.csv` covering `target.horizon_days ∈ {1, 5, 10, 20, 60}` × `{logistic, LightGBM}`.
- Chart `docs/_pr_evidence/horizon_sweep/chart.png` showing `skill_score` vs horizon with both model classes on the same axes, embedded in the PR description via raw.githubusercontent.com pinned to the full 40-char commit SHA.
- Per-period strategy-returns table in the PR description for whichever horizon(s) the chart highlights as interesting.
- CLAUDE.md Active Task section updated with the finding + implication for `news_v0.yaml`'s horizon choice.
- This plan doc deleted in the cleanup commit.
- CI green on the final commit.
