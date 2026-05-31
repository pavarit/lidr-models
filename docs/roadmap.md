# Roadmap

Priority order framed around a single near-term goal: **prove the model has an edge over buy-and-hold** before building any serving/integration plumbing.

> **Status (2026-05-28):** Neither logistic nor LightGBM beats no-skill on the six TA signals → 5d-forward-return-sign target. The model class is not the bottleneck — all well-behaved configs cluster at the no-skill floor. The next move is target/feature reformulation.
>
> **Human-facing headline:** [README → Current status at a glance](../README.md#current-status-at-a-glance)

Cross-references use names, not numbers — so renumbering doesn't break them. When an item ships, remove it here and add to [docs/changelog.md](changelog.md).

---

## Active directions (before the edge gate)

1. **Reformulate the target or features.** Three concrete directions, ordered roughly by cost-to-test:
   - **(a) Longer prediction horizon** — **closed 2026-05-28.** Swept `horizon_days ∈ {5,10,20,60}` × `{logistic, weighted logistic, LightGBM}`. `skill_score` is negative at every horizon and degrades monotonically as horizon lengthens — h5 is least bad in all three model classes. The TA signals carry no usable directional information at any horizon; lengthening just raises the base rate (0.61→0.77), lowering the no-skill floor and making it harder to clear. Don't re-run horizon sweeps on this feature set.
   - **(b) Return-magnitude regression target** instead of binary sign. Requires a new target type in `pipeline.py`, a regressor instead of a classifier, and rethinking the `Model` protocol return shape. Lets the model express *how confident* and *how much*, which carries more information than sign alone.
   - **(c) Regime features.** VIX level, yield-curve slope, 60-day realized vol. New feature axes that don't overlap with the six TA signals. Needs multi-ticker support (`^VIX`, `^TNX`) in the data loader; partly implementable but not yet exercised.

2. **Introduce stacking.** Once two base learners exist *with non-zero skill*, add a `StackedModel` whose `fit` trains the base learners via out-of-fold predictions and then trains a meta-learner (logistic regression) on top. **Currently parked** — neither logistic nor LightGBM has skill, so a stacker inherits no signal. Revisit after item above produces a config that clears the no-skill floor.

---

## Edge gate — items below stay parked until something beats buy-and-hold

3. **Add a final-model fit + serialize step.** The backtest only evaluates (throwaway per-split models); nothing fits a model on all data or writes `artifacts/models/`. Before serving live predictions, fit one model on all available history, serialize it, and write the prediction artifact. Trigger: once a model beats buy-and-hold.

4. **Calibrate `predict_proba` via Platt or isotonic regression.** Wrapping LightGBM in `CalibratedClassifierCV(isotonic, cv=3)` moved its skill_score from -0.148 to -0.004 — raw probabilities are confidently miscalibrated, calibration is required before any probability artifact ships to lidr. Logistic regression's `predict_proba` is closer to calibrated out of the box but should also be wrapped for consistency.

5. **Migrate the output from binary to 3-class (BUY / HOLD / SELL).** Today's target is binary (`fwd_return > 0`). Two paths: (a) post-hoc bucketing of the calibrated probability into BUY / HOLD / SELL bands, (b) reformulate the target itself (e.g., three quantile bins of `fwd_return`). Decision punt until a binary model with edge exists; the 3-class formulation changes how "beats buy-and-hold" is measured.

6. **Evolve the artifact JSON schema as lidr's needs firm up.** Schema is formalized at `schema_version: 2` in `packages/lidr_core/src/lidr_core/contract/schema/artifact.schema.json`. Fields: `schema_version`, `model_id`, `model_version`, `config_name`, `ticker`, `generated_at`, `metrics.{classification,strategy,benchmark}`, `predictions[].{date,recommendation,probability_up,y_pred,y_true}`. Bump the version on a *breaking* change only; additive optional fields don't need a bump. Likely additions: per-signal/feature contributions for explainability.

7. **Wire lidr's `/api/signals/[ticker]` to read the artifact.** The bridge moment. Coordinate with the lidr CLAUDE.md.

8. **Wire up MLflow for experiment tracking.** Replace the timestamped-folder report + CSV log with proper logged runs and a comparison UI. *Deferred: the existing `artifacts/results_log.csv` covers the "did this help?" need until the run count is large enough to justify the heavier tooling.*
