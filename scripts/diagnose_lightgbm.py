"""Trust-building diagnostics for the LightGBM PR. Runs five checks designed
to either confirm or contradict the headline result (skill_score = -0.148).
One-off; removed in the cleanup commit before squash-merge.

Outputs docs/_pr_evidence/lightgbm/diagnostics.md.

Checks:
  1. Column-order spot check        — confirm classes_ = [0, 1] and that
                                      proba[:, 1] really is P(class=1).
  2. In-sample fit check            — train+predict on the same training data;
                                      accuracy must be meaningfully above
                                      base_rate (else the model isn't fitting
                                      at all).
  3. Seed-stability sweep           — 5 different random_state values; if
                                      skill_score is stable, the headline -0.148
                                      isn't a single-seed lottery.
  4. Hyperparameter sensitivity     — tiny (n_est=20, leaves=4, min_child=200)
                                      vs large (n_est=500, leaves=63,
                                      min_child=5). Confirms the result isn't
                                      a quirk of the conservative defaults.
  5. Calibration wrapper            — wrap in CalibratedClassifierCV(isotonic)
                                      and re-run. If most of the -0.148 gap
                                      closes, the failure mode is mis-
                                      calibration (predictions ARE informative,
                                      just overconfident). If the gap stays
                                      open, predictions are genuinely anti-
                                      informative.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from lidr_ml.backtest.engine import add_strategy_returns, expanding_window_backtest
from lidr_ml.data.loaders import DataConfig, load_prices
from lidr_ml.eval.metrics import classification_metrics, strategy_metrics
from lidr_ml.signals import get_signal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "_pr_evidence" / "lightgbm" / "diagnostics.md"
CONFIG_PATH = ROOT / "configs" / "baseline_six_signals_lightgbm.yaml"


def build_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Reproduce pipeline.run_pipeline's feature/target construction up to the
    backtest step. Returns (X_clean, y_clean, daily_fwd_return) so each
    diagnostic can re-run the backtest with a different model_factory.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text())
    data_cfg = DataConfig.from_dict(config["data"])
    prices_by_ticker = load_prices(data_cfg, cache_dir=ROOT / "data" / "raw")
    ticker, prices = next(iter(prices_by_ticker.items()))

    features = []
    for sig_cfg in config["signals"]:
        fn = get_signal(sig_cfg["name"])
        features.append(fn(prices, sig_cfg.get("params", {}) or {}))
    X = pd.concat(features, axis=1)

    target_cfg = config["target"]
    horizon = int(target_cfg["horizon_days"])
    threshold = float(target_cfg.get("threshold", 0.0))
    fwd_return = prices["close"].pct_change(horizon).shift(-horizon)
    y = (fwd_return > threshold).astype(int)

    aligned = pd.concat([X, y.rename("__y__"), fwd_return.rename("__fwd__")], axis=1).dropna()
    X_clean = aligned[X.columns]
    y_clean = aligned["__y__"].astype(int)
    daily_fwd = prices["close"].pct_change().shift(-1)
    return X_clean, y_clean, daily_fwd, ticker


def lgbm_factory(**overrides):
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": 0,
        "verbose": -1,
    }
    defaults.update(overrides)

    class _LGBM:
        def __init__(self):
            self._m = LGBMClassifier(**defaults)

        def fit(self, X, y):
            self._m.fit(X, y)

        def predict_proba(self, X):
            return self._m.predict_proba(X)

        def predict(self, X):
            return self._m.predict(X)

    return _LGBM


def calibrated_lgbm_factory(**overrides):
    """LightGBM wrapped in CalibratedClassifierCV(isotonic, cv=3). cv=3 means
    each backtest split fits the underlying LGBM 3 times on rolling internal
    folds, then fits an isotonic calibrator on the held-out predictions.
    Roughly 3-4x slower than plain LightGBM.
    """
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": 0,
        "verbose": -1,
    }
    defaults.update(overrides)

    class _Cal:
        def __init__(self):
            base = LGBMClassifier(**defaults)
            self._m = CalibratedClassifierCV(base, method="isotonic", cv=3)

        def fit(self, X, y):
            self._m.fit(X, y)

        def predict_proba(self, X):
            return self._m.predict_proba(X)

        def predict(self, X):
            return self._m.predict(X)

    return _Cal


def run_backtest(X, y, daily_fwd, factory_cls) -> tuple[dict, dict, dict, int]:
    bt = expanding_window_backtest(
        X, y,
        model_factory=lambda: factory_cls(),
        initial_train_years=5,
        test_period_months=12,
    )
    pred_with_ret = add_strategy_returns(
        bt.predictions,
        forward_returns=daily_fwd.reindex(bt.predictions.index),
        transaction_cost_bps=5.0,
    )
    cls_m = classification_metrics(
        bt.predictions["y_true"],
        bt.predictions["y_pred"],
        bt.predictions["y_proba_1"],
    )
    strat_m = strategy_metrics(pred_with_ret["strategy_equity"])
    bench_m = strategy_metrics(pred_with_ret["buy_hold_equity"])
    return cls_m, strat_m, bench_m, len(bt.predictions)


def skill(cls_m: dict) -> float:
    return 1.0 - cls_m["log_loss"] / cls_m["base_logloss"]


def fmt_row(label, cls_m, strat_m, bench_m, n, extra=""):
    s = skill(cls_m)
    excess = strat_m["cagr"] - bench_m["cagr"]
    return (
        f"| {label} | {n} | {s:+.4f} | {cls_m['accuracy']:.4f} | "
        f"{cls_m['pred_rate']:.4f} | {cls_m['log_loss']:.4f} | "
        f"{strat_m['cagr']:+.4f} | {excess:+.4f} | {extra} |"
    )


def main():
    print("Building features...")
    X, y, daily_fwd, ticker = build_features()
    print(f"  {len(X)} usable rows, {X.shape[1]} features, base_rate={y.mean():.4f}, ticker={ticker}")

    sections: list[str] = []

    # ------------------------------------------------------------------------
    # 1. Column-order spot check
    # ------------------------------------------------------------------------
    print("\n[1/5] Column-order spot check...")
    split_end = X.index[0] + pd.DateOffset(years=5)
    X_train_one = X.loc[X.index < split_end]
    y_train_one = y.loc[X.index < split_end]
    one_model = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=0, verbose=-1,
    )
    one_model.fit(X_train_one, y_train_one)
    classes = list(one_model.classes_)
    # Sanity: high-RSI → low next-period probability (RSI > 70 is overbought).
    # We can't depend on the SIGN of the relationship being right (that's what
    # we're testing the model on), but we CAN sanity-check that mean P(class=1)
    # on the up-day subset of training data > mean on the down-day subset.
    # If we accidentally got P(class=0) into y_proba_1, this comparison flips.
    proba_train = one_model.predict_proba(X_train_one)
    mean_p_on_ups = proba_train[y_train_one.to_numpy() == 1, 1].mean()
    mean_p_on_dns = proba_train[y_train_one.to_numpy() == 0, 1].mean()
    col_order_ok = classes == [0, 1] and mean_p_on_ups > mean_p_on_dns

    sections.append(f"""## 1. Column-order spot check

Fitted one LightGBM on the first-split training data (~{len(X_train_one)} rows)
and inspected the output of `predict_proba`.

- `model.classes_` = `{classes}` (expected `[0, 1]`)
- Mean `proba[:, 1]` on training rows where `y=1` (up days): **{mean_p_on_ups:.4f}**
- Mean `proba[:, 1]` on training rows where `y=0` (down days): **{mean_p_on_dns:.4f}**

If the column order were silently swapped, the second number would exceed the
first. It doesn't. {'**PASS** — `y_proba_1` really is P(class=1).' if col_order_ok else '**FAIL — column order may be wrong.**'}
""")
    print(f"  classes_={classes}, mean P on ups={mean_p_on_ups:.4f}, on dns={mean_p_on_dns:.4f}")
    print(f"  → {'PASS' if col_order_ok else 'FAIL'}")

    # ------------------------------------------------------------------------
    # 2. In-sample fit check
    # ------------------------------------------------------------------------
    print("\n[2/5] In-sample fit check...")
    insample_acc = float((one_model.predict(X_train_one) == y_train_one.to_numpy()).mean())
    base_train = float(y_train_one.mean())
    insample_ok = insample_acc > base_train + 0.05  # meaningful fit, not just predict-base

    sections.append(f"""## 2. In-sample fit check

If the model can't even fit its own training data, something's broken. Should
be **well above** training base rate.

- Training-set accuracy: **{insample_acc:.4f}**
- Training-set base rate: **{base_train:.4f}**
- Margin above base rate: **{insample_acc - base_train:+.4f}**

{'**PASS** — model is fitting (out-of-sample collapse is the real story, not a fit failure).' if insample_ok else '**FAIL — model failed to fit training data; bug somewhere.**'}
""")
    print(f"  in-sample acc={insample_acc:.4f} vs base={base_train:.4f} → {'PASS' if insample_ok else 'FAIL'}")

    # ------------------------------------------------------------------------
    # 3. Seed-stability sweep
    # ------------------------------------------------------------------------
    print("\n[3/5] Seed-stability sweep (5 seeds)...")
    seed_rows = []
    seed_skills = []
    for seed in [0, 1, 7, 42, 2026]:
        t0 = time.time()
        cls_m, strat_m, bench_m, n = run_backtest(
            X, y, daily_fwd, factory_cls=lgbm_factory(random_state=seed)
        )
        print(f"  seed={seed}: skill={skill(cls_m):+.4f} (took {time.time()-t0:.1f}s)")
        seed_skills.append(skill(cls_m))
        seed_rows.append(fmt_row(f"seed={seed}", cls_m, strat_m, bench_m, n))
    s_arr = np.array(seed_skills)
    seed_identical = s_arr.std() < 1e-6  # all five gave the same answer to 6+ digits
    seed_stable = (s_arr.max() - s_arr.min() < 0.05) and (s_arr.max() < 0)
    if seed_identical:
        seed_verdict = (
            "**PASS, but trivially.** All five seeds returned the *exact same* "
            "skill_score. LightGBM's defaults set both `feature_fraction` and "
            "`bagging_fraction` to 1.0 (no subsampling), so `random_state` has "
            "no source of randomness to consume — every seed produces the "
            "bit-identical model. This confirms determinism (we're not "
            "cherry-picking a lucky run) but it does not provide independent "
            "evidence the way a true seed sweep would. To get real "
            "seed-stability evidence we'd need to enable subsampling first "
            "(e.g., `feature_fraction=0.7`, `bagging_fraction=0.7`, "
            "`bagging_freq=1`). The hyperparameter sweep in §4 below is the "
            "stronger robustness check."
        )
    elif seed_stable:
        seed_verdict = "**PASS** — all seeds clearly negative and clustered tightly; headline is not a fluke."
    else:
        seed_verdict = "**MIXED** — meaningful seed-to-seed variation; headline -0.148 is one draw, not stable."

    seed_table = "\n".join(seed_rows)
    sections.append(f"""## 3. Seed-stability sweep

Intent: if `skill_score = -0.148` is a single-seed lottery, the conclusion
isn't trustworthy. Five seeds:

| seed | n_oos | skill_score | accuracy | pred_rate | log_loss | strat CAGR | excess CAGR |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{seed_table}

**Range:** {s_arr.min():+.4f} to {s_arr.max():+.4f}   **Mean:** {s_arr.mean():+.4f}   **Std:** {s_arr.std():.4f}

{seed_verdict}
""")

    # ------------------------------------------------------------------------
    # 4. Hyperparameter sensitivity
    # ------------------------------------------------------------------------
    print("\n[4/5] Hyperparameter sweep...")
    hyp_configs = [
        ("tiny (n_est=20, leaves=4, min_child=200)",
         dict(n_estimators=20, num_leaves=4, min_child_samples=200)),
        ("default (n_est=200, leaves=31, min_child=20)",
         dict()),
        ("large (n_est=500, leaves=63, min_child=5, lr=0.03)",
         dict(n_estimators=500, num_leaves=63, min_child_samples=5, learning_rate=0.03)),
    ]
    hyp_rows = []
    hyp_skills = []
    for label, overrides in hyp_configs:
        t0 = time.time()
        cls_m, strat_m, bench_m, n = run_backtest(
            X, y, daily_fwd, factory_cls=lgbm_factory(**overrides)
        )
        print(f"  {label}: skill={skill(cls_m):+.4f} (took {time.time()-t0:.1f}s)")
        hyp_skills.append(skill(cls_m))
        hyp_rows.append(fmt_row(label, cls_m, strat_m, bench_m, n))
    hyp_all_neg = max(hyp_skills) < 0

    hyp_table = "\n".join(hyp_rows)
    sections.append(f"""## 4. Hyperparameter sensitivity

Tiny config is biased toward logistic-like behavior (4 leaves, large min_child).
Large config has more capacity (63 leaves, 500 trees, lower lr).

| config | n_oos | skill_score | accuracy | pred_rate | log_loss | strat CAGR | excess CAGR |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{hyp_table}

{'**PASS** — all three configs give negative skill; result is not a hyperparameter quirk.' if hyp_all_neg else '**MIXED** — one config gives positive skill; need to investigate.'}
""")

    # ------------------------------------------------------------------------
    # 5. Calibration wrapper
    # ------------------------------------------------------------------------
    print("\n[5/5] Calibration wrapper (CalibratedClassifierCV, isotonic, cv=3)...")
    t0 = time.time()
    cls_m, strat_m, bench_m, n = run_backtest(
        X, y, daily_fwd, factory_cls=calibrated_lgbm_factory()
    )
    cal_skill = skill(cls_m)
    print(f"  calibrated skill={cal_skill:+.4f} (took {time.time()-t0:.1f}s)")

    raw_skill = seed_skills[0]  # seed=0 from the stability sweep is the apples-to-apples comparison
    delta = cal_skill - raw_skill
    cal_row = fmt_row("LightGBM + isotonic calibration", cls_m, strat_m, bench_m, n)

    if cal_skill > -0.02:
        cal_interp = (
            f"**Calibration recovers nearly all of the gap** ({raw_skill:+.4f} → {cal_skill:+.4f}). "
            "Reframes the result: LightGBM's predictions ARE informative — they were just badly "
            "miscalibrated. Skill is roughly at the no-skill floor (like the unweighted logistic). "
            "The PR's framing should change from 'fits noise' to 'comparable signal to logistic, "
            "needs calibration before any real comparison.'"
        )
    elif cal_skill > raw_skill + 0.05:
        cal_interp = (
            f"**Calibration recovers a meaningful chunk of the gap** ({raw_skill:+.4f} → "
            f"{cal_skill:+.4f}, Δ={delta:+.4f}) but skill is still clearly negative. "
            "Mixed story: miscalibration is part of the problem but not all of it. LightGBM "
            "still has *some* anti-informative signal in its predictions."
        )
    else:
        cal_interp = (
            f"**Calibration barely moves the needle** ({raw_skill:+.4f} → {cal_skill:+.4f}, "
            f"Δ={delta:+.4f}). Confirms the PR's framing: LightGBM is genuinely fitting noise, "
            "not just expressing the same signal with bad probabilities."
        )

    sections.append(f"""## 5. Calibration wrapper

Wrapped LightGBM in `CalibratedClassifierCV(method='isotonic', cv=3)`. Each
backtest split internally trains LightGBM 3 times on rolling sub-folds, then
fits an isotonic regressor on the held-out predictions. ~3-4× slower; produces
calibrated probabilities.

| config | n_oos | skill_score | accuracy | pred_rate | log_loss | strat CAGR | excess CAGR |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM (seed=0, uncalibrated, baseline) | {n} | {raw_skill:+.4f} | — | — | — | — | — | — |
{cal_row}

**Δ from uncalibrated:** {delta:+.4f}

{cal_interp}
""")

    # ------------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------------
    checks_passed = sum([col_order_ok, insample_ok, seed_stable, hyp_all_neg])
    verdict = f"""# LightGBM PR — diagnostics

Generated by `scripts/diagnose_lightgbm.py` (removed in cleanup commit).
Validates the headline result `skill_score = -0.1478` against five failure modes.

**Summary: {checks_passed}/4 binary checks pass + 1 interpretation-sensitive check (calibration).**

"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(verdict + "\n".join(sections), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"\nOverall: {checks_passed}/4 binary checks passed; calibration delta = {delta:+.4f}")


if __name__ == "__main__":
    main()
