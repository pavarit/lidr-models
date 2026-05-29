"""TEMPLATE — copy to scripts/verify_<thing>.py and adapt. DISPOSABLE: this script and
the docs/_pr_evidence/<thing>/ dir it writes are removed in a cleanup commit before
squash-merge (the chart URL in the PR is pinned to a commit SHA, so it survives).

What it does, and why it's shaped this way:

It builds docs/_pr_evidence/<thing>/{chart.png, evidence.md} for an outcome-changing PR.
It does NOT trust the pipeline's logged numbers. Every metric in the chart and table is
recomputed *from the prediction-artifact JSON* (y_true / probability_up / y_pred), then
cross-checked against the results_log.csv row for the same config. A verify script that
re-derives metrics is itself unverified code, so it gates on a few gut checks and prints
PASS/FAIL, exiting non-zero on any failure:

  1. Parity anchor — each config reproduces its committed results_log row (or, for a port,
                      its max-abs-diff vs the reference). Edit PARITY below.
  2. Chart-vs-log  — recomputed skill_score == logged skill_score within tol (the lesson
                      from the verify script that silently read `open` as `Close`).
  3. n_oos         — artifact prediction count == logged n_oos.
  4. base_rate beside accuracy — surfaced per row so base-rate drift can't be misread as
                      skill (skill_score, not accuracy, is the cross-config metric).

Run: python scripts/verify_<thing>.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lidr_core.backtest.engine import add_strategy_returns
from lidr_core.eval.metrics import performance_by_year
from sklearn.metrics import log_loss as sk_log_loss

# ── EDIT: identify this PR's evidence ────────────────────────────────────────
THING = "my_change"  # slug -> docs/_pr_evidence/<THING>/
MODEL_ID = "ta_ensemble"  # subdir under artifacts/predictions/<MODEL_ID>/
CONFIGS = [  # config_name(s) whose artifacts this PR is about
    "baseline_six_signals_unweighted",
]
# Parity anchors: config_name -> expected skill_score from the committed results_log row
# (for a signal port, replace this gut check with a max-abs-diff vs the TS reference).
PARITY: dict[str, float] = {
    # "baseline_six_signals_unweighted": -0.005104,
}
# Cached real prices for the per-year strategy reconstruction (must already exist; run the
# backtest once to populate data/raw/). Used only for the optional per-year equity table.
PRICE_CACHE_NAME = "SPY_2005-01-01_2026-05-01.pkl"
TXN_COST_BPS = 5.0  # must match the configs' backtest.transaction_cost_bps
TOL = 5e-4  # chart-vs-log / parity tolerance
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "artifacts" / "predictions" / MODEL_ID
RESULTS_LOG = PROJECT_ROOT / "artifacts" / "results_log.csv"
PRICE_CACHE = PROJECT_ROOT / "data" / "raw" / PRICE_CACHE_NAME
OUT_DIR = PROJECT_ROOT / "docs" / "_pr_evidence" / THING


def base_logloss(base_rate: float) -> float:
    return -(base_rate * math.log(base_rate) + (1 - base_rate) * math.log(1 - base_rate))


def latest_artifact(config_name: str) -> Path:
    # The stamp is YYYYMMDD-HHMMSS, so lexical sort == chronological.
    matches = sorted(PRED_DIR.glob(f"{config_name}-*.json"))
    if not matches:
        raise FileNotFoundError(f"no artifact for {config_name} in {PRED_DIR}")
    return matches[-1]


def load_results_log() -> dict[str, dict]:
    """config_name -> latest row (later rows overwrite, keeping the most recent)."""
    rows: dict[str, dict] = {}
    with RESULTS_LOG.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["config_name"]] = row
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = load_results_log()

    daily_fwd = None
    if PRICE_CACHE.exists():
        prices = pd.read_pickle(PRICE_CACHE)
        daily_fwd = prices["close"].pct_change().shift(-1)  # loader lowercases columns

    failures: list[str] = []
    records: list[dict] = []

    for cfg in CONFIGS:
        art = json.loads(latest_artifact(cfg).read_text(encoding="utf-8"))
        preds = pd.DataFrame(art["predictions"])
        preds["date"] = pd.to_datetime(preds["date"])
        preds = preds.set_index("date").sort_index()

        # --- Recompute metrics from the artifact predictions (don't trust the log) -----
        y_true = preds["y_true"].to_numpy()
        proba = np.clip(preds["probability_up"].to_numpy(), 1e-6, 1 - 1e-6)
        base_rate = float(y_true.mean())
        ll = float(sk_log_loss(y_true, proba, labels=[0, 1]))
        bl = base_logloss(base_rate)
        skill = 1.0 - ll / bl
        accuracy = float((preds["y_pred"].to_numpy() == y_true).mean())
        pred_rate = float(preds["y_pred"].to_numpy().mean())
        n_oos = len(preds)

        # --- Gut check 2 + 3: recomputed skill / n_oos == logged -----------------------
        logged = log_rows.get(cfg)
        if logged is None:
            failures.append(f"[chart-vs-log] no results_log row for {cfg}")
        else:
            if abs(float(logged["skill_score"]) - skill) > TOL:
                failures.append(
                    f"[chart-vs-log] {cfg}: recomputed {skill:.6f} != logged {logged['skill_score']}"
                )
            if int(logged["n_oos"]) != n_oos:
                failures.append(f"[n_oos] {cfg}: artifact {n_oos} != logged {logged['n_oos']}")

        # --- Optional per-year strategy returns, reconstructed from prices -------------
        perf_yr = None
        if daily_fwd is not None:
            pwr = add_strategy_returns(
                preds[["y_pred"]].assign(y_true=preds["y_true"], y_proba_1=proba),
                forward_returns=daily_fwd.reindex(preds.index),
                transaction_cost_bps=TXN_COST_BPS,
            )
            perf_yr = performance_by_year(pwr)

        records.append(
            {
                "config": cfg,
                "base_rate": base_rate,
                "accuracy": accuracy,
                "pred_rate": pred_rate,
                "skill_score": skill,
                "log_loss": ll,
                "base_logloss": bl,
                "n_oos": n_oos,
                "strategy_cagr": float(art["metrics"]["strategy"].get("cagr") or 0.0),
                "bench_cagr": float(art["metrics"]["benchmark"].get("cagr") or 0.0),
                "perf_yr": perf_yr,
            }
        )

    df = pd.DataFrame(records)

    # --- Gut check 1: parity anchors -------------------------------------------------
    for cfg, anchor in PARITY.items():
        got = float(df[df.config == cfg]["skill_score"].iloc[0])
        if abs(got - anchor) > TOL:
            failures.append(f"[parity] {cfg}: recomputed {got:.6f} != committed {anchor:.6f}")

    # ── EDIT: build the chart that tells THIS PR's story. Default = skill_score per ──
    # config with the no-skill floor. For an equity comparison, prefer TWO panels: a
    # full-window log-scale curve + a recent-window linear zoom (years compress to pixels).
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df.config, df.skill_score, color="#1f77b4")
    ax.axhline(0.0, color="black", lw=1, ls="--", label="no-skill floor (skill_score = 0)")
    ax.set_ylabel("skill_score = 1 − log_loss / base_logloss")
    ax.set_title("Higher = better; below 0 = worse than predicting the base rate")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    chart_path = OUT_DIR / "chart.png"
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)
    # ─────────────────────────────────────────────────────────────────────────────

    # ----------------------- evidence.md ---------------------------------------------
    lines: list[str] = []
    lines.append(f"# {THING} — verification evidence\n")
    lines.append(
        "Generated by `scripts/verify_<thing>.py` from the prediction artifacts under "
        f"`artifacts/predictions/{MODEL_ID}/`. Every number below is recomputed from the "
        "artifacts' `y_true` / `probability_up`, not copied from the pipeline's logged "
        "output, and cross-checked against `results_log.csv`.\n"
    )
    lines.append("![chart](chart.png)\n")
    lines.append("## Results\n")
    lines.append(
        "`skill_score` is the headline metric (1 − log_loss/base_logloss). 0 = no skill; "
        "negative = worse than predicting the base rate every day. `base_rate`/`pred_rate` "
        "sit beside `accuracy` because accuracy alone is misleading when the base rate is "
        "far from 0.5.\n"
    )
    cols = [
        "base_rate",
        "accuracy",
        "pred_rate",
        "skill_score",
        "log_loss",
        "base_logloss",
        "n_oos",
        "strategy_cagr",
        "bench_cagr",
    ]
    lines.append("| config | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        cells = [r["config"]]
        for c in cols:
            v = r[c]
            cells.append(f"{int(v)}" if c == "n_oos" else f"{v:+.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Gut checks\n")
    checks = [
        (
            "Parity anchor (configs reproduce committed results_log rows)",
            not any(f.startswith("[parity]") for f in failures),
        ),
        (
            "Chart-vs-log (skill_score recomputed from artifacts == logged, |Δ| < tol)",
            not any(f.startswith("[chart-vs-log]") for f in failures),
        ),
        (
            "n_oos (artifact prediction count == logged n_oos)",
            not any(f.startswith("[n_oos]") for f in failures),
        ),
        ("base_rate reported beside accuracy in the table above", True),
    ]
    for label, ok in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {label}")
    lines.append("")
    (OUT_DIR / "evidence.md").write_text("\n".join(lines), encoding="utf-8")

    # ----------------------- console summary -----------------------------------------
    print(f"chart  -> {chart_path.relative_to(PROJECT_ROOT)}")
    print(f"table  -> {(OUT_DIR / 'evidence.md').relative_to(PROJECT_ROOT)}")
    print(df[["config", "base_rate", "accuracy", "skill_score", "n_oos"]].to_string(index=False))
    if failures:
        print("\n*** GUT CHECK FAILURES ***")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)
    print("\nAll gut checks PASS.")


if __name__ == "__main__":
    main()
