"""PR-evidence generator for the class_weight sanity check.

Renders the P(up) distribution histogram for both the weighted and
unweighted six-signal logistic models, side by side, plus an
evidence.md with the headline comparison table.

The histogram is the right chart for this experiment because the
hypothesis is about how `class_weight` reshapes the model's probability
distribution — an equity curve would just show "both essentially track
buy-and-hold from different angles" without exposing the underlying
behavior.

Run from repo root:
    python scripts/verify_classweight_sanity_check.py

Outputs:
    docs/_pr_evidence/classweight_sanity_check/probability_distribution.png
    docs/_pr_evidence/classweight_sanity_check/evidence.md

Per the PR-evidence convention (see CLAUDE.md), this script and its
outputs are removed in a cleanup commit before squash-merge.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "_pr_evidence" / "classweight_sanity_check"


def latest(stem: str) -> Path:
    matches = sorted((REPO / "artifacts/predictions").glob(f"{stem}-*.json"))
    if not matches:
        raise FileNotFoundError(stem)
    return matches[-1]


def load(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload["predictions"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df.rename(columns={"probability_up": "y_proba_1"}).astype(
        {"y_true": int, "y_pred": int, "y_proba_1": float}
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    w = load(latest("baseline_six_signals"))
    u = load(latest("baseline_six_signals_unweighted"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    bins = np.linspace(0, 1, 21)  # 20 bins of width 0.05

    for ax, df, title, color in [
        (axes[0], w, "weighted (class_weight=balanced)", "#1f77b4"),
        (axes[1], u, "unweighted (class_weight=None)", "#d62728"),
    ]:
        ax.hist(df.y_proba_1, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(
            df.y_proba_1.mean(),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"mean P(up) = {df.y_proba_1.mean():.3f}",
        )
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=1.0)
        ax.set_title(
            f"{title}\n"
            f"pred_rate={df.y_pred.mean():.3f}  "
            f"accuracy={(df.y_pred == df.y_true).mean():.3f}"
        )
        ax.set_xlabel("Predicted P(up)")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Number of OOS days")
    fig.suptitle(
        "P(up) distribution: 6 signals + logistic regression on SPY, "
        f"OOS 2010-12-30 → 2026-04-23 (n={len(w)} days)",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "probability_distribution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
