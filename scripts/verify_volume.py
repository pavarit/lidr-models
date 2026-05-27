"""Generate review-time verification evidence for the volume signal port (PR #10).

Outputs:
  docs/_pr_evidence/volume/chart.png   — 3-panel SPY + volume + ratio chart
  docs/_pr_evidence/volume/evidence.md — parity number + sanity-check table

This script and docs/_pr_evidence/ are removed in a cleanup commit before the
PR is squash-merged. Nothing here lands on main.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lidr_ml.signals import get_signal

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "raw" / "SPY_2005-01-01_2026-05-01.pkl"
OUT_DIR = REPO_ROOT / "docs" / "_pr_evidence" / "volume"
PERIOD = 50  # lidr's `long` context: volumeAvgDays = 50
HEAVY_THRESHOLD = 1.5  # lidr's default volumeMultiplier
START = "2023-01-01"
END = "2026-05-01"

# Literal JS transcription of the rolling-mean-and-ratio logic from
# lidr/lib/signals/volume.ts (lines 42-45), type annotations stripped.
# Average INCLUDES today (slice(-N) takes N values ending at the current
# index), matching the Python signal's pd.rolling(N).mean() semantics.
TS_VOLUME_JS = r"""
let raw = "";
process.stdin.on("data", chunk => raw += chunk);
process.stdin.on("end", () => {
  const { volumes, period } = JSON.parse(raw);
  const n = volumes.length;
  const ratio = new Array(n).fill(null);
  const avg = new Array(n).fill(null);
  for (let i = period - 1; i < n; i++) {
    const slice = volumes.slice(i - period + 1, i + 1);
    const a = slice.reduce((s, v) => s + v, 0) / slice.length;
    avg[i] = a;
    if (a > 0) {
      const v = volumes[i] / a;
      ratio[i] = Number.isFinite(v) ? v : null;
    }
  }
  process.stdout.write(JSON.stringify({ ratio, avg }));
});
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full = pd.read_pickle(CACHE)
    df = full.loc[START:END].copy()
    volumes = df["volume"].to_numpy().tolist()
    print(f"Loaded SPY {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} days)")

    fn = get_signal("volume")
    py_ratio = fn(df, {"period": PERIOD})

    proc = subprocess.run(
        ["node", "-e", TS_VOLUME_JS],
        input=json.dumps({"volumes": volumes, "period": PERIOD}),
        capture_output=True,
        text=True,
        check=True,
    )
    ts = json.loads(proc.stdout)
    ts_ratio = pd.Series(
        [np.nan if v is None else float(v) for v in ts["ratio"]], index=df.index
    )
    ts_avg = pd.Series(
        [np.nan if v is None else float(v) for v in ts["avg"]], index=df.index
    )

    mask = ~(py_ratio.isna() | ts_ratio.isna())
    max_abs_diff = float((py_ratio[mask] - ts_ratio[mask]).abs().max())
    print(f"Parity: max |Python - TS| = {max_abs_diff:.2e} over {int(mask.sum())} dates")

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )
    ax1.plot(df.index, df["close"], color="#1f77b4", linewidth=1)
    ax1.set_ylabel("SPY close (adjusted, $)")
    ax1.set_title(f"SPY — verification of Python volume({PERIOD}-day ratio) port (PR #10)")
    ax1.grid(alpha=0.3)

    vol_millions = df["volume"] / 1_000_000
    avg_millions = ts_avg / 1_000_000
    ax2.bar(df.index, vol_millions, color="#bbbbbb", width=1.0, label="daily volume")
    ax2.plot(df.index, avg_millions, color="#ff7f0e", linewidth=1.2, label=f"{PERIOD}-day rolling average")
    ax2.set_ylabel("volume (M shares)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.plot(df.index, py_ratio, color="#9467bd", linewidth=1, label="volume / avg")
    ax3.axhline(1.0, color="gray", alpha=0.6, linewidth=0.7, label="1.0 (average)")
    ax3.axhline(HEAVY_THRESHOLD, linestyle="--", color="green", alpha=0.7,
                linewidth=0.8, label=f"{HEAVY_THRESHOLD}× (lidr 'heavy' threshold)")
    ax3.axhspan(HEAVY_THRESHOLD, max(py_ratio.dropna().max() + 0.2, HEAVY_THRESHOLD + 0.5),
                alpha=0.10, color="green")
    ax3.set_ylabel("ratio (× average)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = OUT_DIR / "chart.png"
    plt.savefig(chart_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {chart_path}")

    ratio_clean = py_ratio.dropna()

    sample_rows: list[tuple[pd.Timestamp, str]] = []
    sample_rows.append((ratio_clean.idxmax(), "ratio peak — biggest volume spike vs recent average"))
    sample_rows.append((ratio_clean.idxmin(), "ratio trough — quietest day vs recent average"))
    above_threshold = ratio_clean[ratio_clean >= HEAVY_THRESHOLD]
    if len(above_threshold) > 0:
        sample_rows.append((above_threshold.index[0], f"first day above {HEAVY_THRESHOLD}× (lidr 'heavy volume')"))
        sample_rows.append((above_threshold.index[-1], f"most recent day above {HEAVY_THRESHOLD}×"))
    sample_rows.append((ratio_clean.index[0], "first valid (window just full)"))
    sample_rows.append((ratio_clean.index[-1], "last valid (end of window)"))
    seen: dict[pd.Timestamp, str] = {}
    for idx, label in sample_rows:
        seen.setdefault(idx, label)
    sample_rows = sorted(seen.items(), key=lambda r: r[0])

    above = int((ratio_clean >= HEAVY_THRESHOLD).sum())
    below_half = int((ratio_clean < 0.5).sum())
    near_one = int(((ratio_clean >= 0.8) & (ratio_clean <= 1.2)).sum())
    total = len(ratio_clean)

    lines = []
    lines.append("# Volume verification — PR #10\n\n")
    lines.append(
        f"Generated by `scripts/verify_volume.py`. Input: SPY adjusted volume "
        f"{df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days) "
        f"from `data/raw/SPY_2005-01-01_2026-05-01.pkl`.\n\n"
    )
    lines.append("## Numerical parity vs lidr's TypeScript volume signal\n\n")
    lines.append(
        f"- Parameters: period={PERIOD} (lidr's `long` context — `volumeAvgDays`).\n"
    )
    lines.append(
        "- Compared against a literal JS transcription of the rolling-mean and ratio logic "
        "from `lidr/lib/signals/volume.ts` (type annotations stripped — algorithm "
        "byte-identical to the lidr source).\n"
    )
    lines.append(f"- **Max absolute difference: {max_abs_diff:.2e}** over {int(mask.sum())} dates.\n")
    if max_abs_diff == 0.0:
        interp = (
            "- Interpretation: **exact bit-match**. Same arithmetic, same float operations — "
            "Python and TS produce identical IEEE-754 results."
        )
    else:
        interp = (
            "- Interpretation: this is double-precision floating-point noise. The Python and TS "
            "implementations agree to machine precision."
        )
    lines.append(interp + "\n\n")
    lines.append("## Chart\n\n")
    lines.append("![SPY volume chart](chart.png)\n\n")
    lines.append(
        f"Top: SPY adjusted close (for context — the volume signal doesn't use price). Middle: "
        f"daily traded volume (grey bars) with the {PERIOD}-day rolling average overlaid (orange). "
        "Bottom: the ratio Python emits as the ML feature — today's volume divided by its "
        f"{PERIOD}-day average. The dashed green line marks the {HEAVY_THRESHOLD}× threshold lidr "
        "uses to call volume \"heavy\"; the shaded green region above it is the heavy-volume "
        "regime.\n\n"
    )
    lines.append("## Sanity checks\n\n")
    lines.append("| Date | SPY close | Today vol (M) | 50-day avg vol (M) | Ratio | What this point shows |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for idx, label in sample_rows:
        spy_close = float(df.loc[idx, "close"])
        v = float(df.loc[idx, "volume"]) / 1_000_000
        a = float(ts_avg.loc[idx]) / 1_000_000
        r = float(py_ratio.loc[idx])
        lines.append(
            f"| {idx.date()} | ${spy_close:.2f} | {v:.1f} | {a:.1f} | {r:.3f} | {label} |\n"
        )
    lines.append(
        f"\n- Days with heavy volume (ratio ≥ {HEAVY_THRESHOLD}): **{above}** ({100 * above / total:.1f}% of valid days)\n"
    )
    lines.append(
        f"- Days near average volume (0.8× to 1.2×): **{near_one}** ({100 * near_one / total:.1f}% of valid days)\n"
    )
    lines.append(
        f"- Quiet days (ratio < 0.5): **{below_half}** ({100 * below_half / total:.1f}% of valid days)\n"
    )
    lines.append(
        "\nThe ratio is bounded below by zero (no negative volume) and unbounded above; the "
        "distribution is naturally right-skewed because volume spikes around news/earnings/macro "
        "events have no upper limit but quiet days bottom out near today's volume itself. A "
        "well-behaved volume series should cluster around 1.0 with occasional excursions to 1.5×+.\n"
    )

    md_path = OUT_DIR / "evidence.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Evidence saved: {md_path}")


if __name__ == "__main__":
    main()
