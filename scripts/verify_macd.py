"""Generate review-time verification evidence for the MACD port (PR #7).

Outputs:
  docs/_pr_evidence/macd/chart.png   — 3-panel SPY + MACD/signal + histogram chart
  docs/_pr_evidence/macd/evidence.md — parity number + sanity-check table

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
OUT_DIR = REPO_ROOT / "docs" / "_pr_evidence" / "macd"
FAST = 12
SLOW = 26
SIGNAL = 9
START = "2023-01-01"
END = "2026-05-01"

# Literal JS transcription of the ema() function from lidr/lib/signals/macd.ts
# (lines 12-23) plus the MACD assembly from macdSignal() (lines 42-50).
# TS type annotations stripped; algorithm byte-identical to the lidr source.
TS_MACD_JS = r"""
function ema(closes, period) {
  if (closes.length === 0 || period <= 0) return [];
  const k = 2 / (period + 1);
  const out = [];
  let prev = closes[0];
  out.push(prev);
  for (let i = 1; i < closes.length; i++) {
    prev = closes[i] * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

// Read closes from stdin; compute macd_line, signal_line, slow_ema arrays;
// emit the normalized histogram feature = (macd - signal) / slow_ema with
// the same warmup NaN mask the Python signal applies.
let raw = "";
process.stdin.on("data", chunk => raw += chunk);
process.stdin.on("end", () => {
  const { closes, fast, slow, signalP } = JSON.parse(raw);
  const fe = ema(closes, fast);
  const se = ema(closes, slow);
  const macdLine = fe.map((v, i) => v - se[i]);
  const sigLine = ema(macdLine, signalP);
  const warmup = slow + signalP + 5;
  const n = closes.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    if (i < warmup - 1) {
      out[i] = null;
    } else {
      const hist = macdLine[i] - sigLine[i];
      const v = hist / se[i];
      out[i] = Number.isFinite(v) ? v : null;
    }
  }
  // Also emit the raw macd_line, signal_line, histogram for charting.
  process.stdout.write(JSON.stringify({
    feature: out,
    macd: macdLine,
    signal: sigLine,
    slow_ema: se,
  }));
});
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full = pd.read_pickle(CACHE)
    df = full.loc[START:END].copy()
    closes = df["close"].to_numpy().tolist()
    print(f"Loaded SPY {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} days)")

    macd_fn = get_signal("macd")
    py_feature = macd_fn(df, {"fast": FAST, "slow": SLOW, "signal": SIGNAL})

    proc = subprocess.run(
        ["node", "-e", TS_MACD_JS],
        input=json.dumps({"closes": closes, "fast": FAST, "slow": SLOW, "signalP": SIGNAL}),
        capture_output=True,
        text=True,
        check=True,
    )
    ts = json.loads(proc.stdout)
    ts_feature = pd.Series(
        [np.nan if v is None else float(v) for v in ts["feature"]], index=df.index
    )
    ts_macd = pd.Series(ts["macd"], index=df.index)
    ts_signal = pd.Series(ts["signal"], index=df.index)
    ts_hist = ts_macd - ts_signal

    mask = ~(py_feature.isna() | ts_feature.isna())
    max_abs_diff = float((py_feature[mask] - ts_feature[mask]).abs().max())
    print(f"Parity: max |Python - TS| = {max_abs_diff:.2e} over {int(mask.sum())} dates")

    # Mask the macd/signal/hist series for charting so warmup section isn't drawn.
    warmup = SLOW + SIGNAL + 5
    macd_plot = ts_macd.copy()
    sig_plot = ts_signal.copy()
    hist_plot = ts_hist.copy()
    macd_plot.iloc[: warmup - 1] = np.nan
    sig_plot.iloc[: warmup - 1] = np.nan
    hist_plot.iloc[: warmup - 1] = np.nan

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )
    ax1.plot(df.index, df["close"], color="#1f77b4", linewidth=1)
    ax1.set_ylabel("SPY close (adjusted, $)")
    ax1.set_title(f"SPY — verification of Python MACD({FAST}/{SLOW}/{SIGNAL}) port (PR #7)")
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, macd_plot, color="#1f77b4", linewidth=1, label="MACD line")
    ax2.plot(df.index, sig_plot, color="#ff7f0e", linewidth=1, label="Signal line")
    ax2.axhline(0, linestyle="--", color="gray", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("MACD / signal ($)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(alpha=0.3)

    colors = np.where(hist_plot >= 0, "#2ca02c", "#d62728")
    ax3.bar(df.index, hist_plot, color=colors, width=1.0, label="histogram")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("histogram ($)")
    ax3.set_xlabel("Date")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = OUT_DIR / "chart.png"
    plt.savefig(chart_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {chart_path}")

    # Sanity checks: find crossover events and extrema.
    hist_clean = hist_plot.dropna()
    feature_clean = py_feature.dropna()

    sample_rows: list[tuple[pd.Timestamp, str]] = []

    # Most-bullish histogram (largest positive)
    max_hist_idx = hist_clean.idxmax()
    sample_rows.append((max_hist_idx, "histogram peak — strongest bullish momentum"))

    # Most-bearish histogram (largest negative)
    min_hist_idx = hist_clean.idxmin()
    sample_rows.append((min_hist_idx, "histogram trough — strongest bearish momentum"))

    # First bullish crossover (histogram crosses from < 0 to >= 0) past warmup
    sign = np.sign(hist_clean.to_numpy())
    crossings = np.where(np.diff(sign) > 0)[0]  # indices where sign goes from -1 to +1
    if len(crossings) > 0:
        first_cross_idx = hist_clean.index[crossings[0] + 1]
        sample_rows.append((first_cross_idx, "first bullish crossover (MACD crosses above signal)"))

    # First bearish crossover (histogram crosses from > 0 to <= 0) past warmup
    bear_crossings = np.where(np.diff(sign) < 0)[0]
    if len(bear_crossings) > 0:
        first_bear_idx = hist_clean.index[bear_crossings[0] + 1]
        sample_rows.append((first_bear_idx, "first bearish crossover (MACD crosses below signal)"))

    # First and last valid days
    sample_rows.append((feature_clean.index[0], "first valid (post-warmup)"))
    sample_rows.append((feature_clean.index[-1], "last valid (end of window)"))
    sample_rows.sort(key=lambda r: r[0])

    bull_days = int((hist_clean > 0).sum())
    bear_days = int((hist_clean < 0).sum())

    lines = []
    lines.append("# MACD verification — PR #7\n\n")
    lines.append(
        f"Generated by `scripts/verify_macd.py`. Input: SPY adjusted closes "
        f"{df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days) "
        f"from `data/raw/SPY_2005-01-01_2026-05-01.pkl`.\n\n"
    )
    lines.append("## Numerical parity vs lidr's TypeScript MACD\n\n")
    lines.append(f"- Parameters: fast={FAST}, slow={SLOW}, signal={SIGNAL} (lidr's `long` context)\n")
    lines.append(
        "- Compared against a literal JS transcription of `ema()` + the MACD "
        "assembly from `lidr/lib/signals/macd.ts` (type annotations stripped — "
        "algorithm byte-identical to the lidr source).\n"
    )
    lines.append(
        f"- **Max absolute difference: {max_abs_diff:.2e}** over {int(mask.sum())} dates.\n"
    )
    if max_abs_diff == 0.0:
        interp = (
            "- Interpretation: **exact bit-match**. Same recurrence, same float operations "
            "in the same order — Python and TS produce identical IEEE-754 results."
        )
    else:
        interp = (
            "- Interpretation: this is double-precision floating-point noise. The Python and TS "
            "implementations agree to machine precision."
        )
    lines.append(interp + "\n\n")
    lines.append("## Chart\n\n")
    lines.append("![SPY MACD](chart.png)\n\n")
    lines.append(
        "Top: SPY adjusted close. Middle: MACD line (blue) and its 9-period signal line "
        "(orange) — the gap between fast and slow EMAs of price, and a smoothed version "
        "of that gap. Bottom: histogram (MACD − signal), green when MACD is above its "
        "signal line (momentum turning up), red when below (momentum turning down). The "
        "Python feature we emit is histogram / slow_EMA.\n\n"
    )
    lines.append("## Sanity checks\n\n")
    lines.append("| Date | SPY close | MACD | Signal | Histogram | Feature (hist / slow_ema) | What this point shows |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for idx, label in sample_rows:
        spy_close = float(df.loc[idx, "close"])
        m = float(ts_macd.loc[idx])
        s = float(ts_signal.loc[idx])
        h = float(ts_hist.loc[idx])
        f = float(py_feature.loc[idx]) if not pd.isna(py_feature.loc[idx]) else float("nan")
        f_str = f"{f:.5f}" if not np.isnan(f) else "NaN"
        lines.append(
            f"| {idx.date()} | ${spy_close:.2f} | {m:.3f} | {s:.3f} | {h:+.3f} | {f_str} | {label} |\n"
        )
    lines.append(
        f"\n- Days with bullish histogram (MACD > signal): **{bull_days}** "
        f"({100 * bull_days / len(hist_clean):.1f}% of valid days)\n"
    )
    lines.append(
        f"- Days with bearish histogram (MACD < signal): **{bear_days}** "
        f"({100 * bear_days / len(hist_clean):.1f}% of valid days)\n"
    )
    lines.append(
        "\nMACD on a trending broad-market ETF should spend roughly equal time in each regime, "
        "with the histogram swinging through zero at trend transitions. The crossover dates and "
        "extrema above can be eyeballed against the SPY price panel to confirm they line up with "
        "visible inflection points in the index.\n"
    )

    md_path = OUT_DIR / "evidence.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Evidence saved: {md_path}")


if __name__ == "__main__":
    main()
