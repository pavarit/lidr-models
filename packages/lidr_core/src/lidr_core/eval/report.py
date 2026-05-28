"""HTML report generator. Self-contained — embeds the equity-curve chart as a
base64 PNG so the report opens correctly even with no internet connection.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display; safe in headless / sandbox runs
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_chart(predictions_with_returns: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions_with_returns.index, predictions_with_returns["buy_hold_equity"],
            label="Buy & Hold", linewidth=1.2)
    ax.plot(predictions_with_returns.index, predictions_with_returns["strategy_equity"],
            label="Strategy", linewidth=1.5)
    ax.set_title("Out-of-sample equity curve")
    ax.set_ylabel("Equity (starting at $1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _fig_to_base64(fig)


def _metric_rows(metrics: dict) -> str:
    return "".join(
        f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>" for k, v in metrics.items()
    )


def _comparison_rows(strategy: dict, benchmark: dict) -> str:
    """One row per metric. Buy & Hold first (the primary benchmark), Strategy next.
    The Strategy cell is highlighted green when it beats Buy & Hold, red when worse.
    Higher = better for every current metric (cagr, sharpe, final_equity, and
    max_drawdown, which is stored negative so closer-to-zero is the higher/better value)."""
    rows = []
    for k in strategy:
        s = strategy.get(k)
        b = benchmark.get(k)
        hl = ""
        if isinstance(s, (int, float)) and isinstance(b, (int, float)):
            if s > b:
                hl = "background:#e6f4ea;color:#0a7d28;font-weight:600"
            elif s < b:
                hl = "background:#fdecea;color:#c0392b;font-weight:600"
        sty = f' style="{hl}"' if hl else ""
        rows.append(f"<tr><td>{k}</td><td>{_fmt(b)}</td><td{sty}>{_fmt(s)}</td></tr>")
    return "".join(rows)


def _performance_table(df: pd.DataFrame) -> str:
    """Per-year table. Buy & Hold column comes before Strategy. The strategy_return
    and excess cells are highlighted green when the strategy beat buy & hold that
    year (excess > 0) and red when it lagged (excess < 0)."""
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in [df.index.name or "year", *cols])
    body = []
    for idx, row in df.iterrows():
        excess = float(row.get("excess", 0.0))
        if excess > 0:
            hl = "background:#e6f4ea;color:#0a7d28;font-weight:600"
        elif excess < 0:
            hl = "background:#fdecea;color:#c0392b;font-weight:600"
        else:
            hl = ""
        tds = [f"<td>{idx}</td>"]
        for c in cols:
            v = row[c]
            disp = f"{int(v)}" if c == "n" else f"{v:.4f}"
            sty = f' style="{hl}"' if (c in ("strategy_return", "excess") and hl) else ""
            tds.append(f"<td{sty}>{disp}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="data"><tr>{header}</tr>{"".join(body)}</table>'


def _classification_table(df: pd.DataFrame) -> str:
    """Per-year classification table. The log_loss cell is highlighted green when it
    beats the no-skill floor (log_loss < base_logloss) and red when it doesn't."""
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in [df.index.name or "year", *cols])
    body = []
    for idx, row in df.iterrows():
        ll = float(row.get("log_loss", 0.0))
        floor = float(row.get("base_logloss", 0.0))
        if ll < floor:
            hl = "background:#e6f4ea;color:#0a7d28;font-weight:600"
        elif ll > floor:
            hl = "background:#fdecea;color:#c0392b;font-weight:600"
        else:
            hl = ""
        tds = [f"<td>{idx}</td>"]
        for c in cols:
            v = row[c]
            disp = f"{int(v)}" if c == "n" else f"{v:.4f}"
            sty = f' style="{hl}"' if (c == "log_loss" and hl) else ""
            tds.append(f"<td{sty}>{disp}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="data"><tr>{header}</tr>{"".join(body)}</table>'


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _by_year_table(by_year_df: pd.DataFrame) -> str:
    return by_year_df.to_html(border=0, classes="data", float_format=lambda v: f"{v:.4f}")


def _config_summary(config_dict: dict, predictions_with_returns: pd.DataFrame) -> str:
    """Translate the experiment config into plain English + a quick-reference table.

    The out-of-sample span is derived from the predictions, not the config, so the
    summary states the period actually evaluated (not the raw data range).
    """
    data = config_dict.get("data", {})
    tickers = ", ".join(data.get("tickers", [])) or "?"
    source = data.get("source", "?")
    start = data.get("start_date", "?")
    end = data.get("end_date", "?")

    sig_parts = []
    for s in config_dict.get("signals", []):
        params = s.get("params", {}) or {}
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        sig_parts.append(f"{s.get('name')} ({ps})" if ps else str(s.get("name")))
    signals = "; ".join(sig_parts) or "?"

    target = config_dict.get("target", {})
    horizon = int(target.get("horizon_days", 0))
    threshold = float(target.get("threshold", 0.0))

    model = config_dict.get("model", {})
    mparams = model.get("params", {}) or {}
    mparams_str = ", ".join(f"{k}={v}" for k, v in mparams.items())
    model_desc = model.get("type", "?") + (f" ({mparams_str})" if mparams_str else "")

    bt = config_dict.get("backtest", {})
    cv = bt.get("cv", "?")
    train_yrs = bt.get("initial_train_years", "?")
    test_months = bt.get("test_period_months", "?")
    cost = bt.get("transaction_cost_bps", 5.0)

    idx = predictions_with_returns.index
    oos_start = idx.min().date() if len(idx) else "?"
    oos_end = idx.max().date() if len(idx) else "?"
    n = len(idx)

    desc = (config_dict.get("description") or "").strip()
    desc_html = f"<p>{desc}</p>" if desc else ""

    return f"""{desc_html}
<p>This backtest evaluates a <b>{model_desc}</b> model that predicts whether
<b>{tickers}</b>'s return over the next <b>{horizon} day(s)</b> will exceed
{threshold:g} (i.e. a positive move), using these signal(s) as input features:
<b>{signals}</b>.</p>
<table class="data">
<tr><td>Ticker(s)</td><td>{tickers}</td></tr>
<tr><td>Price data</td><td>{source}, {start} &rarr; {end}</td></tr>
<tr><td>Prediction target</td><td>{horizon}-day forward return &gt; {threshold:g}</td></tr>
<tr><td>Signal(s)</td><td>{signals}</td></tr>
<tr><td>Model</td><td>{model_desc}</td></tr>
<tr><td>Validation</td><td>{cv} walk-forward &mdash; {train_yrs}-yr initial train, re-fit every {test_months} months</td></tr>
<tr><td>Transaction cost</td><td>{cost} bps per position change</td></tr>
<tr><td>Out-of-sample period</td><td>{oos_start} &rarr; {oos_end} ({n} trading days)</td></tr>
</table>
<p class="small">All metrics below are <b>out-of-sample</b>: computed only on dates the
model never trained on (the {oos_start} &rarr; {oos_end} span above), not the full data range.</p>"""


def write_report(
    out_dir: Path,
    config_name: str,
    config_dict: dict,
    classification_metrics: dict,
    strategy_metrics: dict,
    benchmark_metrics: dict,
    predictions_with_returns: pd.DataFrame,
    by_year_df: pd.DataFrame,
    performance_by_year_df: pd.DataFrame,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_b64 = _equity_chart(predictions_with_returns)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    import yaml  # local import — yaml only needed for pretty-printing
    config_yaml = yaml.safe_dump(config_dict, sort_keys=False, default_flow_style=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>lidr-models report — {config_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-weight: 600; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #eee; padding-bottom: .35rem; }}
  table {{ border-collapse: collapse; margin: .5rem 0; }}
  table.data td, table.data th {{ padding: .35rem .8rem; border-bottom: 1px solid #eee; }}
  .small {{ color: #777; font-size: .9rem; }}
  pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: .85rem; }}
  img {{ max-width: 100%; border: 1px solid #eee; border-radius: 6px; }}
</style>
</head>
<body>
<h1>lidr-models backtest — {config_name}</h1>
<p class="small">Generated {timestamp}</p>

<h2>Summary</h2>
{_config_summary(config_dict, predictions_with_returns)}

<h2>Classification metrics</h2>
<table class="data">{_metric_rows(classification_metrics)}</table>

<h2>Buy &amp; Hold vs Strategy</h2>
<p class="small">Is the model better than just holding the index? Buy &amp; Hold is the benchmark; Strategy follows.</p>
<table class="data">
<tr><th>Metric</th><th>Buy &amp; Hold</th><th>Strategy</th></tr>
{_comparison_rows(strategy_metrics, benchmark_metrics)}
</table>

<h2>Equity curve</h2>
<img alt="equity curve" src="data:image/png;base64,{chart_b64}">

<h2>Performance by year (return, net of costs)</h2>
<p class="small">Per-year buy-and-hold vs strategy return, and the excess. Green = strategy beat buy &amp; hold that year; red = it lagged. Catches strategies that worked in the past but not recently.</p>
{_performance_table(performance_by_year_df)}

<h2>Accuracy &amp; log loss by year</h2>
<p class="small">Two reads per year. Accuracy vs base_rate: did the up/down calls beat a naive "always up" guess? log_loss vs base_logloss (the no-skill floor): green = log loss <em>below</em> the floor (probabilities were informative that year); red = at or above (no edge).</p>
{_classification_table(by_year_df)}

<h2>Config</h2>
<pre>{config_yaml}</pre>
</body>
</html>
"""

    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
