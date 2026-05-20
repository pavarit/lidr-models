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


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _by_year_table(by_year_df: pd.DataFrame) -> str:
    return by_year_df.to_html(border=0, classes="data", float_format=lambda v: f"{v:.4f}")


def write_report(
    out_dir: Path,
    config_name: str,
    config_dict: dict,
    classification_metrics: dict,
    strategy_metrics: dict,
    predictions_with_returns: pd.DataFrame,
    by_year_df: pd.DataFrame,
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
<title>lidr-ml report — {config_name}</title>
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
<h1>lidr-ml backtest — {config_name}</h1>
<p class="small">Generated {timestamp}</p>

<h2>Classification metrics</h2>
<table class="data">{_metric_rows(classification_metrics)}</table>

<h2>Strategy metrics</h2>
<table class="data">{_metric_rows(strategy_metrics)}</table>

<h2>Equity curve</h2>
<img alt="equity curve" src="data:image/png;base64,{chart_b64}">

<h2>Accuracy by year</h2>
{_by_year_table(by_year_df)}

<h2>Config</h2>
<pre>{config_yaml}</pre>
</body>
</html>
"""

    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
