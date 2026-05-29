"""news_sentiment command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from lidr_core.console import ensure_utf8_stdout

from news_sentiment.datasources import REGISTRY as SOURCE_REGISTRY
from news_sentiment.features import REGISTRY as FEATURE_REGISTRY
from news_sentiment.pipeline import run_pipeline

# Make `print(...)` of non-ASCII (e.g. the pipeline's `→`) work on a stock
# Windows cp1252 console without requiring PYTHONIOENCODING=utf-8.
ensure_utf8_stdout()

app = typer.Typer(
    add_completion=False,
    help="news_sentiment: backtest the news/sentiment model.",
    no_args_is_help=True,
)


@app.command()
def backtest(
    config: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Path to a YAML config file."),
    ],
) -> None:
    """Run the full backtest pipeline for a given config."""
    result = run_pipeline(config)
    typer.echo(f"\nDone. Report: {result.report_path}")
    if result.predictions_path:
        typer.echo(f"Predictions: {result.predictions_path}")


@app.command(name="list-features")
def list_features() -> None:
    """List every news feature registered in the feature registry."""
    if not FEATURE_REGISTRY:
        typer.echo("(no features registered)")
        return
    for name in sorted(FEATURE_REGISTRY):
        typer.echo(f"- {name}")


@app.command(name="list-sources")
def list_sources() -> None:
    """List every data source registered in the source registry."""
    if not SOURCE_REGISTRY:
        typer.echo("(no sources registered)")
        return
    for name in sorted(SOURCE_REGISTRY):
        typer.echo(f"- {name}")


if __name__ == "__main__":
    app()
