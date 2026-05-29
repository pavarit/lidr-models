"""Command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from lidr_core.console import ensure_utf8_stdout

from ta_ensemble.pipeline import run_pipeline
from ta_ensemble.signals import REGISTRY

# Make `print(...)` of non-ASCII (e.g. the pipeline's `→`) work on a stock
# Windows cp1252 console without requiring PYTHONIOENCODING=utf-8.
ensure_utf8_stdout()

app = typer.Typer(
    add_completion=False,
    help="ta_ensemble: backtest the six-signal TA ensemble model.",
    no_args_is_help=True,
)


@app.command()
def backtest(
    config: Annotated[Path, typer.Argument(exists=True, readable=True, help="Path to a YAML config file.")],
) -> None:
    """Run the full backtest pipeline for a given config."""
    result = run_pipeline(config)
    typer.echo(f"\nDone. Report: {result.report_path}")
    if result.predictions_path:
        typer.echo(f"Predictions: {result.predictions_path}")


@app.command(name="list-signals")
def list_signals() -> None:
    """List every signal registered in the signals registry."""
    if not REGISTRY:
        typer.echo("(no signals registered)")
        return
    for name in sorted(REGISTRY):
        typer.echo(f"- {name}")


if __name__ == "__main__":
    app()
