"""Data loaders. Pulls OHLCV from yfinance, or generates synthetic series offline.

The synthetic loader exists so the pipeline can run with no internet — important
for CI, tests, and quick smoke-checks. Do not delete it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    source: str  # "yfinance" or "synthetic"
    tickers: list[str]
    start_date: str
    end_date: str
    synthetic: dict | None = None  # optional synthetic-data params

    @classmethod
    def from_dict(cls, d: dict) -> DataConfig:
        return cls(
            source=d["source"],
            tickers=list(d["tickers"]),
            start_date=str(d["start_date"]),
            end_date=str(d["end_date"]),
            synthetic=d.get("synthetic"),
        )


def load_prices(cfg: DataConfig, cache_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame} with columns [open, high, low, close, volume] indexed by date."""
    if cfg.source == "synthetic":
        return {t: _synthetic_series(t, cfg) for t in cfg.tickers}
    if cfg.source == "yfinance":
        return {t: _yfinance_series(t, cfg, cache_dir) for t in cfg.tickers}
    raise ValueError(f"Unknown data source: {cfg.source!r}")


# ---------------------------------------------------------------------------
# yfinance loader
# ---------------------------------------------------------------------------


def _yfinance_series(ticker: str, cfg: DataConfig, cache_dir: Path | None) -> pd.DataFrame:
    # Cache format is pickle, not parquet: zero extra deps (parquet would require
    # pyarrow or fastparquet), the cache is local-only and regeneratable, so
    # cross-version portability doesn't matter. If we ever need cross-tool
    # portability (e.g. reading the cache from R or DuckDB), revisit.
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{ticker}_{cfg.start_date}_{cfg.end_date}.pkl"
        if cache_path.exists():
            return pd.read_pickle(cache_path)

    import yfinance as yf  # imported lazily so synthetic runs don't need it installed

    raw = yf.download(
        ticker,
        start=cfg.start_date,
        end=cfg.end_date,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}.")

    # yfinance can return a MultiIndex on columns; flatten it.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    if cache_path is not None:
        df.to_pickle(cache_path)
    return df


# ---------------------------------------------------------------------------
# Synthetic loader — geometric Brownian motion with drift, business days only.
# ---------------------------------------------------------------------------


def _synthetic_series(ticker: str, cfg: DataConfig) -> pd.DataFrame:
    params = cfg.synthetic or {}
    drift_annual = float(params.get("drift_annual", 0.07))
    vol_annual = float(params.get("vol_annual", 0.16))
    seed = int(params.get("seed", 42))

    dates = pd.bdate_range(cfg.start_date, cfg.end_date)
    n = len(dates)
    if n == 0:
        raise ValueError("Empty synthetic date range.")

    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = drift_annual
    sigma = vol_annual

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    log_prices = np.cumsum(log_returns) + np.log(100.0)  # start near $100
    closes = np.exp(log_prices)

    # Fake OHLV around the closes — good enough for plumbing.
    daily_range = sigma * np.sqrt(dt) * closes
    highs = closes + 0.5 * daily_range * rng.uniform(0.5, 1.5, n)
    lows = closes - 0.5 * daily_range * rng.uniform(0.5, 1.5, n)
    opens = np.where(np.arange(n) == 0, closes, np.roll(closes, 1))
    volumes = rng.integers(1_000_000, 10_000_000, n)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "date"
    return df
