"""Recorded-fixture integration tests for the HTTP news adapters.

Each test feeds a small recorded sample of the provider's real JSON response
through the adapter and asserts the field mapping into ``NewsItem`` — including
the point-in-time-critical ``published_at`` parse. ``requests.get`` is
monkeypatched per adapter module so **no live API quota is ever burned in CI**
(EODHD bills 5 calls/request; Finnhub/Anthropic have rate limits).

The recorded payloads are trimmed to the fields the adapter reads, in the exact
shape the provider returns them (verified against each provider's live response
during PR-B verification).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from news_sentiment.datasources.apewisdom import ApewisdomSource
from news_sentiment.datasources.eodhd import EodhdSource
from news_sentiment.datasources.finnhub import FinnhubSource
from news_sentiment.datasources.hn import HackerNewsSource


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def json(self) -> object:
        return self._payload


def _patch_get(monkeypatch, module, payload, capture: dict | None = None) -> None:
    def fake_get(url, params=None, timeout=None, **kw):
        if capture is not None:
            capture["url"] = url
            capture["params"] = params
        return _FakeResponse(payload)

    monkeypatch.setattr(module.requests, "get", fake_get)


# --------------------------------------------------------------------------- #
# Finnhub                                                                      #
# --------------------------------------------------------------------------- #

# Recorded /company-news shape: a flat array; `datetime` is Unix seconds.
_FINNHUB_PAYLOAD = [
    {
        "category": "company news",
        "datetime": 1705320000,  # 2024-01-15 12:00:00 UTC
        "headline": "AAPL unveils record quarter",
        "id": 111,
        "source": "Reuters",
        "summary": "Apple beat expectations on services revenue.",
        "url": "https://example.com/aapl-q1",
    },
    {
        "category": "company news",
        "datetime": 1705924800,  # 2024-01-22 12:00:00 UTC
        "headline": "AAPL faces antitrust probe",
        "id": 112,
        "source": "Bloomberg",
        "summary": "Regulators open an inquiry.",
        "url": "https://example.com/aapl-probe",
    },
    {"datetime": 0, "headline": "should be dropped (no timestamp)"},
]


def test_finnhub_maps_fields_and_timestamp(monkeypatch) -> None:
    import news_sentiment.datasources.finnhub as mod

    cap: dict = {}
    _patch_get(monkeypatch, mod, _FINNHUB_PAYLOAD, cap)
    src = FinnhubSource(request_delay_s=0.0, api_key="test-key")
    items = src.fetch("AAPL", "2024-01-01", "2024-02-01")

    assert len(items) == 2  # the timestamp-0 row is dropped
    first = items[0]
    assert first.source == "finnhub"
    assert first.ticker == "AAPL"
    assert first.title == "AAPL unveils record quarter"
    assert first.published_at == datetime(2024, 1, 15, 12, 0, 0)
    assert first.meta["finnhub_source"] == "Reuters"
    # window/sorting handled by the base class
    assert items[0].published_at < items[1].published_at
    # API key forwarded as token
    assert cap["params"]["token"] == "test-key"


def test_finnhub_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        FinnhubSource(request_delay_s=0.0).fetch("AAPL", "2024-01-01", "2024-02-01")


def test_finnhub_error_body_raises_clear_message(monkeypatch) -> None:
    # Finnhub returns {"error": ...} with HTTP 200 on a plan/auth problem; the
    # adapter must raise a clear RuntimeError, not iterate the dict's keys.
    import news_sentiment.datasources.finnhub as mod

    _patch_get(monkeypatch, mod, {"error": "You don't have access to this resource."})
    src = FinnhubSource(request_delay_s=0.0, api_key="test-key")
    with pytest.raises(RuntimeError, match="company-news API error"):
        src.fetch("AAPL", "2024-01-01", "2024-02-01")


# --------------------------------------------------------------------------- #
# Apewisdom                                                                    #
# --------------------------------------------------------------------------- #

_APEWISDOM_PAYLOAD = {
    "count": 2,
    "pages": 1,
    "current_page": 1,
    "results": [
        {
            "rank": 1,
            "ticker": "GME",
            "name": "GameStop",
            "mentions": 420,
            "upvotes": 1337,
            "mentions_24h_ago": 300,
            "sentiment": "1.2",
        },
        {"rank": 2, "ticker": "AMC", "mentions": 99},
    ],
}


def test_apewisdom_emits_current_snapshot(monkeypatch) -> None:
    import news_sentiment.datasources.apewisdom as mod

    _patch_get(monkeypatch, mod, _APEWISDOM_PAYLOAD)
    src = ApewisdomSource()
    # Wide window that includes "now" so the live snapshot survives the filter.
    items = src.fetch("GME", "2000-01-01", "2100-01-01")

    assert len(items) == 1
    it = items[0]
    assert it.source == "apewisdom"
    assert it.ticker == "GME"
    assert it.meta["mentions"] == 420
    assert it.meta["mentions_24h_ago"] == 300
    assert it.meta["snapshot"] is True


def test_apewisdom_live_snapshot_excluded_from_historical_window(monkeypatch) -> None:
    """The live snapshot must NOT appear inside a past backtest window.

    This is the honest behaviour the plan relies on: Apewisdom has no history,
    so a historical window returns nothing rather than fabricating a past row.
    """
    import news_sentiment.datasources.apewisdom as mod

    _patch_get(monkeypatch, mod, _APEWISDOM_PAYLOAD)
    items = ApewisdomSource().fetch("GME", "2020-01-01", "2020-02-01")
    assert items == []


def test_apewisdom_ticker_not_found(monkeypatch) -> None:
    import news_sentiment.datasources.apewisdom as mod

    _patch_get(monkeypatch, mod, _APEWISDOM_PAYLOAD)
    items = ApewisdomSource().fetch("TSLA", "2000-01-01", "2100-01-01")
    assert items == []


# --------------------------------------------------------------------------- #
# EODHD                                                                        #
# --------------------------------------------------------------------------- #

# Recorded /api/news shape: array; `date` is ISO 8601 with a tz offset.
_EODHD_PAYLOAD = [
    {
        "date": "2023-06-15T13:30:00+00:00",
        "title": "TSLA opens new gigafactory",
        "content": "Tesla expands production capacity.",
        "link": "https://example.com/tsla-giga",
        "symbols": ["TSLA.US"],
        "tags": ["product"],
        "sentiment": {"polarity": 0.6, "neg": 0.1, "neu": 0.3, "pos": 0.6},
    },
    {
        "date": "2023-07-20T09:00:00+00:00",
        "title": "TSLA recalls vehicles",
        "content": "Safety recall announced.",
        "link": "https://example.com/tsla-recall",
        "symbols": ["TSLA.US"],
        "sentiment": {"polarity": -0.4, "neg": 0.5, "neu": 0.4, "pos": 0.1},
    },
]


def test_eodhd_maps_fields_and_keeps_sentiment_as_metadata(monkeypatch) -> None:
    import news_sentiment.datasources.eodhd as mod

    cap: dict = {}
    _patch_get(monkeypatch, mod, _EODHD_PAYLOAD, cap)
    src = EodhdSource(api_token="test-token")
    items = src.fetch("TSLA", "2023-01-01", "2024-01-01")

    assert len(items) == 2
    it = items[0]
    assert it.source == "eodhd"
    assert it.published_at == datetime(2023, 6, 15, 13, 30, 0)  # tz stripped to naive
    assert it.title == "TSLA opens new gigafactory"
    # EODHD's own sentiment is retained as a validation baseline only.
    assert it.meta["eodhd_sentiment"]["polarity"] == 0.6
    # symbol assembled with exchange suffix
    assert cap["params"]["s"] == "TSLA.US"
    assert cap["params"]["api_token"] == "test-token"


def test_eodhd_missing_token_raises(monkeypatch) -> None:
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="EODHD_API_TOKEN"):
        EodhdSource().fetch("TSLA", "2023-01-01", "2024-01-01")


def test_eodhd_error_body_raises_clear_message(monkeypatch) -> None:
    # EODHD returns {"error": "Token is wrong, you have no access to this API."}
    # with HTTP 200; the adapter must surface that, not crash on a dict.
    import news_sentiment.datasources.eodhd as mod

    _patch_get(
        monkeypatch,
        mod,
        {"s": "AAPL.US", "error": "Token is wrong, you have no access to this API."},
    )
    src = EodhdSource(api_token="bad-token")
    with pytest.raises(RuntimeError, match="news API error"):
        src.fetch("AAPL", "2023-01-01", "2024-01-01")


def test_eodhd_date_parser_handles_plain_date() -> None:
    from news_sentiment.datasources.eodhd import _parse_eodhd_date

    assert _parse_eodhd_date("2023-06-15") == datetime(2023, 6, 15)
    assert _parse_eodhd_date("") is None
    assert _parse_eodhd_date(None) is None


# --------------------------------------------------------------------------- #
# Hacker News                                                                  #
# --------------------------------------------------------------------------- #

_HN_PAYLOAD = {
    "hits": [
        {
            "objectID": "900",
            "created_at_i": 1705320000,  # 2024-01-15 12:00:00 UTC
            "title": "Show HN: NVDA GPU benchmark tool",
            "url": "https://example.com/nvda-bench",
            "points": 240,
            "num_comments": 51,
            "author": "dev",
        },
        {"objectID": "901", "created_at_i": 0, "title": "dropped (no ts)"},
    ]
}


def test_hn_maps_fields(monkeypatch) -> None:
    import news_sentiment.datasources.hn as mod

    _patch_get(monkeypatch, mod, _HN_PAYLOAD)
    items = HackerNewsSource().fetch("NVDA", "2024-01-01", "2024-02-01")
    assert len(items) == 1
    it = items[0]
    assert it.source == "hn"
    assert it.published_at == datetime(2024, 1, 15, 12, 0, 0)
    assert it.meta["points"] == 240
