"""PR-B live verification — produces an inspectable log. Prints no API keys.

EODHD note: the API returns articles newest-first and its `to` bound is not
strictly enforced server-side, so a single page (max_pages=1) over a window
that ENDED in the past captures nothing (all of the newest page is after that
end and gets filtered by published_at < end). For the timestamp-authenticity
spot-check we therefore use a window whose end is just after now, which returns
the most-recent real articles to inspect. Reaching deep historical months needs
offset paging — a PR-C concern, flagged in the PR, not part of PR-B's DoD.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from news_sentiment.datasources.eodhd import EodhdSource
from news_sentiment.datasources.finnhub import FinnhubSource
from news_sentiment.scoring.llm import LlmScorer
from news_sentiment.types import NewsItem

LOG: list[str] = []


def log(*a):
    LOG.append(" ".join(str(x) for x in a))


def section(t):
    log("")
    log("=" * 72)
    log(t)
    log("=" * 72)


# ---------------- Finnhub ----------------
section("1. FINNHUB /company-news  (live, through the adapter)")
log(f"FINNHUB_API_KEY present: {bool(os.environ.get('FINNHUB_API_KEY'))}")
try:
    end = datetime.now()
    start = end - timedelta(days=30)
    items = FinnhubSource(request_delay_s=0.0).fetch(
        "AAPL", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )
    log(f"window: {start.date()} -> {end.date()}")
    log(f"items fetched: {len(items)}")
    if items:
        log(f"sorted ascending: {items == sorted(items, key=lambda i: i.published_at)}")
        log(f"all in window: {all(start <= i.published_at <= end for i in items)}")
        log("first 3 (published_at | title):")
        for it in items[:3]:
            log(f"  {it.published_at.isoformat()}  {it.title[:60]!r}")
except Exception as e:  # noqa: BLE001
    log(f"FINNHUB ERROR: {type(e).__name__}: {e}")


# ---------------- EODHD timestamp spot-check ----------------
section("2. EODHD /api/news  TIMESTAMP SPOT-CHECK (1 request = 5 API calls)")
log(f"EODHD_API_TOKEN present: {bool(os.environ.get('EODHD_API_TOKEN'))}")
log("note: EODHD returns newest-first; end set to tomorrow so the newest page")
log("      survives the published_at<end filter. These are recent real articles,")
log("      NOT spanning months (deep history needs offset paging — a PR-C item).")
try:
    e_end = datetime.now() + timedelta(days=1)
    e_start = e_end - timedelta(days=120)
    items = EodhdSource(limit=20, max_pages=1).fetch(
        "AAPL", e_start.strftime("%Y-%m-%d"), e_end.strftime("%Y-%m-%d")
    )
    log(f"window: {e_start.date()} -> {e_end.date()}")
    log(f"items fetched: {len(items)}")
    if items:
        log(f"all in window: {all(e_start <= i.published_at < e_end for i in items)}")
        log(f"sorted ascending: {items == sorted(items, key=lambda i: i.published_at)}")
        log(f"distinct timestamps: {len({i.published_at for i in items})} of {len(items)}")
        log("articles (published_at | url) — eyeball for backfill:")
        for it in items:
            log(f"  {it.published_at.isoformat()}  {it.url[:74]}")
        log(f"sample eodhd_sentiment (metadata baseline only): {items[0].meta.get('eodhd_sentiment')}")
except Exception as e:  # noqa: BLE001
    log(f"EODHD ERROR: {type(e).__name__}: {e}")


# ---------------- LLM live smoke ----------------
section("3. LIVE LLM smoke  (through the cost-control harness)")
log(f"ANTHROPIC_API_KEY present: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
try:
    spend = Path("_tmp_spend.csv")
    if spend.exists():
        spend.unlink()
    scorer = LlmScorer(model="claude-haiku-4-5", max_calls=2, max_usd=0.10,
                       spend_log_path=spend, run_id="pr_b_live_verify")
    pos = NewsItem(ticker="AAPL", published_at=datetime(2024, 1, 15, 12, 0, 0), source="verify",
                   title="Apple beats earnings expectations, raises full-year guidance")
    neg = NewsItem(ticker="AAPL", published_at=datetime(2024, 2, 1, 12, 0, 0), source="verify",
                   title="Apple faces DOJ antitrust lawsuit; shares tumble on weak iPhone sales")
    for label, item in [("BULLISH headline", pos), ("BEARISH headline", neg)]:
        s = scorer.score([item])[0]
        log(f"  {label}: scorer={s.scorer} sentiment={s.sentiment:+.3f} "
            f"relevance={s.relevance:.2f} confidence={s.confidence:.2f}")
    rem = scorer.budget_remaining()
    log(f"  budget remaining: {rem['calls_remaining']} calls, ${rem['usd_remaining']:.4f}")
    if spend.exists():
        rows = spend.read_text(encoding="utf-8").strip().splitlines()
        log(f"  spend log rows (incl header): {len(rows)}")
        for r in rows:
            log(f"    {r}")
except Exception as e:  # noqa: BLE001
    log(f"LLM ERROR: {type(e).__name__}: {e}")

Path("_tmp_verify_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("VERIFY DONE; lines=", len(LOG))
