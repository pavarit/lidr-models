# Data sources for news / sentiment signals

> **Purpose.** Stored research so we don't re-derive it later, and a reference for when we revisit expanding the data sources. Captures free-tier status (verified May 2026), paid pricing, and the leverage-per-dollar call.
> **Last verified:** 2026-05-27. API terms and pricing change — re-verify the starred (★) rows before committing, they're the most volatile.

## TL;DR

- The genuinely useful **free** stack is **SEC EDGAR + GDELT + Reddit (live) + Google Trends**. Enough to run from collection through a first backtest at $0.
- The binding constraint is **historical data for backtesting**, not live collection. Live data is free and easy; aligned *history* is the scarce, paywalled thing.
- The cheapest item that removes that constraint is **Tiingo's News add-on at ~$10/month** (≈10 yr of ticker-tagged history). Decision: **pay for Tiingo to trial it.**
- Everything above ~$50/month solves problems we don't have yet. RavenPack-class data is institutional overkill until a $10 backtest shows a hint of edge.

## Free tiers — verified May 2026

| Source | Free? | Limits / catch | Usable history for backtest? |
|---|---|---|---|
| **SEC EDGAR** full-text + 8-K | Yes, no key | 10 req/sec; must send a `User-Agent` header | **Yes** — full archive, point-in-time clean. The gem for event-driven. |
| **GDELT** (news volume + tone) | Yes, fully | Updates every 15 min; DOC 2.0 API + free BigQuery | **Yes** — deep history, free. The backtest workhorse for news volume/tone. |
| ★ **Reddit** API / PRAW | Yes (non-commercial) | ~60–100 queries/min; commercial use needs approval (~$0.24/1k calls) | **Live only.** Historical firehose (Pushshift) lost its API in 2023 — no easy free history. |
| ★ **Google Trends** (pytrends) | Yes (unofficial) | Unofficial client; rate-limited / blocked if pushed hard | Yes — attention proxy, decent depth |
| **Hacker News** (Algolia API) | Yes, fully | None meaningful | Yes — full history, but tech-skewed coverage |
| ★ **Finnhub** | Yes | 60 calls/min; company *news* free, but the dedicated *sentiment* endpoint is now premium / US-only | Limited on free tier |
| **Alpha Vantage** `NEWS_SENTIMENT` | Technically | **25 requests/day** — too thin to be useful | Some, but the cap kills it |
| **NewsAPI.org** | Dev only | 100 req/day, **24h delay, no production use** | No |
| ★ **StockTwits** | **Closed** | Not accepting new API registrations (under review as of 2026-05) | n/a — was a candidate for bull/bear-tagged social; currently unavailable |

## Paid options — pricing & what they unlock

| Provider | Price/mo | What you get | Verdict for this project |
|---|---|---|---|
| **Tiingo** (News add-on, "Power" tier) | **~$10** | Ticker- & topic-tagged news, ~10 yr history, clean feed | **Best leverage — buy to trial** |
| Stock News API (stocknewsapi.com) | $19.99 | Pre-computed sentiment, US market, tagged | Optional convenience |
| EODHD (Calendar & News API) | $19.99 | News + daily per-ticker sentiment scores | Optional convenience |
| EODHD All-in-One | ~$108 (€99.99) | Above + prices + fundamentals bundle | Only if consolidating vendors |
| Marketaux | free → ~$25–200 | Entity tagging + sentiment built in; free 100 req/day | Decent free tier to trial |
| Finnhub premium | $12–100 | Higher limits + sentiment endpoint + alt-data | Marginal over free tier |
| Polygon.io | $29–199 | News bundled into broad market-data API | Overkill — don't need the breadth |
| Alpha Vantage premium | $50–250 | Lifts the 25/day cap; AI sentiment | Pricey for what it adds |
| Benzinga | custom quote | Institutional news feed | Enterprise — likely $1k+/mo |
| RavenPack / Bigdata.com | contact sales | Gold standard: point-in-time, entity-resolved, decades | Institutional — tens of $k/yr. Overkill for now. |

## The point-in-time caveat (important for any backtest)

The cheap providers (Tiingo, EODHD, Marketaux) generally do **not** guarantee strict *point-in-time* correctness — i.e. that a sentiment score or article timestamp wasn't quietly revised/backfilled after the fact. That backfill risk is exactly what RavenPack charges a fortune to eliminate. At the $10 tier we get ~90% of the way there, but we must **validate ourselves that timestamps are true publish times**, or the backtest will look better than reality. This is the same lookahead-bias trap that `tests/test_no_lookahead.py` guards on the signal side — it has to extend to news ingestion.

## Decision log

- **2026-05-27** — Start on the free backbone (EDGAR + GDELT + Reddit + Google Trends). Add **Tiingo News (~$10/mo)** as a paid trial to unlock historical backtest depth immediately rather than waiting months for a live collector to accumulate data. Hold off on all other paid sources until a backtest shows promise.

## Revisit / expansion ideas (future)

- **If edge appears:** evaluate a point-in-time provider (RavenPack/Bigdata) for a clean re-run, to confirm the edge isn't a backfill artifact.
- **Social depth:** re-check StockTwits once their API reopens; consider a paid Reddit historical archive if retail-attention features prove load-bearing.
- **Breadth:** Marketaux / EODHD for multi-source entity-tagged sentiment if single-source coverage is too thin on mid-cap names.
- **Pre-computed vs self-scored:** we default to self-scoring (FinBERT/LLM) for control; revisit buying pre-computed sentiment if scoring cost/latency becomes the bottleneck.

## Sources

- Reddit/Pushshift: octolens.com/blog/reddit-api-pricing · emergentmind.com (Pushshift dataset status)
- SEC EDGAR: tldrfiling.com/blog/sec-edgar-full-text-search-api · sec.gov accessing-edgar-data
- GDELT: gdeltproject.org · blog.gdeltproject.org (DOC 2.0 API)
- Alpha Vantage: alphavantage.co/premium · macroption.com/alpha-vantage-api-limits
- Finnhub: finnhub.io/pricing · finnhub.io/docs/api/rate-limit
- StockTwits: api.stocktwits.com/developers · NewsAPI: newsapi.org/pricing
- Tiingo: tiingo.com/products/news-api · tiingo.com/about/pricing
- Marketaux: marketaux.com/pricing · Polygon: polygon.io · Stock News API: stocknewsapi.com/pricing · EODHD: eodhd.com/financial-apis/stock-market-financial-news-api
- RavenPack: ravenpack.com · datarade.ai/data-providers/ravenpack/profile
