# Data sources for news / sentiment signals

> **Purpose.** Stored research so we don't re-derive it later, and a reference for when we revisit expanding the data sources. Captures free-tier status, paid pricing, and the leverage-per-dollar call.
> **Last verified:** 2026-05-28 (major revision after credential-setup found two breaking changes from the 2026-05-27 picture). API terms and pricing change — re-verify the starred (★) rows before committing, they're the most volatile.

## TL;DR

- The validated **free backbone** is **SEC EDGAR + GDELT + Finnhub (free tier) + Apewisdom**. Enough to run from collection through a first backtest at $0.
- The binding constraint is **historical data for backtesting**, not live collection. Live data is free and easy; aligned *history* is the scarce, paywalled thing.
- The single highest-leverage paid pick is **EODHD News + Calendar at $19.99/mo** — multi-year per-article news + sentiment scores. *This replaced Tiingo* after Tiingo turned out to be $30/mo with only 3 months of queryable news history at that tier.
- **Reddit (PRAW) is effectively blocked** for our use case — Reddit's Responsible Builder Policy requires moderation or accredited-academic justification, neither of which fits a personal-research market-signal collector. Apewisdom replaces the live role; Quiver Quantitative ($30/mo, optional later) replaces the historical role.
- **pytrends is archived** (Apr 2025); Google Trends is dropped from the plan rather than replaced.
- Everything above ~$50/mo solves problems we don't have yet. RavenPack-class data is institutional overkill until the cheap stack shows a hint of edge.

## Free tiers — verified 2026-05-28

| Source | Status | Limits / catch | Usable history for backtest? |
|---|---|---|---|
| **SEC EDGAR** full-text + 8-K | ✅ Free, no key | 10 req/sec; must send a `User-Agent` header | **Yes** — full archive, point-in-time clean. The gem for event-driven. |
| **GDELT** (news volume + tone) | ✅ Free | Updates every 15 min; DOC 2.0 API + free BigQuery; ElasticSearch-protective rate limits but generous in practice | **Yes** — deep history, free. The workhorse for aggregate news volume/tone. |
| **Finnhub** (free tier) | ✅ Free, personal-use license | 60 calls/min; **1 year of US company news + real-time updates**; SEC filings, basic fundamentals | **Yes, 1yr** — solid for a starting backtest on US tickers. Premium needed for international or longer history. |
| **Apewisdom** | ✅ Free, no auth | Live snapshot only via API (current ranking + 24h-ago); 100 results/page; covers WSB / r/stocks / r/investing / r/options / r/SPACs / 4chan /biz | **Live only** — forward-collect to build our own historical store. Apewisdom themselves run an aggregator with deeper internal history but the public API exposes the snapshot. |
| **Hacker News** (Algolia API) | ✅ Free | None meaningful | Yes — full history, but tech-skewed coverage. Optional supplement for mega-cap tech names. |
| ★ **Reddit** API / PRAW | ❌ **BLOCKED** for our case | Responsible Builder Policy (updated 2026-05-18) requires explicit approval; moderator access is "solely for performing moderation actions"; commercial use requires written approval; research access gated to "academic researchers affiliated with an accredited university" | n/a |
| ★ **Google Trends** (pytrends) | ❌ **BROKEN** | pytrends repo **archived 2025-04-17**, no longer maintained; 429 errors at modest volume; community `pytrends-modern` fork has same unofficial-API fragility | n/a |
| ★ **StockTwits** | ❌ Closed | Not accepting new API registrations (under review as of 2026-05) | n/a |
| **Alpha Vantage** `NEWS_SENTIMENT` | Technically free | **25 requests/day** — too thin to be useful | Some, but the cap kills it |
| **NewsAPI.org** | Dev only | 100 req/day, 24h delay, **no production use** | No |

## Paid options — verified pricing & what they unlock

| Provider / tier | $/mo | Historical news | Pre-computed sentiment | Ticker/entity tagging | Verdict for our case |
|---|---|---|---|---|---|
| **EODHD News + Calendar** | **$19.99** | **Yes, date-filterable** (examples span 2022+, multi-year) | **Yes** — per-article (polarity / neg / neu / pos) + daily aggregate sentiment + word-weights endpoint | Yes — `symbols[]` array + 50 standard tags + AI-detected tags | **Primary paid pick — buy to trial** |
| Stock News API — Premium | $49.99 | Yes (depth unspecified) | Yes, plus top mentions / trending / upgrades-downgrades | Yes | Solid backup if EODHD disappoints |
| Stock News API — Basic | $19.99 | No (history excluded) | Yes | Yes | Skip — no history + non-commercial only |
| Marketaux — Basic / Standard | $24–49 (annual $24/$41) | Yes (depth unspecified) | Yes | Strong entity tagging; global + multi-lang | Strong if non-US/multi-lang matters |
| **Quiver Quantitative — Hobbyist** | **$30** | **Yes, WSB mentions back to Aug 2018, 6,000 equities** | Yes (Quiver's own sentiment) | Yes — by ticker | **Optional later — adds historical WSB so retail-attention features become backtestable** |
| Quiver Quantitative — Trader | $75 | Tier 1 + Tier 2 datasets | Yes | Yes | Overkill unless we add more alt-data feeds |
| ★ **Tiingo — Power** | **$30** | **Only 3 months queryable** at this tier (15-yr archive sales-quoted enterprise) | No (you score it) | Yes — ticker, slang, company/product mentions | **Skip for news alone** — was originally the headline pick at "$10/mo with ~10yr history"; both numbers turned out wrong |
| Finnhub — Premium | $12–100 | Yes; US-only company news + sentiment endpoint | Yes (premium endpoint) | Yes | Marginal over free tier for our use case |
| EODHD — All-in-One | ~$108 (€99.99) | Yes + bundled prices + fundamentals + crypto | Yes | Yes | Only if consolidating vendors |
| Polygon.io — Starter | $29 | Yes (bundled into broad market-data API) | No | Yes | Overkill — news is a side feature |
| Alpha Vantage — Premium | $50–250 | Yes | Yes (AI-powered) | Yes | Pricey for the news alone |
| Benzinga | custom quote | Yes | Yes | Yes | Enterprise — likely $1k+/mo |
| RavenPack / Bigdata.com | contact sales | Yes (with point-in-time guarantees) | Yes | Yes | Gold standard; institutional, tens of $k/yr; overkill until cheap stack shows edge |

## The point-in-time caveat (still important)

The cheap providers (EODHD, Marketaux, Stock News API) generally do **not** guarantee strict *point-in-time* correctness — i.e. that a sentiment score or article timestamp wasn't quietly revised/backfilled after the fact. That backfill risk is exactly what RavenPack charges a fortune to eliminate. At the $19.99 tier we get ~90% of the way there, but we must **validate ourselves that timestamps are true publish times**, or the backtest will look better than reality. The same lookahead-bias trap that `packages/ta_ensemble/tests/test_no_lookahead.py` guards on the signal side has to extend to news ingestion — this is enforced in PR-A's `align_to_trading_days` (one-trading-day shift) and the parametrized news-feature lookahead test.

## Decision log

- **2026-05-28** — Revised the plan after credential setup surfaced two breaking changes from the original. **(1) Reddit (PRAW) → out**: Responsible Builder Policy blocks our use case; replaced live role with **Apewisdom (free)**, historical role optionally with **Quiver Quantitative Hobbyist ($30/mo)** which is licensed to redistribute WSB data. **(2) Tiingo → out**: Power tier is $30/mo (3× original $10 assumption) and only exposes 3 months of news history at that tier — fails the historical-backtest thesis. **(3) pytrends → out**: archived 2025-04-17; dropping Google Trends from the plan since Apewisdom covers retail-attention more directly. **(4) Paid pick swap**: **EODHD News + Calendar at $19.99/mo** becomes the historical-news source (multi-year history + per-article sentiment scores included). **(5) New free entry**: Finnhub free tier promoted to backbone (1yr US company news + real-time at 60/min). **Sequencing**: do the **Next Up #1 horizon spike** on TA model first (free, ~1–2 days) to settle the 5d-vs-20d target-noise question before designing `news_v0.yaml`, then revised PR-B with the new stack.
- **2026-05-27** — Original plan: free backbone EDGAR + GDELT + Reddit + Google Trends, paid Tiingo News (~$10/mo) for historical depth. *Superseded by the 2026-05-28 revision above.*

## Revisit / expansion ideas (future)

- **If edge appears:** evaluate a point-in-time provider (RavenPack / Bigdata.com) for a clean re-run, to confirm the edge isn't a backfill artifact.
- **Retail-attention historical depth:** add **Quiver Quantitative Hobbyist ($30/mo)** if the news-side hypothesis lands and you want to backtest the retail-attention features properly instead of forward-collecting Apewisdom for months.
- **Reddit, if the policy reopens:** the Responsible Builder Policy is recent (2024); if it loosens for non-commercial research, PRAW would re-enter the free backbone.
- **Search-interest signals:** if a non-investor attention proxy ever becomes important, SerpAPI's Trends endpoint (~$50+/mo) or Glimpse are paid options now that pytrends is dead.
- **Breadth on news:** Marketaux ($24–49/mo) for global multi-language coverage, or Stock News API Premium ($49.99) for richer event tagging — both reasonable round-2 candidates if EODHD coverage is too narrow on mid-cap names.
- **Pre-computed vs self-scored:** we default to self-scoring (FinBERT/LLM) for control; revisit buying pre-computed sentiment if scoring cost/latency becomes the bottleneck.

## Sources

- **Reddit Responsible Builder Policy** — support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy (verified directly, updated 2026-05-18)
- **Tiingo pricing** — tiingo.com/about/pricing (verified directly: Power $30/mo, News API = 3mo queryable at this tier, 15yr archive via sales)
- **EODHD pricing + News API** — eodhd.com/lp/calendar-and-news-api · eodhd.com/financial-apis/stock-market-financial-news-api (verified directly)
- **Apewisdom API** — apewisdom.io/api/ (verified directly: free, no auth, live snapshot)
- **Quiver Quantitative** — quiverquant.com · quiverquant.com/wallstreetbets/ · github.com/Quiver-Quantitative/python-api (Hobbyist $30/mo; WSB dataset from Aug 2018, 6K equities)
- **pytrends archived** — github.com/GeneralMills/pytrends (archived 2025-04-17)
- **Finnhub free tier** — finnhub.io/pricing (verified directly: 60 calls/min, 1yr company news, personal-use license)
- **SEC EDGAR** — sec.gov accessing-edgar-data · tldrfiling.com/blog/sec-edgar-api-rate-limits-best-practices (10 req/sec, User-Agent required)
- **GDELT** — gdeltproject.org · blog.gdeltproject.org (DOC 2.0 API)
- **StockTwits** — api.stocktwits.com/developers (still closed to new registrations)
- **NewsAPI / Alpha Vantage / Marketaux / Polygon / Stock News API / Benzinga / RavenPack** — newsapi.org/pricing · alphavantage.co/premium · marketaux.com/pricing · polygon.io · stocknewsapi.com/pricing · benzinga.com/apis · ravenpack.com
