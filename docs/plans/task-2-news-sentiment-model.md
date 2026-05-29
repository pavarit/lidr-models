# Task 2 — Build the news-sentiment model

> **For:** Claude Code sessions, after PR-A merged (commit `b9ce76a`, 2026-05-28). **Plan revised 2026-05-28** after credential setup found that Reddit and Tiingo as originally planned don't work for our use case — see [`../research/data-sources.md`](../research/data-sources.md) for the validated picture.
> **Read first:** `CLAUDE.md`, [`../adr/0001-multi-model-repo-architecture.md`](../adr/0001-multi-model-repo-architecture.md) (incl. its "The artifact contract (schema v2)" section), [`../research/data-sources.md`](../research/data-sources.md).

## Goal (unchanged)

Fill in the `news_sentiment` package: a standalone model that turns recent news / internet discussion about **high-news individual stocks** into a BUY / HOLD / SELL recommendation, scored by the same `lidr_core` harness as `ta_ensemble` so the two compete apples-to-apples. The hypothesis is a **synthesis/attention edge** (recognizing shifts in aggregate sentiment, attention spikes, and post-event drift), **not** a latency edge — we will always be late to a single headline, so don't optimize for speed.

## Design priorities (from Boon)

1. **Swappable data sources.** Adding or swapping a source must be one adapter + one config line. No source hard-coded into the model.
2. **Easy iterate-and-compare loop.** Running a new version and comparing it against prior versions and against `ta_ensemble` must be trivial — lean on `results_log` + `manifest.json` from `lidr_core`.
3. **Swappable features and model.** Features and learner are config-selected, like the signals already are.
4. **LLM that works, with cost control.** Use the FinBERT + LLM hybrid, but cap and cache LLM spend.

## Revised data sources (2026-05-28)

Two originals broke, one paid pick swapped — full reasoning in `docs/research/data-sources.md`.

| Source | In/out | Role | Cost |
|---|---|---|---|
| **SEC EDGAR** | In (PR-A, real) | Event-driven backbone | Free |
| **GDELT** | In (PR-A, real) | Aggregate news volume + tone | Free |
| **Finnhub (free tier)** | **In (NEW — PR-B adapter)** | Live US company news, 1yr history + real-time | Free (personal-use) |
| **Apewisdom** | **In (NEW — PR-B adapter)** | Live retail-attention (replaces Reddit's role) | Free |
| **EODHD News + Calendar** | **In (NEW — PR-B adapter)** | Historical news + per-article sentiment (replaces Tiingo's role) | **$19.99/mo** |
| **Hacker News** (Algolia) | Optional later | Tech-stock discussion supplement | Free |
| **Quiver Quantitative** (Hobbyist) | Optional later | Historical WSB mention data (back to Aug 2018) | $30/mo |
| Reddit (PRAW) | **Out** — permanent stub | Blocked by Responsible Builder Policy | — |
| Google Trends (pytrends) | **Out** — permanent stub | pytrends archived 2025-04-17 | — |
| Tiingo News | **Out** — delete stub | $30/mo for only 3mo news history at Power tier; doesn't buy what we needed | — |

**Net cost:** $19.99/mo for EODHD (vs the originally-budgeted $10/mo, vs the $30/mo Tiingo trap we avoided). Quiver Quant $30/mo is opt-in if/when news-side shows promise.

## Sequencing

**Horizon-spike first (Next Up #1, free, ~1–2 days), then revised PR-B.** The 5d-vs-20d target-noise question shapes `news_v0.yaml`'s `target.horizon_days` directly — if 20d helps meaningfully on TA-only, news features should be designed against a 20d target from the start. Cheap to settle, expensive to retrofit. CLAUDE.md Active Task carries the horizon-spike kickoff prompt; revised PR-B runs after.

## Revised PR-B scope (what Claude Code does next)

PR-A shipped the scaffolding + free adapters + lexicon scorer + offline dev pipeline. PR-B's job is now:

**Prereqs (set before kickoff).** Persist these as env vars (Windows: `[Environment]::SetEnvironmentVariable("VAR","val","User")`): `EODHD_API_TOKEN`, `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY`. Apewisdom needs no auth. The `[scoring]` extra installs FinBERT's `transformers + torch` (≈440MB download on first use).

**Data sources (rewire).** The three new adapters (Finnhub / Apewisdom / EODHD) are all HTTP-only and use only `requests`, which is already a base dep — **do not add per-adapter extras** for them. Pyproject extras change is just **drop `[reddit]` and `[trends]`**; keep `[scoring]` and `[llm]` for FinBERT and the Anthropic SDK respectively.

- **Delete** `packages/news_sentiment/src/news_sentiment/datasources/tiingo.py` and unregister `"tiingo"` from the source registry.
- **Convert to permanent stubs** (raise `NotImplementedError` with a one-line reason + pointer to `docs/research/data-sources.md` for context; mirror the existing PR-A stub pattern):
  - `reddit.py` — message: *"Reddit blocked by Responsible Builder Policy; see docs/research/data-sources.md. Use Apewisdom for live retail-attention; Quiver Quant for historical WSB."*
  - `google_trends.py` — message: *"pytrends archived 2025-04-17; see docs/research/data-sources.md. Apewisdom covers retail-attention more directly."*
- **Add new adapters** (real implementations, each subclassing `BaseNewsSource`, lazy-importing any heavy deps, with one recorded-fixture integration test like the other PR-B work):
  - `finnhub.py` — `/company-news?symbol=...&from=...&to=...` against `FINNHUB_API_KEY`. Free tier 60 calls/min. Map response fields into `NewsItem` (use `datetime` field as `published_at`).
  - `apewisdom.py` — `apewisdom.io/api/v1.0/filter/{filter}/page/{n}`, no auth. Live snapshot only — emit synthetic per-day `NewsItem`s for current rankings so the collector can forward-collect them into history. Filter defaults to `all-stocks`; surface as a config field.
  - `eodhd.py` — `eodhd.com/api/news?s={ticker}.US&from=...&to=...` against `EODHD_API_TOKEN`. Each request = 5 API calls — surface the rate-limit math in adapter docstring. Use article `sentiment` block as a validation baseline (not as model ground truth); we still score ourselves.
  - (Optional, time permitting) `hn.py` — Algolia `/search?query=...&tags=story&numericFilters=created_at_i>=...`. Tech-skewed, free.

**Scoring (fill the stubs as PR-A planned).**
- `finbert.py` — wire the local FinBERT model (440MB download via `transformers`), bulk-score `NewsItem.text`, emit a sentiment score in `[-1, 1]`. PR-A's `_common.items_to_daily` already aggregates.
- `llm.py` — live Anthropic call into the existing PR-A cache + budget cap + spend-log harness (no plumbing changes; just drop the live call in). Per-run cap defaults to the workspace's $5/mo ceiling. Tiered routing: LLM only when FinBERT confidence is low or relevance is uncertain.

**Tests.** Recorded-fixture integration tests for each new adapter (no live API quota burned in CI). Extend the lookahead test to cover any new feature signatures emerging from richer scoring outputs. Keep the smoke path on `dev.yaml` (synthetic source) running offline.

**Definition of done for PR-B.** All four free adapters (EDGAR, GDELT, Finnhub, Apewisdom) and the one paid adapter (EODHD) are real; FinBERT scoring runs end-to-end; live LLM call works through the cost-control harness; `tiingo.py` is gone, `reddit.py` and `google_trends.py` are permanent stubs; pyproject extras reflect the new shape; integration tests green; smoke test still passes offline. **EODHD timestamp sanity-check:** pull ~20 EODHD articles spanning a few months for a real ticker, spot-check `published_at` against the source URL (or web archive) for 1–2 of them, and record the finding in the PR description — this is the cheap version of validating that EODHD's timestamps aren't backfilled. **No backtest claim** — that's PR-C.

## PR-C (unchanged in spirit, updated stack)

Run `news_v0.yaml` on a 5–10 high-news ticker universe with the rewired data stack, the FinBERT/LLM hybrid scoring, and `target.horizon_days` set to whatever the horizon-spike outcome supports. Append a `results_log` row (`model_id: news_sentiment`), regenerate `manifest.json`, produce a comparison (chart + per-period table) of news_sentiment vs ta_ensemble vs buy-and-hold per the outcome-changing-PR evidence convention.

**Apewisdom features are excluded from `news_v0.yaml`'s first backtest.** Apewisdom is live-snapshot only, so at PR-C run-time (immediately after PR-B merges) the forward-collected store will have essentially zero history — any feature derived from it would have a degenerate timeseries. The adapter still runs in PR-B to start the data clock; Apewisdom-derived features get included in a later `news_v0.1.yaml` iteration once enough history has accumulated (months, not days). PR-C's active feature set draws only from EDGAR / GDELT / Finnhub / EODHD, all of which have real history at run-time.

Honest verdict on edge — negative result clearly shown is a valid outcome. **Delete this plan doc** (`docs/plans/task-2-news-sentiment-model.md`) as the cleanup commit when the PR-C checkpoint merges. The ADR and `docs/research/data-sources.md` stay as the durable record.

## Watch out for

- **Lookahead / backfill leakage** — the #1 way sentiment backtests lie. Validate EODHD article timestamps are true publish times (the cheap-tier providers don't guarantee point-in-time correctness; see research doc). PR-A's `align_to_trading_days` one-trading-day shift is the enforcement mechanism on the feature side.
- **Survivorship bias** — worse for high-news individual names (blown-up meme stocks delist and vanish). The existing yfinance gotcha applies harder here.
- **Don't chase latency.** If results only look good with intraday timing, that's not a free-data-realistic edge.
- **EODHD rate-limit math.** Each news request costs 5 API calls. Surface this in the adapter docstring and respect the plan-tier daily quota; tests that burn live calls will exhaust quota fast.

---

## Claude Code kickoff prompt (revised PR-B — run AFTER the horizon-spike PR)

```
Read CLAUDE.md, then docs/adr/0001-multi-model-repo-architecture.md
(including its "The artifact contract (schema v2)" section),
docs/research/data-sources.md, and docs/plans/task-2-news-sentiment-model.md.

Execute the REVISED PR-B (post-2026-05-28 plan revision) for the news_sentiment
package. The data-source landscape changed since PR-A: Reddit is blocked by
Responsible Builder Policy, pytrends is archived, Tiingo at $30/mo only gives 3
months of news history. Full reasoning in docs/research/data-sources.md.

Prereqs (must be set as env vars before kickoff):
   EODHD_API_TOKEN, FINNHUB_API_KEY, ANTHROPIC_API_KEY
Apewisdom needs no auth.

What this PR does:

1. Data-source rewire:
   - Delete packages/news_sentiment/src/news_sentiment/datasources/tiingo.py and
     unregister "tiingo" from the source registry.
   - Convert reddit.py and google_trends.py to PERMANENT stubs that raise
     NotImplementedError with a one-line reason pointing at docs/research/data-sources.md
     (mirror PR-A's stub pattern; don't delete — keep the file as documentation).
   - Add real adapters (each subclassing BaseNewsSource with a recorded-fixture
     integration test):
       * finnhub.py    (free, personal-use, FINNHUB_API_KEY, /company-news)
       * apewisdom.py  (free, no auth, live-snapshot — emit synthetic per-day
                       NewsItems so the collector forward-collects history)
       * eodhd.py      (paid $19.99/mo, EODHD_API_TOKEN, /api/news; each request
                       = 5 API calls — respect daily quota)
       * (optional) hn.py  (Algolia API, free, tech-skewed supplement)
   - Pyproject extras change: DROP [reddit] and [trends]; keep [scoring] and
     [llm]. The three new adapters (Finnhub, Apewisdom, EODHD) are HTTP-only and
     use only `requests` (already in base) — DO NOT add per-adapter extras for
     them. No empty extras for symmetry.

2. Scoring (drop live calls into PR-A's existing cost-control harness):
   - finbert.py — live FinBERT scoring via transformers (440MB download).
   - llm.py — live Anthropic call wired through the existing cache + budget cap
     + spend-log scaffolding from PR-A. Default per-run cap matches workspace
     $5/mo ceiling. Tier: LLM only when FinBERT confidence is low or relevance
     is uncertain.

3. Tests: recorded-fixture integration test per new adapter; lookahead test still
   passes; smoke (dev.yaml) still runs fully offline.

4. EODHD timestamp sanity-check: pull ~20 EODHD articles spanning a few months
   for a real ticker, spot-check `published_at` against the source URL (or web
   archive) for 1-2 of them, and record the finding in the PR description.
   Cheap way to validate the timestamps aren't backfilled before relying on
   them in PR-C's backtest.

DO NOT add Quiver Quantitative in this PR — it's an opt-in $30/mo add-on for
backtesting retail-attention features that we defer until news-side shows promise.

DoD for PR-B: all five rewired adapters real, two adapters as permanent stubs,
tiingo.py gone, FinBERT scoring works, live LLM works through cost controls,
integration tests green, smoke test still passes offline, EODHD timestamp
spot-check recorded in PR description. NO backtest claim in this PR — that's
PR-C. Note for PR-C: Apewisdom features will be excluded from news_v0.yaml's
first config because the forward-collected store has no history yet.

Follow protected-main PR workflow; CI green is required. Do not start PR-C.
```
