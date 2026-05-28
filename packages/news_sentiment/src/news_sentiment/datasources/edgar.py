"""SEC EDGAR full-text + 8-K adapter.

Free, no API key, but SEC requires a descriptive ``User-Agent`` header
(<contact>) on every request — see https://www.sec.gov/os/accessing-edgar-data.
Rate limited to ~10 req/sec; the adapter sleeps conservatively between pages.

EDGAR is the "gem for event-driven" per the research doc: full archive,
point-in-time clean. Filings are reported at their submission timestamp,
which is the true publish time.

PR-A note: the adapter is structurally complete but does not exercise the
network in CI. The collector and lookahead test use the synthetic source.
"""

from __future__ import annotations

import time
from datetime import datetime

import requests

from news_sentiment.datasources.base import BaseNewsSource
from news_sentiment.types import NewsItem

_USER_AGENT_DEFAULT = "lidr-models research crawler (contact: research@example.invalid)"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"


class EdgarSource(BaseNewsSource):
    name = "edgar"

    def __init__(
        self,
        forms: tuple[str, ...] = ("8-K",),
        user_agent: str | None = None,
        request_delay_s: float = 0.2,
    ) -> None:
        self.forms = tuple(forms)
        self.user_agent = user_agent or _USER_AGENT_DEFAULT
        self.request_delay_s = float(request_delay_s)

    def fetch_raw(self, ticker: str, start: datetime, end: datetime) -> list[NewsItem]:
        # EDGAR's full-text search is paginated; this is the structural skeleton.
        # PR-A doesn't exercise it in CI; PR-B will harden pagination + retries.
        items: list[NewsItem] = []
        params = {
            "q": ticker,
            "dateRange": "custom",
            "startdt": start.strftime("%Y-%m-%d"),
            "enddt": end.strftime("%Y-%m-%d"),
            "forms": ",".join(self.forms),
        }
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        resp = requests.get(_EDGAR_SEARCH, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        hits = payload.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {}) or {}
            filed = src.get("file_date") or src.get("filing_date")
            if not filed:
                continue
            try:
                published = datetime.strptime(filed, "%Y-%m-%d")
            except ValueError:
                continue
            form = src.get("form", "")
            title = f"{ticker} files {form}: {src.get('display_names', [''])[0]}".strip()
            accession = src.get("adsh", "")
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}"
            items.append(
                NewsItem(
                    ticker=ticker,
                    published_at=published,
                    source=self.name,
                    title=title,
                    body=src.get("description", "") or "",
                    url=url,
                    meta={"form": form, "accession": accession},
                )
            )
        time.sleep(self.request_delay_s)
        return items
