"""Collect Bank of Canada publications from the public sitemap.

Polite by design: reads robots.txt and skips anything it disallows, waits between
requests (`SCRAPE_DELAY_SECONDS`), sends an identifying User-Agent, and caches every page
under data/cache/ so re-runs do not hit the site again. Raw pages are never committed.

Document types collected (from 2015, `SCRAPE_FROM_YEAR`):
  * announcement       - interest rate announcement press releases (fad-press-release-...)
  * opening_statement  - Governor's opening statements (press conferences, committees)
  * deliberations      - summaries of Governing Council deliberations (from 2023)
  * mpr                - Monetary Policy Report overview pages (HTML from July 2024)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from copilot.config import get_settings

log = logging.getLogger("ingest.scrape")
BASE = "https://www.bankofcanada.ca"
SITEMAPS = [f"{BASE}/wp-sitemap-posts-post-{i}.xml" for i in range(1, 5)] + [
    f"{BASE}/wp-sitemap-posts-page-1.xml"
]
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("announcement", re.compile(r"/(\d{4})/\d{2}/fad-press-release-\d{4}-\d{2}-\d{2}/$")),
    ("opening_statement", re.compile(r"/(\d{4})/\d{2}/opening-statement-[^/]+/$")),
    ("deliberations", re.compile(r"/(\d{4})/\d{2}/(?:publication-)?summary-(?:of-deliberations|governing-council-deliberations)[^/]*/$")),
    ("mpr", re.compile(r"/publications/mpr/mpr-(\d{4})-\d{2}(?:-\d{2})?/$")),
]  # fmt: skip
MIN_WORDS = 120  # pages shorter than this are schedule notices, not content


@dataclass
class Doc:
    """One publication with the metadata every chunk inherits."""

    doc_id: str
    url: str
    doc_type: str
    title: str
    date: str  # ISO yyyy-mm-dd
    text: str


def classify_url(url: str, from_year: int) -> str | None:
    """Return the document type for a sitemap URL, or None if it is out of scope."""
    for doc_type, pattern in PATTERNS:
        m = pattern.search(url)
        if m and int(m.group(1)) >= from_year:
            return doc_type
    return None


def doc_id_for(url: str) -> str:
    """Stable short id from the URL."""
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def date_from_url(url: str) -> str | None:
    """Best date from the URL: a full yyyy-mm-dd in the slug, else the yyyy/mm folder."""
    if m := re.search(r"(\d{4})-(\d{2})-(\d{2})/?$", url.rstrip("/") + "/"):
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if m := re.search(r"/(\d{4})/(\d{2})/", url):
        return f"{m.group(1)}-{m.group(2)}-01"
    return None


def parse_page(html: str, url: str, doc_type: str) -> Doc | None:
    """Extract title, date and main text from a page; None if it has no real content."""
    extracted = trafilatura.bare_extraction(html, url=url, include_comments=False,
                                            include_tables=False, with_metadata=True)  # fmt: skip
    if extracted is None:
        return None
    data = extracted.as_dict() if hasattr(extracted, "as_dict") else dict(extracted)
    text = (data.get("text") or "").strip()
    if len(text.split()) < MIN_WORDS:
        return None
    title = (data.get("title") or "").strip() or url.rstrip("/").split("/")[-1]
    page_date = data.get("date") or date_from_url(url)
    if not page_date or page_date > date.today().isoformat():
        return None  # future-dated schedule entries
    return Doc(doc_id=doc_id_for(url), url=url, doc_type=doc_type, title=title,
               date=page_date[:10], text=text)  # fmt: skip


class PoliteClient:
    """HTTP client with robots.txt checks, a fixed delay and an on-disk cache."""

    def __init__(self, cache_dir: Path, delay: float, user_agent: str) -> None:
        """Create the client; robots.txt is read once."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.client = httpx.Client(
            headers={"User-Agent": user_agent}, follow_redirects=True, timeout=60
        )
        self.robots = RobotFileParser(f"{BASE}/robots.txt")
        self.robots.parse(self.client.get(f"{BASE}/robots.txt").text.splitlines())
        self.user_agent = user_agent
        self.fetched = 0

    def get(self, url: str) -> str | None:
        """Return the page body from cache or the network; None if disallowed or missing."""
        path = self.cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        if not self.robots.can_fetch(self.user_agent, url):
            log.warning("disallowed by robots.txt", extra={"url": url})
            return None
        time.sleep(self.delay)
        r = self.client.get(url)
        self.fetched += 1
        if r.status_code != 200:
            log.warning("fetch failed", extra={"url": url, "status": r.status_code})
            return None
        path.write_text(r.text, encoding="utf-8")
        return r.text


def discover(client: PoliteClient, from_year: int) -> list[tuple[str, str]]:
    """(url, doc_type) pairs from the sitemaps."""
    found: dict[str, str] = {}
    for sitemap in SITEMAPS:
        body = client.get(sitemap) or ""
        for url in re.findall(r"<loc>(.*?)</loc>", body):
            if doc_type := classify_url(url, from_year):
                found[url] = doc_type
    return sorted(found.items())


def scrape(limit: int | None = None) -> list[Doc]:
    """Discover, fetch (politely) and extract all in-scope documents to data/docs.jsonl."""
    s = get_settings()
    client = PoliteClient(s.cache_dir, s.scrape_delay_seconds, s.user_agent)
    targets = discover(client, s.scrape_from_year)
    if limit:
        targets = targets[:limit]
    docs: list[Doc] = []
    for url, doc_type in targets:
        html = client.get(url)
        if html and (doc := parse_page(html, url, doc_type)):
            docs.append(doc)
    docs.sort(key=lambda d: (d.date, d.url))
    s.docs_path.parent.mkdir(parents=True, exist_ok=True)
    with s.docs_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    log.info(
        "scraped",
        extra={"targets": len(targets), "documents": len(docs), "network_fetches": client.fetched},
    )
    return docs


def load_docs(path: Path | None = None) -> list[Doc]:
    """Read data/docs.jsonl."""
    path = path or get_settings().docs_path
    with path.open(encoding="utf-8") as f:
        return [Doc(**json.loads(line)) for line in f if line.strip()]
