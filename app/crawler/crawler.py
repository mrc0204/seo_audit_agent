"""Asynchronous web crawler for traversing pages within domain limits.

Breadth-first crawl of one origin, seeded from the homepage plus any sitemap URLs.
The public :meth:`Crawler.crawl` is synchronous — it owns an event loop internally
and fetches each depth level concurrently — so callers (and the CLI) stay plain
sequential code.

What this module deliberately does *not* do:

* **It does not parse HTML.** A :class:`FetchResult` carries raw bytes-as-text plus
  the transport facts. Turning that into a ``PageData`` is Phase 3's job. The one
  exception is link discovery, which needs enough of an ``href`` scan to build the
  frontier and is done with a narrow regex rather than a DOM parse.
* **It does not render JavaScript.** ``httpx`` retrieves the server's HTML and
  nothing else, so a client-rendered SPA yields a near-empty ``html`` and no
  discovered links. This is a known, documented limitation rather than a silent
  one: :attr:`CrawlReport.notes` records when a page looked JS-rendered so later
  phases can decline to report findings that are really just invisible markup.

Every skipped URL is recorded with a reason, so the crawl can explain itself.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.crawler.robots import DEFAULT_USER_AGENT, RobotsChecker
from app.crawler.sitemap import fetch_sitemap
from app.crawler.url_utils import (
    InvalidURLError,
    generate_dedup_key,
    is_same_domain,
    normalize_url,
)

#: Content types worth crawling. Anything else (PDF, image, video, archive) is
#: fetched only as far as its headers, then skipped — downloading a 200MB video to
#: discover it is not HTML wastes the entire crawl budget.
HTML_CONTENT_TYPES: tuple[str, ...] = ("text/html", "application/xhtml+xml")

#: Status codes worth a second attempt. 4xx is a real answer and is never retried.
RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Attempts per URL, including the first.
DEFAULT_MAX_ATTEMPTS = 3

#: Base seconds for exponential backoff between attempts (0.5, 1.0, 2.0, ...).
DEFAULT_BACKOFF_BASE = 0.5

#: Delay between requests when robots.txt declares no crawl-delay. Politeness
#: default, not a rule.
DEFAULT_POLITE_DELAY = 0.2

#: Below this much text in the body, a 200-OK HTML page is very likely
#: client-rendered and we saw only its shell.
JS_SHELL_TEXT_THRESHOLD = 200

_HREF_RE = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_BODY_TEXT_RE = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_SCRIPT_RE = re.compile(r"<script\b", re.IGNORECASE)


class FetchResult(BaseModel):
    """One fetched URL and the transport facts about it.

    This is the crawler's output contract, distinct from ``PageData``: it holds
    what the *transport* observed, before any extraction has happened. Phase 3
    consumes these to build ``PageData`` objects.

    Attributes:
        url: The normalized URL that was requested.
        final_url: Where the request ended up after redirects. Equal to ``url``
            when there were none.
        status_code: Final HTTP status, or 0 when every attempt failed to connect.
        fetched_at: When the response was received (UTC).
        html: Response body as text. Empty for non-HTML or failed fetches.
        depth: Link distance from the start URL. Sitemap seeds are depth 1.
        content_type: Raw ``Content-Type`` header, minus parameters.
        x_robots_tag: Raw ``X-Robots-Tag`` response header value(s), if present —
            every occurrence, unparsed. A server can direct a page to be
            ``noindex`` purely at the HTTP layer, with no ``<meta name="robots">``
            anywhere in the HTML, so this is invisible to any check that only
            reads markup. Phase 3 merges these into ``PageData.robots_meta``, and
            Phase 4's checks can tell which directives came from here rather than
            from HTML (see ``check_http_header_robots``).
        redirect_chain: Intermediate URLs traversed, excluding ``final_url``.
            Length > 1 is the redirect-chain signal Phase 4 reports on.
        error: Populated when the fetch failed outright, e.g. ``"ConnectTimeout"``.
        discovered_links: Every absolute href found on the page, on-domain or not.
            Phase 4 uses the off-domain ones for external-link checks.
    """

    url: str
    final_url: str
    status_code: int
    fetched_at: datetime
    html: str = ""
    depth: int = 0
    content_type: str | None = None
    x_robots_tag: list[str] = Field(default_factory=list)
    redirect_chain: list[str] = Field(default_factory=list)
    error: str | None = None
    discovered_links: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when this fetch produced a usable 2xx HTML response."""
        return self.error is None and 200 <= self.status_code < 300


class CrawlReport(BaseModel):
    """Summary of one crawl, for honest reporting of what was and wasn't seen.

    Attributes:
        start_url: The normalized seed URL.
        pages: Successful and failed fetches, in the order completed.
        skipped: Mapping of URL to the reason it was never fetched, e.g.
            ``"robots_disallow"``, ``"off_domain"``, ``"non_html"``.
        robots_status: What happened when robots.txt was fetched.
        sitemap_urls_found: How many URLs the sitemap contributed to the frontier.
        max_depth_reached: Deepest level actually crawled.
        notes: Free-text warnings, including the JS-rendering caveat.
    """

    start_url: str
    pages: list[FetchResult] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)
    robots_status: str = "not_fetched"
    sitemap_urls_found: int = 0
    max_depth_reached: int = 0
    notes: list[str] = Field(default_factory=list)


class Crawler:
    """Crawler performs controlled async HTTP web crawling on a target website.

    Args:
        user_agent: Sent as User-Agent and matched against robots.txt groups.
        respect_robots: Obey robots.txt disallow rules. Leave on.
        use_sitemap: Seed the frontier from the site's sitemap in addition to
            following links.
        include_subdomains: Treat ``blog.example.com`` as part of ``example.com``.
        max_attempts: Attempts per URL, including the first.
        polite_delay: Fallback delay between requests when robots.txt declares
            no crawl-delay. A robots.txt crawl-delay always wins over this.
        concurrency: Maximum simultaneous in-flight requests.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots: bool = True,
        use_sitemap: bool = True,
        include_subdomains: bool = False,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        polite_delay: float = DEFAULT_POLITE_DELAY,
        concurrency: int = 5,
    ) -> None:
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.use_sitemap = use_sitemap
        self.include_subdomains = include_subdomains
        self.max_attempts = max_attempts
        self.polite_delay = polite_delay
        self.concurrency = concurrency

        #: Populated by the most recent :meth:`crawl`. Holds the skip reasons,
        #: robots status and JS-rendering notes that ``crawl``'s list return
        #: cannot carry.
        self.last_report: CrawlReport | None = None

    def crawl(
        self,
        start_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        timeout: int = 10,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> list[Any]:
        """Crawl pages starting from start_url up to max_pages, max_depth, and timeout seconds.

        Synchronous by design; the concurrency is internal.

        Args:
            start_url: Seed URL. Its host defines the crawl boundary.
            max_pages: Hard ceiling on pages fetched. Never exceeded.
            max_depth: Hard ceiling on link distance from ``start_url``.
                Depth 0 is the seed alone.
            timeout: Per-request timeout in seconds.
            transport: Optional transport, for tests to serve fixtures without
                touching the network.

        Returns:
            The :class:`FetchResult` objects, in completion order. The richer
            :class:`CrawlReport` is available on :attr:`last_report`.

        Raises:
            InvalidURLError: If ``start_url`` is not a crawlable http(s) URL.
        """
        seed = normalize_url(start_url)
        report = CrawlReport(start_url=seed)
        self.last_report = report

        asyncio.run(
            self._crawl_async(
                seed=seed,
                max_pages=max_pages,
                max_depth=max_depth,
                timeout=timeout,
                transport=transport,
                report=report,
            )
        )
        return report.pages

    async def _crawl_async(
        self,
        *,
        seed: str,
        max_pages: int,
        max_depth: int,
        timeout: int,
        transport: httpx.AsyncBaseTransport | None,
        report: CrawlReport,
    ) -> None:
        """BFS driver: process one depth level at a time until a limit is hit."""
        robots = RobotsChecker(seed, user_agent=self.user_agent, timeout=float(timeout))
        delay = self.polite_delay

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
            transport=transport,
        ) as client:
            if self.respect_robots:
                await asyncio.to_thread(self._load_robots, robots, timeout, transport)
                report.robots_status = robots.status
                declared = robots.crawl_delay(self.user_agent)
                if declared is not None:
                    delay = max(delay, declared)

            seen: set[str] = set()
            frontier: list[tuple[str, int]] = []

            def enqueue(candidate: str, depth: int) -> None:
                """Add a URL to the frontier if it passes every gate."""
                if len(seen) >= max_pages and candidate not in seen:
                    return
                try:
                    normalized = normalize_url(candidate)
                    key = generate_dedup_key(normalized)
                except InvalidURLError:
                    return
                if key in seen:
                    return
                if not is_same_domain(normalized, seed, self.include_subdomains):
                    report.skipped.setdefault(normalized, "off_domain")
                    return
                if self.respect_robots and not robots.can_fetch(normalized):
                    report.skipped.setdefault(normalized, "robots_disallow")
                    return
                seen.add(key)
                frontier.append((normalized, depth))

            enqueue(seed, 0)

            if self.use_sitemap:
                sitemap_urls = await asyncio.to_thread(
                    self._load_sitemap, seed, timeout, transport
                )
                before = len(frontier)
                for url in sitemap_urls:
                    if len(report.pages) + len(frontier) >= max_pages:
                        break
                    # Sitemap seeds sit at depth 1: they are site-declared entry
                    # points, not links found on the homepage, but they should
                    # still respect a shallow max_depth.
                    if max_depth >= 1:
                        enqueue(url, 1)
                report.sitemap_urls_found = len(frontier) - before

            semaphore = asyncio.Semaphore(self.concurrency)

            while frontier and len(report.pages) < max_pages:
                budget = max_pages - len(report.pages)
                batch = frontier[:budget]
                frontier = frontier[budget:]

                results = await asyncio.gather(
                    *(
                        self._fetch_with_retry(client, url, depth, semaphore, delay)
                        for url, depth in batch
                    )
                )

                for result in results:
                    if result is None:
                        continue
                    report.pages.append(result)
                    report.max_depth_reached = max(report.max_depth_reached, result.depth)

                    if result.content_type and not _is_html(result.content_type):
                        report.skipped.setdefault(result.url, "non_html")
                        continue

                    if result.ok and _looks_js_rendered(result.html):
                        report.notes.append(
                            f"{result.final_url}: 200 OK but almost no server-rendered text — "
                            "the page is likely client-rendered and its real content was not seen."
                        )

                    if result.depth >= max_depth:
                        continue

                    for href in result.discovered_links:
                        if len(report.pages) + len(frontier) >= max_pages:
                            break
                        enqueue(href, result.depth + 1)

        if report.robots_status == "server_error":
            report.notes.append(
                "robots.txt returned 5xx; every URL was treated as disallowed."
            )

    @staticmethod
    def _load_robots(
        robots: RobotsChecker,
        timeout: int,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        """Load robots.txt in a worker thread (RobotFileParser is sync-only)."""
        sync_transport = transport if isinstance(transport, httpx.BaseTransport) else None
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": robots.user_agent},
            transport=sync_transport,
        ) as client:
            robots.load(client=client)

    def _load_sitemap(
        self,
        seed: str,
        timeout: int,
        transport: httpx.AsyncBaseTransport | None,
    ) -> list[str]:
        """Fetch sitemap URLs in a worker thread; failures degrade to an empty list."""
        sync_transport = transport if isinstance(transport, httpx.BaseTransport) else None
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
                transport=sync_transport,
            ) as client:
                return fetch_sitemap(
                    seed,
                    timeout=float(timeout),
                    user_agent=self.user_agent,
                    client=client,
                )
        except Exception:
            return []

    async def _fetch_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        depth: int,
        semaphore: asyncio.Semaphore,
        delay: float,
    ) -> FetchResult | None:
        """Fetch one URL, retrying transient failures with exponential backoff."""
        async with semaphore:
            last_error: str | None = None
            response: httpx.Response | None = None

            for attempt in range(self.max_attempts):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    response = await client.get(url)
                    last_error = None
                    if response.status_code not in RETRYABLE_STATUS:
                        break
                except httpx.HTTPError as exc:
                    response = None
                    last_error = type(exc).__name__

                if attempt < self.max_attempts - 1:
                    await asyncio.sleep(DEFAULT_BACKOFF_BASE * (2**attempt))

            fetched_at = datetime.now(timezone.utc)

            if response is None:
                return FetchResult(
                    url=url,
                    final_url=url,
                    status_code=0,
                    fetched_at=fetched_at,
                    depth=depth,
                    error=last_error or "unknown_error",
                )

            content_type = (response.headers.get("content-type") or "").split(";")[0].strip()
            # get_list, not get: a response can legitimately carry more than one
            # X-Robots-Tag header (a general one plus a per-bot one is a common
            # real pattern), and dropping any of them could hide a real noindex.
            x_robots_tag = response.headers.get_list("x-robots-tag")
            final_url = str(response.url)
            redirect_chain = [str(r.url) for r in response.history]

            # Non-HTML bodies are never decoded or scanned for links.
            html = response.text if _is_html(content_type) else ""
            links = _extract_links(html, base=final_url) if html else []

            return FetchResult(
                url=url,
                final_url=final_url,
                status_code=response.status_code,
                fetched_at=fetched_at,
                html=html,
                depth=depth,
                content_type=content_type or None,
                x_robots_tag=x_robots_tag,
                redirect_chain=redirect_chain,
                discovered_links=links,
            )


def _is_html(content_type: str) -> bool:
    """True when a Content-Type names an HTML document."""
    if not content_type:
        # An unlabelled body is optimistically treated as HTML; the alternative
        # is dropping pages from servers that omit the header.
        return True
    return content_type.lower().startswith(HTML_CONTENT_TYPES)


def _extract_links(html: str, base: str) -> list[str]:
    """Return absolute, de-duplicated hrefs found in ``html``.

    A regex rather than a DOM parse: this runs on every fetched page purely to
    build the frontier, and the real, correctness-critical link extraction (with
    anchor text and internal/external flags for ``PageData``) belongs to Phase 3's
    ``seo_extractor``. Missing an href inside an HTML comment here costs one
    frontier entry, not a wrong finding.
    """
    out: list[str] = []
    seen: set[str] = set()

    for href in _HREF_RE.findall(html):
        href = href.strip()
        if not href or href.startswith("#"):
            continue
        try:
            absolute = normalize_url(href, base=base)
        except InvalidURLError:
            continue
        if absolute not in seen:
            seen.add(absolute)
            out.append(absolute)

    return out


def _looks_js_rendered(html: str) -> bool:
    """Heuristic: a 200 OK page with almost no text *and* a script that could supply it.

    Both halves are required. Low text alone is not evidence of client rendering —
    ``example.com`` is a real, complete, server-rendered page of about 170
    characters, and calling it a JS shell would be a false claim in the crawl
    report. Demanding a ``<script>`` as well separates "this page is small" from
    "this page's content arrives later".

    Used only to attach a note. It never suppresses a page, because a genuinely
    sparse page is also a real SEO finding — the distinction is Phase 4's to make,
    with this note as evidence.
    """
    if not html:
        return False

    text = _BODY_TEXT_RE.sub(" ", html)
    if len(" ".join(text.split())) >= JS_SHELL_TEXT_THRESHOLD:
        return False

    return _SCRIPT_RE.search(html) is not None
