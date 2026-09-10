"""Robots.txt parser and checker for site crawling compliance.

A thin wrapper over :mod:`urllib.robotparser` that fetches over HTTP with a real
timeout (the stdlib's own ``read()`` uses :mod:`urllib.request`, which is hard to
bound and ignores our client settings), caches one parse per domain, and takes an
explicit position on each failure mode:

===================  ==========================================================
robots.txt response  Behaviour
===================  ==========================================================
2xx                  Parse and obey it.
4xx (incl. 404)      Allow everything. A missing robots.txt permits crawling.
5xx                  Disallow everything. The site is failing; treat its rules
                     as unknown rather than assuming consent.
network error        Allow everything, but record the reason. Failing closed
                     here would make a transient DNS blip look like a site that
                     forbids crawling, which is a worse lie than the alternative.
===================  ==========================================================

The 5xx rule follows Google's documented behaviour. Every decision is recorded on
:attr:`RobotsChecker.status` so the crawl report can state *why* pages were skipped
instead of silently returning nothing.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

#: Identifies this crawler in the User-Agent header and in robots.txt lookups.
DEFAULT_USER_AGENT = "seo-audit-agent"


class RobotsChecker:
    """RobotsChecker handles parsing robots.txt and verifying crawl permissions.

    One instance covers one origin and caches the fetched rules, so
    :meth:`can_fetch` is cheap to call per URL.

    Attributes:
        domain: The origin these rules apply to.
        robots_url: The ``/robots.txt`` URL that was (or will be) fetched.
        status: Record of what happened on fetch — one of ``"not_fetched"``,
            ``"ok"``, ``"missing"``, ``"server_error"``, or
            ``"fetch_failed: <reason>"``. Surface this in the crawl report.
    """

    def __init__(
        self,
        domain: str,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 10.0,
    ) -> None:
        """Initialize RobotsChecker with a domain.

        Args:
            domain: A bare host (``example.com``) or any URL on the target origin.
                The scheme defaults to https when omitted.
            user_agent: The token matched against robots.txt ``User-agent`` groups.
            timeout: Seconds to wait for the robots.txt fetch.
        """
        self.domain = domain
        self.user_agent = user_agent
        self.timeout = timeout

        self.robots_url = self._robots_url_for(domain)
        self.status: str = "not_fetched"

        self._parser: RobotFileParser | None = None
        self._allow_all = True
        self._loaded = False

    def load(self, client: httpx.Client | None = None) -> None:
        """Fetch and parse robots.txt. Idempotent; later calls are no-ops.

        Called automatically on first use, so callers rarely need it — pass a
        ``client`` only to reuse an existing connection pool.

        Args:
            client: Optional pre-configured HTTP client.
        """
        if self._loaded:
            return
        self._loaded = True

        owns_client = client is None
        client = client or httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        )
        try:
            response = client.get(self.robots_url)
        except httpx.HTTPError as exc:
            # Fail open: a transient network problem is not a crawl prohibition.
            self.status = f"fetch_failed: {type(exc).__name__}"
            self._allow_all = True
            return
        finally:
            if owns_client:
                client.close()

        if response.status_code >= 500:
            # Rules unknown while the origin is broken — assume nothing is permitted.
            self.status = "server_error"
            self._allow_all = False
            return

        if response.status_code >= 400:
            self.status = "missing"
            self._allow_all = True
            return

        parser = RobotFileParser()
        parser.set_url(self.robots_url)
        parser.parse(response.text.splitlines())
        self._parser = parser
        self._allow_all = False
        self.status = "ok"

    def can_fetch(self, url: str) -> bool:
        """Check whether the given URL is allowed to be crawled according to robots.txt.

        Args:
            url: Absolute URL to test.

        Returns:
            ``True`` if crawling is permitted. When robots.txt was missing or
            unreachable this is ``True``; when the origin returned 5xx it is
            ``False`` for every URL.
        """
        self.load()

        if self._parser is None:
            return self._allow_all

        return self._parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, user_agent: str) -> float | None:
        """Get the crawl delay specified in robots.txt for a given user agent, if any.

        Falls back to the delay implied by a ``Request-rate`` directive when no
        explicit ``Crawl-delay`` is present.

        Args:
            user_agent: The token to look up.

        Returns:
            Seconds to wait between requests, or ``None`` if unspecified.
        """
        self.load()

        if self._parser is None:
            return None

        delay = self._parser.crawl_delay(user_agent)
        if delay is not None:
            return float(delay)

        rate = self._parser.request_rate(user_agent)
        if rate is not None and rate.requests > 0:
            return float(rate.seconds) / float(rate.requests)

        return None

    def sitemaps(self) -> list[str]:
        """Return sitemap URLs declared in robots.txt.

        ``Sitemap:`` lines are the most reliable way to discover a sitemap that
        does not live at the conventional ``/sitemap.xml``.

        Returns:
            Declared sitemap URLs, or an empty list if none.
        """
        self.load()

        if self._parser is None:
            return []

        return list(self._parser.site_maps() or [])

    @staticmethod
    def _robots_url_for(domain: str) -> str:
        """Build the ``/robots.txt`` URL for a bare host or a full URL."""
        candidate = (domain or "").strip()
        if "//" not in candidate:
            candidate = f"https://{candidate}"

        split = urlsplit(candidate)
        scheme = split.scheme or "https"
        return urlunsplit((scheme, split.netloc, "/robots.txt", "", ""))
