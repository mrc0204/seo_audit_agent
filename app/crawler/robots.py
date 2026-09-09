"""Robots.txt parser and checker for site crawling compliance."""

import httpx


class RobotsChecker:
    """RobotsChecker handles parsing robots.txt and verifying crawl permissions."""

    def __init__(self, domain: str) -> None:
        """Initialize RobotsChecker with a domain."""
        self.domain = domain

    def can_fetch(self, url: str) -> bool:
        """Check whether the given URL is allowed to be crawled according to robots.txt."""
        raise NotImplementedError

    def crawl_delay(self, user_agent: str) -> float | None:
        """Get the crawl delay specified in robots.txt for a given user agent, if any."""
        raise NotImplementedError
