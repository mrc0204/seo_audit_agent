"""Sitemap fetcher and parser, handling XML sitemaps and sitemap index files."""

import httpx


def fetch_sitemap(domain: str) -> list[str]:
    """Fetch and parse sitemap(s) for a domain, returning a list of target URLs.

    Handles both standard sitemap files and sitemap indexes.
    """
    raise NotImplementedError
