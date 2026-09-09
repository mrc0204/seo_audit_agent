"""SEO agent executing deterministic auditing rules and generating findings across crawled pages."""

from typing import Any


def run_seo_audit(pages: list[Any]) -> list[Any]:
    """Perform deterministic SEO audit checks on crawled web pages.

    Deterministic checks to be implemented:
    - Missing title tags (<title>)
    - Missing meta descriptions (<meta name="description">)
    - Duplicate page titles across pages
    - Missing or invalid canonical tags (<link rel="canonical">)
    - Heading hierarchy issues (missing H1, multiple H1s, skipped heading levels)
    - Missing alt text on images (<img> without alt attribute)
    - Broken internal and external links (HTTP 4xx/5xx responses)
    - Robots meta tag restrictions (noindex, nofollow)
    - Open Graph / Social metadata completeness
    """
    raise NotImplementedError
