"""Sitemap fetcher and parser, handling XML sitemaps and sitemap index files.

A sitemap is a hint, never a requirement. Every failure here — absent file,
malformed XML, a sitemap index nested inside a sitemap index inside another —
degrades to "return the URLs we did manage to read", because the crawler can
always fall back to following links from the homepage. Nothing in this module
raises on a bad sitemap.

Discovery order:

1. ``Sitemap:`` lines in robots.txt, which is where a site declares a sitemap
   that does not live at the conventional path.
2. The conventional locations (``/sitemap.xml``, ``/sitemap_index.xml``,
   ``/sitemap-index.xml``, ``/sitemap.xml.gz``).

Both plain ``<urlset>`` sitemaps and ``<sitemapindex>`` files are handled, with
nested indexes followed up to :data:`MAX_INDEX_DEPTH`.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.crawler.robots import DEFAULT_USER_AGENT, RobotsChecker
from app.crawler.url_utils import InvalidURLError, is_same_domain, normalize_url

#: Conventional sitemap paths, tried in order when robots.txt declares none.
CONVENTIONAL_SITEMAP_PATHS: tuple[str, ...] = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/sitemap.xml.gz",
)

#: How far to follow sitemap-index -> sitemap-index nesting before giving up.
#: Real sites nest one level; anything deeper is a loop or a mistake.
MAX_INDEX_DEPTH = 3

#: Ceiling on URLs returned, so one enormous sitemap cannot exhaust memory before
#: the crawler's own ``max_pages`` gets a chance to apply.
MAX_SITEMAP_URLS = 5000

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def fetch_sitemap(
    domain: str,
    timeout: float = 10.0,
    user_agent: str = DEFAULT_USER_AGENT,
    client: httpx.Client | None = None,
    restrict_to_domain: bool = True,
) -> list[str]:
    """Fetch and parse sitemap(s) for a domain, returning a list of target URLs.

    Handles both standard sitemap files and sitemap indexes.

    Args:
        domain: A bare host (``example.com``) or any URL on the target origin.
        timeout: Seconds to wait for each sitemap request.
        user_agent: Value sent in the User-Agent header.
        client: Optional pre-configured HTTP client to reuse.
        restrict_to_domain: Drop sitemap entries pointing at other hosts. A
            sitemap is allowed to list only URLs on its own site, and entries
            that break that rule are typically spam or a misconfiguration.

    Returns:
        Normalized, de-duplicated page URLs in the order first seen. Empty if no
        sitemap exists or none could be parsed — never an error.
    """
    owns_client = client is None
    client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    )

    try:
        candidates = _discover_sitemap_urls(domain, timeout, user_agent, client)

        found: list[str] = []
        seen: set[str] = set()
        visited_sitemaps: set[str] = set()

        for sitemap_url in candidates:
            _collect(
                sitemap_url,
                client=client,
                domain=domain,
                depth=0,
                found=found,
                seen=seen,
                visited_sitemaps=visited_sitemaps,
                restrict_to_domain=restrict_to_domain,
            )
            if len(found) >= MAX_SITEMAP_URLS:
                break

        return found[:MAX_SITEMAP_URLS]
    finally:
        if owns_client:
            client.close()


def _discover_sitemap_urls(
    domain: str,
    timeout: float,
    user_agent: str,
    client: httpx.Client,
) -> list[str]:
    """Return candidate sitemap URLs: robots.txt declarations, then conventions."""
    candidates: list[str] = []

    try:
        robots = RobotsChecker(domain, user_agent=user_agent, timeout=timeout)
        robots.load(client=client)
        candidates.extend(robots.sitemaps())
    except Exception:
        # robots.txt is only a discovery hint here; its absence is not fatal.
        pass

    origin = _origin_of(domain)
    for path in CONVENTIONAL_SITEMAP_PATHS:
        candidates.append(f"{origin}{path}")

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _collect(
    sitemap_url: str,
    *,
    client: httpx.Client,
    domain: str,
    depth: int,
    found: list[str],
    seen: set[str],
    visited_sitemaps: set[str],
    restrict_to_domain: bool,
) -> None:
    """Fetch one sitemap and append its page URLs to ``found``, recursing into indexes."""
    if depth > MAX_INDEX_DEPTH or len(found) >= MAX_SITEMAP_URLS:
        return
    if sitemap_url in visited_sitemaps:
        return
    visited_sitemaps.add(sitemap_url)

    body = _fetch_bytes(sitemap_url, client)
    if body is None:
        return

    root = _parse_xml(body)
    if root is None:
        return

    tag = _local_name(root.tag)

    if tag == "sitemapindex":
        for loc in _locs(root, "sitemap"):
            _collect(
                loc,
                client=client,
                domain=domain,
                depth=depth + 1,
                found=found,
                seen=seen,
                visited_sitemaps=visited_sitemaps,
                restrict_to_domain=restrict_to_domain,
            )
        return

    if tag != "urlset":
        return

    for loc in _locs(root, "url"):
        if len(found) >= MAX_SITEMAP_URLS:
            return
        try:
            normalized = normalize_url(loc)
        except InvalidURLError:
            continue
        if restrict_to_domain and not is_same_domain(normalized, domain):
            continue
        if normalized not in seen:
            seen.add(normalized)
            found.append(normalized)


def _fetch_bytes(url: str, client: httpx.Client) -> bytes | None:
    """GET a sitemap, transparently gunzipping ``.gz``. ``None`` on any failure."""
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return None

    if response.status_code >= 400:
        return None

    content = response.content
    if url.endswith(".gz") or content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except (OSError, EOFError):
            return None

    return content


def _parse_xml(body: bytes) -> ET.Element | None:
    """Parse sitemap XML, returning ``None`` for anything malformed.

    A truncated or HTML-instead-of-XML response is common enough (soft 404 pages
    are frequently served at ``/sitemap.xml``) that it must not raise.
    """
    try:
        return ET.fromstring(body)
    except ET.ParseError:
        return None


def _locs(root: ET.Element, child_tag: str) -> list[str]:
    """Return the ``<loc>`` text of every ``<child_tag>`` under ``root``.

    Namespace-tolerant: sitemaps in the wild are served both with and without the
    sitemaps.org namespace declaration, and a namespace-qualified ``findall``
    silently returns nothing for the latter.
    """
    values: list[str] = []
    for child in root:
        if _local_name(child.tag) != child_tag:
            continue
        for grandchild in child:
            if _local_name(grandchild.tag) == "loc" and grandchild.text:
                values.append(grandchild.text.strip())
                break
    return values


def _local_name(tag: str) -> str:
    """Strip any ``{namespace}`` prefix from an XML tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _origin_of(domain: str) -> str:
    """Return ``scheme://host`` for a bare host or full URL, defaulting to https."""
    candidate = (domain or "").strip()
    if "//" not in candidate:
        candidate = f"https://{candidate}"
    split = urlsplit(candidate)
    return urlunsplit((split.scheme or "https", split.netloc, "", "", ""))
