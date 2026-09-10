"""SEO metadata extractor for extracting page titles, meta tags, canonicals, headings, links, and images.

Populates ``PageData`` fields and makes **no judgments**. Nothing here decides that
a title is too long or that a missing alt is a problem — those are Phase 4's
deterministic checks. This module's only job is to report, faithfully, what the
markup says.

One distinction runs through the whole module and matters more than it looks:

    **absent is not the same as empty.**

``<title></title>`` is a different defect from having no ``<title>`` at all, and
``alt=""`` is *correct* markup for a decorative image while a missing ``alt`` is a
real accessibility and SEO problem. So an absent element yields ``None`` and a
present-but-empty one yields ``""``. Collapsing the two here would destroy evidence
Phase 4 needs and make the agent report a fix the site has already applied.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

from app.crawler.url_utils import InvalidURLError, is_same_domain, normalize_url
from app.extraction.content_extractor import extract_clean_content
from app.extraction.html_parser import (
    element_text,
    extract_visible_text,
    normalize_whitespace,
    parse_html,
)
from app.extraction.schema_extractor import extract_json_ld
from app.models.page import ImageRef, LinkRef, PageData

#: Heading levels collected into ``PageData.headings``. Every level is present as
#: a key even when empty, so Phase 4 can test ``headings["h1"]`` without a guard.
HEADING_LEVELS: tuple[str, ...] = ("h1", "h2", "h3", "h4", "h5", "h6")

#: ``<meta name=...>`` values that carry indexing directives. ``googlebot`` is
#: included because a ``noindex`` there is just as real as one on ``robots``.
ROBOTS_META_NAMES: frozenset[str] = frozenset({"robots", "googlebot"})


def extract_seo_fields(soup: BeautifulSoup, url: str) -> dict[str, Any]:
    """Extract partial PageData SEO fields (title, meta, canonical, robots meta, headings, images, links).

    Args:
        soup: A cleaned tree from :func:`app.extraction.html_parser.parse_html`.
        url: The page's final URL, used to resolve relative hrefs and to decide
            which links are internal. Pass the post-redirect URL, not the
            requested one, or relative links resolve against the wrong base.

    Returns:
        A dict of ``PageData`` fields, plus three extra keys that ``PageData``
        has no room for but Phase 4's checks need:

        * ``canonical_count`` — number of ``<link rel="canonical">`` elements.
          More than one is its own finding, and keeping only the first would
          hide it.
        * ``title_count`` — number of ``<title>`` elements.
        * ``meta_description_count`` — number of description meta tags.

        :func:`build_page_data` drops these when constructing the model, so the
        Phase 1 contract stays locked; a check that needs them calls this
        function directly.
    """
    if soup is None:
        soup = BeautifulSoup("", "html.parser")

    title_tags = soup.find_all("title")
    description_tags = _meta_tags_named(soup, "description")
    canonical_tags = [
        tag
        for tag in soup.find_all("link")
        if "canonical" in _rel_values(tag)
    ]

    return {
        "title": element_text(title_tags[0]) if title_tags else None,
        "title_count": len(title_tags),
        "meta_description": (
            normalize_whitespace(description_tags[0].get("content") or "")
            if description_tags
            else None
        ),
        "meta_description_count": len(description_tags),
        "canonical": _resolve(canonical_tags[0].get("href"), url) if canonical_tags else None,
        "canonical_count": len(canonical_tags),
        "robots_meta": _robots_directives(soup),
        "headings": _headings(soup),
        "images": _images(soup, url),
        "links": _links(soup, url),
    }


def build_page_data(
    url: str,
    final_url: str,
    status_code: int,
    html: str,
    fetched_at: datetime | None = None,
    x_robots_tag: list[str] | None = None,
) -> PageData:
    """Assemble a complete :class:`~app.models.page.PageData` from one fetched page.

    The seam between Phase 2 and Phase 4: hand it a ``FetchResult``'s fields and
    it returns the locked Phase 1 contract, fully populated.

    ``structured_data`` is parsed from the **raw** ``html`` rather than the cleaned
    tree, because :func:`app.extraction.html_parser.parse_html` strips the
    ``<script>`` elements JSON-LD lives in.

    Args:
        url: The URL originally requested.
        final_url: Where it ended up after redirects. Used as the base for
            resolving relative links.
        status_code: Final HTTP status.
        html: Raw response body.
        fetched_at: When it was fetched. Defaults to now (UTC).
        x_robots_tag: Raw ``X-Robots-Tag`` response header value(s), from
            ``FetchResult.x_robots_tag``. Merged into ``PageData.robots_meta``
            alongside any HTML ``<meta name="robots">`` directives — deliberately
            merged into that one existing (Phase-1-locked) field rather than
            given a new one, since both are the same underlying signal
            ("indexing directives found for this page") from two different
            transports. A server can direct ``noindex`` purely at the HTTP layer
            with no trace of it in the HTML at all, so without this a page could
            be excluded from search in a way nothing in this pipeline would ever
            see. ``None`` (the default) merges nothing, matching every existing
            call site that predates this parameter.

    Returns:
        A populated ``PageData``. Never raises on malformed markup — a page that
        cannot be parsed yields empty fields, not an exception.
    """
    soup = parse_html(html)
    text = extract_visible_text(soup)
    fields = extract_seo_fields(soup, final_url)

    robots_meta = list(fields["robots_meta"])
    for directive in parse_x_robots_tag(x_robots_tag or []):
        if directive not in robots_meta:
            robots_meta.append(directive)

    return PageData(
        url=url,
        final_url=final_url,
        status_code=status_code,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        html=html or "",
        text=text,
        title=fields["title"],
        meta_description=fields["meta_description"],
        canonical=fields["canonical"],
        robots_meta=robots_meta,
        headings=fields["headings"],
        images=fields["images"],
        links=fields["links"],
        structured_data=extract_json_ld(html),
        content_hash=content_hash(text),
    )


#: A bot-name prefix on an X-Robots-Tag value, e.g. the "googlebot" in
#: "googlebot: noindex, nofollow". Per the header's spec this scopes every
#: comma-separated directive that follows it, not just the first.
_ROBOTS_HEADER_BOT_PREFIX = re.compile(r"^\s*([a-zA-Z][\w-]*)\s*:\s*(.*)$")


def parse_x_robots_tag(raw_values: list[str]) -> list[str]:
    """Parse ``X-Robots-Tag`` header value(s) into the same directive vocabulary
    used for HTML ``<meta name="robots">`` — lowercase tokens like ``"noindex"``.

    Handles the header's optional leading bot-name scope (``"googlebot: noindex,
    nofollow"``) by dropping the bot name and keeping the directives, since this
    pipeline does not currently distinguish which bot a directive targets — any
    ``noindex`` anywhere is worth surfacing.

    Args:
        raw_values: Every occurrence of the header, unparsed (a response may
            legitimately carry more than one).

    Returns:
        Deduplicated, order-preserving lowercase directives.
    """
    directives: list[str] = []

    for raw in raw_values:
        if not raw:
            continue
        match = _ROBOTS_HEADER_BOT_PREFIX.match(raw)
        remainder = match.group(2) if match else raw
        for token in remainder.split(","):
            directive = token.strip().lower()
            if directive and directive not in directives:
                directives.append(directive)

    return directives


def content_hash(text: str) -> str:
    """Return a stable hash of a page's normalized text, for duplicate detection.

    Hashes the *normalized visible text*, not the raw HTML: two pages with
    identical prose but different build ids, CSRF tokens or asset hashes are
    duplicate content in the sense an SEO audit cares about, and hashing the HTML
    would miss every one of them.

    Args:
        text: Normalized visible text.

    Returns:
        A hex SHA-256 digest. Empty text hashes to the digest of the empty string,
        so blank pages collide with each other — which is the correct signal.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def extract_clean_text(html: str) -> str:
    """Return the boilerplate-stripped text used for retrieval chunking.

    Convenience wrapper so Phase 6 does not need to know the parse/extract order.

    Args:
        html: Raw page HTML.

    Returns:
        Blocks joined by ``content_extractor.BLOCK_SEPARATOR``, each a literal
        substring of the page's ``PageData.text``.
    """
    return extract_clean_content(parse_html(html))


def _meta_tags_named(soup: BeautifulSoup, name: str) -> list:
    """Return ``<meta>`` tags whose ``name`` matches, case-insensitively."""
    return [
        tag
        for tag in soup.find_all("meta")
        if (tag.get("name") or "").strip().lower() == name
    ]


def _rel_values(tag) -> set[str]:
    """Return a link tag's ``rel`` tokens, lowercased."""
    rel = tag.get("rel") or []
    values = rel if isinstance(rel, list) else str(rel).split()
    return {str(v).strip().lower() for v in values}


def _robots_directives(soup: BeautifulSoup) -> list[str]:
    """Return indexing directives from robots/googlebot meta tags.

    Comma-separated values are split and lowercased, so ``content="NOINDEX,
    NOFOLLOW"`` becomes ``["noindex", "nofollow"]`` — the shape the ``PageData``
    contract documents. Order is preserved and duplicates removed.
    """
    directives: list[str] = []

    for tag in soup.find_all("meta"):
        name = (tag.get("name") or "").strip().lower()
        if name not in ROBOTS_META_NAMES:
            continue
        for token in (tag.get("content") or "").split(","):
            directive = token.strip().lower()
            if directive and directive not in directives:
                directives.append(directive)

    return directives


def _headings(soup: BeautifulSoup) -> dict[str, list[str]]:
    """Return ``{"h1": [...], ..., "h6": [...]}`` of normalized heading text.

    Empty headings are kept as ``""``: an empty ``<h1>`` is a real defect, and
    dropping it would make the page look like it simply has no H1 — a different
    finding with a different fix.
    """
    return {
        level: [element_text(tag) for tag in soup.find_all(level)]
        for level in HEADING_LEVELS
    }


def _images(soup: BeautifulSoup, base: str) -> list[ImageRef]:
    """Return every ``<img>`` as an :class:`ImageRef` with an absolute ``src``.

    ``alt`` is ``None`` when the attribute is absent and ``""`` when it is present
    but empty — the difference between a missing alt (a finding) and an
    intentionally decorative image (correct markup).

    Images with no ``src`` at all are still returned, with ``src=""``, since a
    broken image tag is itself worth reporting.
    """
    images: list[ImageRef] = []

    for tag in soup.find_all("img"):
        raw_src = tag.get("src") or tag.get("data-src") or ""
        alt = tag.get("alt")

        images.append(
            ImageRef(
                src=_resolve(raw_src, base) or normalize_whitespace(raw_src),
                alt=normalize_whitespace(alt) if alt is not None else None,
            )
        )

    return images


def _links(soup: BeautifulSoup, base: str) -> list[LinkRef]:
    """Return every ``<a href>`` as a :class:`LinkRef`.

    Anchors that are not crawlable targets — ``mailto:``, ``tel:``,
    ``javascript:``, bare ``#`` fragments — are skipped, because they are not
    links in the sense any Phase 4 check means. Phase 5 reads ``tel:`` links
    directly from the tree for NAP extraction instead.

    Anchor text is normalized and may legitimately be ``""`` (an image-only link),
    which is itself a finding Phase 4 can make.
    """
    links: list[LinkRef] = []

    for tag in soup.find_all("a"):
        href = (tag.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue

        resolved = _resolve(href, base)
        if resolved is None:
            continue

        links.append(
            LinkRef(
                href=resolved,
                anchor_text=element_text(tag),
                is_internal=is_same_domain(resolved, base),
            )
        )

    return links


def _resolve(href: str | None, base: str) -> str | None:
    """Resolve an href against ``base``, returning ``None`` if not crawlable."""
    if not href:
        return None
    try:
        return normalize_url(href, base=base)
    except InvalidURLError:
        return None
