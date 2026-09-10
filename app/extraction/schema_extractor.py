"""Structured schema data extractor for JSON-LD scripts and Microdata tags.

Both extractors are total: malformed JSON, a wrong ``@type``, or nonsensical
nesting yields fewer results, never an exception. Structured data is the single
most frequently broken thing on real sites — a trailing comma in one JSON-LD block
must not abort a two-hundred-page crawl.

Note the parser asymmetry. :func:`extract_json_ld` accepts either a tree or the
raw HTML string, because :func:`app.extraction.html_parser.parse_html` strips
``<script>`` elements — including the ``application/ld+json`` ones this module
needs. Pass raw HTML when you have it; passing a stripped tree silently finds
nothing, so :func:`app.extraction.seo_extractor.build_page_data` always passes raw.
"""

from __future__ import annotations

import json
from typing import Any

from bs4 import BeautifulSoup

from app.extraction.html_parser import _PARSERS, normalize_whitespace

#: ``<script>`` types that carry JSON-LD. The spec names the first; the others
#: appear in the wild from older CMS templates.
JSON_LD_TYPES: tuple[str, ...] = (
    "application/ld+json",
    "application/json+ld",
    "application/ld json",
)

#: Ceiling on microdata nesting, to bound pathological or self-referential markup.
MAX_MICRODATA_DEPTH = 6


def extract_json_ld(soup: BeautifulSoup | str) -> list[dict[str, Any]]:
    """Extract JSON-LD structured data objects embedded in script tags.

    Handles the three shapes real sites emit: a single object, a top-level array
    of objects, and a ``@graph`` wrapper. All are flattened to a flat list of
    objects so callers never have to branch on the container.

    Args:
        soup: Raw HTML, or a BeautifulSoup tree that still contains its
            ``<script>`` elements. A tree from
            :func:`app.extraction.html_parser.parse_html` has had them stripped
            and will yield an empty list.

    Returns:
        Parsed JSON-LD objects in document order. Blocks that fail to parse are
        skipped silently; a page with one broken block still returns its good ones.
    """
    tree = _as_tree(soup)
    if tree is None:
        return []

    blocks: list[dict[str, Any]] = []

    for script in tree.find_all("script"):
        script_type = (script.get("type") or "").strip().lower()
        if script_type not in JSON_LD_TYPES:
            continue

        payload = script.string or script.get_text() or ""
        if not payload.strip():
            continue

        try:
            parsed = json.loads(payload)
        except (ValueError, TypeError):
            # Malformed JSON-LD is extremely common. Skip the block, keep the page.
            continue

        blocks.extend(_flatten_json_ld(parsed))

    return blocks


def extract_microdata(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Extract Microdata structured items from HTML attributes.

    A fallback for sites that predate JSON-LD, and the second-priority NAP source
    in Phase 5. Each top-level ``itemscope`` becomes one dict shaped like its
    JSON-LD equivalent — ``@type`` plus its properties — so downstream code can
    treat both sources uniformly.

    Args:
        soup: A parsed tree. Works on a stripped tree, since microdata lives in
            attributes on visible elements rather than in scripts.

    Returns:
        One dict per top-level ``itemscope`` element, in document order.
    """
    tree = _as_tree(soup)
    if tree is None:
        return []

    items: list[dict[str, Any]] = []

    for element in tree.find_all(attrs={"itemscope": True}):
        # Only top-level scopes; nested ones are recursed into by their parent.
        if element.find_parent(attrs={"itemscope": True}) is not None:
            continue
        items.append(_microdata_item(element, depth=0))

    return items


def _microdata_item(element, depth: int) -> dict[str, Any]:
    """Build one microdata item dict from an ``itemscope`` element."""
    item: dict[str, Any] = {}

    itemtype = element.get("itemtype")
    if itemtype:
        types = itemtype if isinstance(itemtype, list) else [itemtype]
        item["@type"] = [t.rstrip("/").rsplit("/", 1)[-1] for t in types if t]
        if len(item["@type"]) == 1:
            item["@type"] = item["@type"][0]

    if depth >= MAX_MICRODATA_DEPTH:
        return item

    for prop in element.find_all(attrs={"itemprop": True}):
        # Skip properties owned by a nested scope; that scope collects its own.
        owner = prop.find_parent(attrs={"itemscope": True})
        if owner is not element:
            continue

        names = prop.get("itemprop")
        names = names if isinstance(names, list) else [names]

        if prop.get("itemscope") is not None:
            value: Any = _microdata_item(prop, depth + 1)
        else:
            value = _microdata_value(prop)

        for name in names:
            if not name:
                continue
            if name in item:
                existing = item[name]
                item[name] = existing + [value] if isinstance(existing, list) else [existing, value]
            else:
                item[name] = value

    return item


def _microdata_value(element) -> str:
    """Return a microdata property's value from the attribute the spec designates.

    The attribute that carries the value depends on the element: ``<meta>`` uses
    ``content``, ``<a>`` uses ``href``, ``<time>`` uses ``datetime``, and everything
    else falls back to its text. Reading text from a ``<meta>`` would return the
    empty string and silently drop the value.
    """
    name = element.name.lower()

    if name == "meta":
        return normalize_whitespace(element.get("content") or "")
    if name in {"a", "area", "link"}:
        return normalize_whitespace(element.get("href") or "")
    if name in {"img", "audio", "embed", "iframe", "source", "track", "video"}:
        return normalize_whitespace(element.get("src") or "")
    if name == "object":
        return normalize_whitespace(element.get("data") or "")
    if name == "time":
        return normalize_whitespace(element.get("datetime") or element.get_text(" "))
    if name in {"data", "meter"}:
        return normalize_whitespace(element.get("value") or element.get_text(" "))

    return normalize_whitespace(element.get_text(" "))


def _flatten_json_ld(parsed: Any) -> list[dict[str, Any]]:
    """Flatten a parsed JSON-LD payload into a list of objects.

    Unwraps ``@graph`` containers and top-level arrays, keeping the objects they
    hold. Scalars and other non-objects are discarded.
    """
    if isinstance(parsed, dict):
        graph = parsed.get("@graph")
        if isinstance(graph, list):
            nested: list[dict[str, Any]] = []
            for entry in graph:
                nested.extend(_flatten_json_ld(entry))
            # Keep the wrapper's own fields too when it carries more than @graph.
            outer = {k: v for k, v in parsed.items() if k != "@graph"}
            if len(outer) > 1:  # more than a bare @context
                nested.insert(0, outer)
            return nested
        return [parsed]

    if isinstance(parsed, list):
        out: list[dict[str, Any]] = []
        for entry in parsed:
            out.extend(_flatten_json_ld(entry))
        return out

    return []


def _as_tree(source: BeautifulSoup | str | None) -> BeautifulSoup | None:
    """Accept either raw HTML or an already-parsed tree."""
    if source is None:
        return None
    if isinstance(source, str):
        if not source.strip():
            return None
        for parser in _PARSERS:
            try:
                return BeautifulSoup(source, parser)
            except Exception:
                continue
        return None
    return source
