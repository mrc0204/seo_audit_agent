"""HTML parsing utilities for cleaning trees and extracting visible text content.

:func:`extract_visible_text` produces the string that becomes ``PageData.text``, and
that string is the **verification surface for the whole project**. Q3's grounding
guarantee is that a returned excerpt is a literal substring of the page's text, so
every other text-producing function in the codebase must derive from this one and
normalize whitespace exactly the same way. If two modules normalized differently,
a genuinely correct excerpt would fail the substring check and the agent would
return null on questions it could actually answer.

That is why :func:`normalize_whitespace` exists as a single shared helper rather
than each module calling ``" ".join(s.split())`` on its own.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Comment

#: Elements whose text is never visible to a reader and must never reach
#: ``PageData.text``. ``<template>`` holds inert markup; ``<noscript>`` is shown
#: only when scripting is off and is frequently a duplicate of real content.
NON_VISIBLE_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
)

#: Parsers tried in order. ``lxml`` is fast and recovers well from broken markup;
#: ``html.parser`` is the stdlib fallback so a missing lxml build degrades rather
#: than crashing the run.
_PARSERS: tuple[str, ...] = ("lxml", "html.parser")


def parse_html(raw_html: str) -> BeautifulSoup:
    """Parse raw HTML string into a BeautifulSoup tree with scripts and styles stripped.

    Removes non-visible elements and HTML comments in place, so every downstream
    caller sees the same cleaned tree and no module has to remember to re-strip.

    Structured-data extraction is the deliberate exception: JSON-LD lives inside
    ``<script type="application/ld+json">``, which this function removes. That is
    why :func:`app.extraction.schema_extractor.extract_json_ld` parses the *raw*
    HTML rather than this tree.

    Args:
        raw_html: Page source. May be empty, truncated, or malformed.

    Returns:
        A cleaned tree. Never raises — unparseable input yields an empty tree,
        because one broken page must not end a crawl of two hundred.
    """
    if not raw_html:
        return BeautifulSoup("", _PARSERS[-1])

    soup: BeautifulSoup | None = None
    for parser in _PARSERS:
        try:
            soup = BeautifulSoup(raw_html, parser)
            break
        except Exception:
            continue

    if soup is None:
        return BeautifulSoup("", _PARSERS[-1])

    for tag in soup.find_all(NON_VISIBLE_TAGS):
        tag.decompose()

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    return soup


def extract_visible_text(soup: BeautifulSoup) -> str:
    """Extract clean visible textual content from a parsed BeautifulSoup object.

    Text is joined with single spaces and collapsed, matching the
    ``PageData.text`` contract ("visible text, whitespace-normalized").

    Args:
        soup: A tree from :func:`parse_html`.

    Returns:
        The page's visible text, or ``""`` for an empty tree.
    """
    if soup is None:
        return ""
    return normalize_whitespace(soup.get_text(separator=" "))


def normalize_whitespace(value: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends.

    The single definition of "whitespace-normalized" for the entire project. Q3's
    substring guarantee depends on every module agreeing on this exact rule.

    Args:
        value: Any string.

    Returns:
        The normalized string, or ``""`` when ``value`` is empty or ``None``.
    """
    if not value:
        return ""
    return " ".join(value.split())


def element_text(element) -> str:
    """Return one element's normalized visible text.

    Uses the same separator and normalization as :func:`extract_visible_text`, so
    the result is guaranteed to appear verbatim inside the full page text.

    Args:
        element: A BeautifulSoup tag, or ``None``.

    Returns:
        The element's normalized text, or ``""``.
    """
    if element is None:
        return ""
    return normalize_whitespace(element.get_text(separator=" "))
