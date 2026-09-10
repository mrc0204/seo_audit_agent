"""Clean main content extractor tailored for retrieval chunking and indexing.

The output feeds Phase 6's chunker. It carries one invariant that the whole
anti-hallucination guarantee rests on:

    **Every block returned here appears verbatim inside ``PageData.text``.**

Blocks are separated by :data:`BLOCK_SEPARATOR` and each block is individually a
literal substring of the full page text. Chunking therefore cannot manufacture a
span that does not exist on the page, and ``qa_validator``'s substring check has a
real chance of passing on genuinely correct answers.

This holds because blocks are never rewritten — only *dropped*. Nav, footer and
other boilerplate are removed whole; the surviving text is normalized by the same
:func:`app.extraction.html_parser.normalize_whitespace` used to build
``PageData.text``. Nothing is joined, summarized, de-hyphenated or re-cased. Any
future change that edits block text breaks Q3 silently, so
``tests/test_extraction.py`` asserts the invariant directly.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from app.extraction.html_parser import element_text, normalize_whitespace

#: Separates retained blocks. A blank line, so Phase 6 can split on it and keep
#: each chunk a substring of the page text.
BLOCK_SEPARATOR = "\n\n"

#: Elements that are chrome rather than content. Removed wholesale before the
#: main-content search, since a nav that survives into the index produces
#: retrieval hits on menu labels instead of prose.
BOILERPLATE_TAGS: tuple[str, ...] = ("nav", "header", "footer", "aside", "form")

#: ``role`` attribute values that mark chrome in ARIA-annotated markup.
BOILERPLATE_ROLES: frozenset[str] = frozenset(
    {"navigation", "banner", "contentinfo", "complementary", "search", "menu", "menubar"}
)

#: Substrings in ``class``/``id`` that reliably mark boilerplate. Kept short and
#: conservative: an over-eager list deletes real content, and a missed nav only
#: adds noise. "menu" and "header" are deliberately absent — they collide with
#: legitimate content such as a restaurant's menu.
BOILERPLATE_HINTS: tuple[str, ...] = (
    "site-nav",
    "navbar",
    "nav-bar",
    "breadcrumb",
    "sidebar",
    "site-footer",
    "page-footer",
    "cookie",
    "consent",
    "skip-link",
    "social-share",
    "share-buttons",
    "newsletter-signup",
)

#: Containers searched, in order, for the page's main content.
MAIN_CONTENT_SELECTORS: tuple[str, ...] = ("main", "article", '[role="main"]', "#content", "#main")

#: Tags whose text becomes one block.
BLOCK_TAGS: tuple[str, ...] = (
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "td",
    "th",
    "dd",
    "dt",
    "blockquote",
    "pre",
    "figcaption",
)

#: Blocks shorter than this are dropped: single words and stray punctuation are
#: retrieval noise, not answers.
MIN_BLOCK_CHARS = 25


def extract_clean_content(soup: BeautifulSoup) -> str:
    """Extract clean main content string from a BeautifulSoup tree for retrieval chunking.

    Args:
        soup: A tree from :func:`app.extraction.html_parser.parse_html`. The tree
            is **not** mutated; boilerplate removal happens on a copy, so the
            caller's tree stays intact for the SEO extractor, which still needs
            the nav links it would otherwise lose.

    Returns:
        Retained blocks joined by :data:`BLOCK_SEPARATOR`. Each block is a literal
        substring of the page's ``PageData.text``. Empty when the page has no
        extractable prose.
    """
    if soup is None:
        return ""

    working = _copy_tree(soup)
    if working is None:
        return ""

    _strip_boilerplate(working)

    root = _main_content_root(working)
    blocks = _blocks_from(root)

    if not blocks:
        # No recognizable block structure (a single unwrapped text node, say).
        # Fall back to the container's own text rather than returning nothing.
        fallback = element_text(root)
        return fallback if len(fallback) >= MIN_BLOCK_CHARS else ""

    return BLOCK_SEPARATOR.join(blocks)


def _copy_tree(soup: BeautifulSoup) -> BeautifulSoup | None:
    """Return an independent copy of ``soup`` so mutations do not leak to callers."""
    try:
        return BeautifulSoup(str(soup), "html.parser")
    except Exception:
        return None


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    """Remove chrome elements from ``soup`` in place.

    Every loop re-checks :func:`_is_live` because ``find_all`` returns a snapshot:
    decomposing a ``<nav>`` destroys the elements inside it, but those children are
    still sitting in a list we are about to iterate. A destroyed tag's ``attrs``
    becomes ``None``, so touching it raises. Real sites nest boilerplate inside
    boilerplate constantly, which is exactly the case a tidy fixture page misses.
    """
    for tag in soup.find_all(BOILERPLATE_TAGS):
        if _is_live(tag):
            tag.decompose()

    for element in soup.find_all(attrs={"role": True}):
        if not _is_live(element):
            continue
        if (element.get("role") or "").strip().lower() in BOILERPLATE_ROLES:
            element.decompose()

    for element in soup.find_all(attrs={"class": True}):
        if not _is_live(element):
            continue
        classes = element.get("class") or []
        if _has_boilerplate_hint(" ".join(classes if isinstance(classes, list) else [classes])):
            element.decompose()

    for element in soup.find_all(attrs={"id": True}):
        if not _is_live(element):
            continue
        if _has_boilerplate_hint(element.get("id") or ""):
            element.decompose()


def _is_live(element) -> bool:
    """True when a tag has not already been destroyed by an earlier decompose()."""
    return element is not None and not getattr(element, "decomposed", False) and element.attrs is not None


def _has_boilerplate_hint(value: str) -> bool:
    """True when a class or id string names a known boilerplate pattern."""
    lowered = value.lower()
    return any(hint in lowered for hint in BOILERPLATE_HINTS)


def _main_content_root(soup: BeautifulSoup):
    """Return the most content-bearing container, preferring semantic markup.

    Falls back to ``<body>``, then the tree itself. Semantic tags are checked
    first because a hand-authored ``<main>`` beats any density heuristic — and
    when several match, the one with the most text wins, since empty ``<main>``
    wrappers are common in template-driven sites.
    """
    best = None
    best_len = 0

    for selector in MAIN_CONTENT_SELECTORS:
        for candidate in soup.select(selector):
            length = len(element_text(candidate))
            if length > best_len:
                best, best_len = candidate, length

    if best is not None and best_len >= MIN_BLOCK_CHARS:
        return best

    return soup.body or soup


def _blocks_from(root) -> list[str]:
    """Return normalized text blocks in document order, without duplicates.

    A block nested inside another retained block (a ``<p>`` inside a ``<li>``)
    would otherwise be emitted twice and skew the retrieval scores, so only the
    outermost of any nested pair is kept.
    """
    if root is None:
        return []

    blocks: list[str] = []
    seen: set[str] = set()

    for element in root.find_all(BLOCK_TAGS):
        if element.find_parent(BLOCK_TAGS) is not None:
            continue

        text = normalize_whitespace(element.get_text(separator=" "))
        if len(text) < MIN_BLOCK_CHARS:
            continue
        if text in seen:
            continue

        seen.add(text)
        blocks.append(text)

    return blocks
