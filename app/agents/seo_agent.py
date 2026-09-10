"""SEO agent executing deterministic auditing rules and generating findings across crawled pages.

Every finding in ``audit.json`` originates here, from a **pure function over
``PageData``** — never from a model. Each check owns a ``check_id`` so Phase 7's
validator can re-run exactly the rule that fired and confirm it still does.

The LLM's role is deliberately tiny and enforced structurally rather than by
instruction. :func:`apply_llm_suggested_fixes` accepts finished ``Finding`` objects
and rebuilds each one with only ``suggested_fix`` replaced; ``metric``, ``page``,
``evidence`` and ``check_id`` are copied from the original, so a model that tries to
edit them has no channel to do so. A returned fix that is empty, absurdly long, or
merely echoes the evidence is discarded in favour of the deterministic text. This
answers the plan's third risk: LLM scope creep is prevented at the code level.

Severity is calibrated against one question, deliberately narrower than "is this
worth fixing": **does this actually affect whether a search engine can index,
understand, or trust the page** — not "is this good practice"? A great many
best-practice recommendations are true and worth doing without being *indexability*
problems, and conflating the two overstates the report. Concretely:

* ``critical`` — the page is excluded from search outright (``noindex``, a broken
  or absent ``<title>``), or the crawl found a resource actually broken (a 404'd
  internal link, conflicting canonical signals).
* ``warning`` — a real, verifiable defect with crawler-visible consequences short
  of exclusion: duplicate content/titles/descriptions across pages (ranking
  cannibalization), a ``nofollow`` directive (link equity), a redirect chain
  (crawl budget), missing image ``alt`` (accessibility *and* image-search
  indexing), more than one ``<title>`` element (a real HTML5 validity violation).
* ``info`` — true by threshold or convention, not by rule, and does **not** affect
  indexability even when fixing it is still good advice. Meta description length
  and presence, canonical presence on a single page, H1 count, title length: none
  of these stop a page from being crawled, indexed or ranked — Google has said
  outright that meta description is not a ranking factor — they affect the SERP
  snippet, social preview, or authoring hygiene. Reported because a careful
  auditor would still mention them, at a severity that doesn't overstate what
  they are.

This distinction is enforced only through the ``severity`` value — the brief's
locked ``audit.json`` shape has no room for a separate category field, and
``severity`` is exactly the mechanism it already provides for saying how much a
finding matters.

Thresholds live in module constants and every finding quotes the number it used, so
a reader can disagree with a threshold without doubting the measurement.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

from app.crawler.url_utils import is_same_domain
from app.extraction.html_parser import parse_html
from app.extraction.seo_extractor import extract_seo_fields
from app.models.finding import Finding
from app.models.page import PageData

#: Google truncates titles near this width. Longer is not broken, only truncated,
#: so it is a warning rather than a critical.
TITLE_MAX_CHARS = 60

#: Below this, a title usually lacks brand or keyword context.
TITLE_MIN_CHARS = 30

#: Meta descriptions are truncated in the SERP past roughly this width.
META_DESCRIPTION_MAX_CHARS = 160

#: Very short descriptions waste the snippet.
META_DESCRIPTION_MIN_CHARS = 70

#: Word count below which a page is flagged as thin. Heuristic, hence ``info``:
#: a contact page is legitimately short and is not a defect.
THIN_CONTENT_WORD_COUNT = 150

#: Redirect hops permitted before the chain is reported. One hop (http->https, or a
#: trailing-slash fix) is normal and healthy; two or more wastes crawl budget.
MAX_REDIRECT_HOPS = 1

#: Longest ``suggested_fix`` accepted from an LLM. Beyond this is a model monologue,
#: not a fix.
MAX_SUGGESTED_FIX_CHARS = 400

#: Evidence snippets are truncated to this width so one finding cannot carry an
#: entire page into the report.
EVIDENCE_SNIPPET_CHARS = 200


def run_seo_audit(
    pages: list[Any],
    fetch_results: list[Any] | None = None,
) -> list[Any]:
    """Perform deterministic SEO audit checks on crawled web pages.

    Runs every per-page check against each page, then the cross-page checks that
    need the whole site (duplicate titles, duplicate descriptions, duplicate
    content, broken internal links).

    Args:
        pages: ``PageData`` objects from Phase 3, one per crawled page.
        fetch_results: Optional ``FetchResult`` objects from Phase 2. Only the
            redirect-chain check needs them, because ``PageData`` records where a
            page ended up but not how many hops it took. Omitting them skips that
            one check rather than guessing.

    Returns:
        ``Finding`` objects sorted by severity then page, so the most serious
        issues lead the report.
    """
    pages = [p for p in pages if isinstance(p, PageData)]
    findings: list[Finding] = []

    for page in pages:
        extras = page_extras(page)
        for check in PAGE_CHECKS:
            findings.extend(check(page, extras))

    for site_check in SITE_CHECKS:
        findings.extend(site_check(pages))

    findings.extend(check_broken_internal_links(pages, fetch_results=fetch_results))

    if fetch_results:
        findings.extend(check_redirect_chains(fetch_results))

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(findings, key=lambda f: (order.get(f.severity, 3), f.page, f.check_id))


def page_extras(page: PageData) -> dict[str, Any]:
    """Re-derive the per-page counts that ``PageData`` has no field for.

    ``PageData`` stores one canonical, one title and one meta description because
    that is what a correct page has. Detecting *duplicates within a page* needs the
    counts, so they are recomputed from ``page.html`` here rather than widening the
    locked Phase 1 contract.

    Args:
        page: The page to inspect.

    Returns:
        The dict from :func:`app.extraction.seo_extractor.extract_seo_fields`.
    """
    return extract_seo_fields(parse_html(page.html), page.final_url)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------


def check_title(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """Missing, empty, repeated, over-long or over-short ``<title>``."""
    title = page.title

    if title is None:
        return [
            _finding(
                "missing_title",
                page,
                "critical",
                "No <title> element is present in the document <head>.",
                "Add a unique <title> of roughly 30-60 characters describing this page.",
                "SEO-TITLE-001",
            )
        ]

    if not title.strip():
        return [
            _finding(
                "empty_title",
                page,
                "critical",
                "A <title> element is present but contains no text.",
                "Replace the empty <title> with a unique, descriptive title.",
                "SEO-TITLE-002",
            )
        ]

    findings: list[Finding] = []

    if extras.get("title_count", 1) > 1:
        findings.append(
            _finding(
                "multiple_titles",
                page,
                "warning",
                f"The document contains {extras['title_count']} <title> elements; "
                f"the first is {_quote(title)}.",
                "Keep exactly one <title> element in <head> and remove the rest.",
                "SEO-TITLE-003",
            )
        )

    length = len(title)
    if length > TITLE_MAX_CHARS:
        findings.append(
            _finding(
                "title_too_long",
                page,
                "info",
                f"<title> is {length} characters, over the {TITLE_MAX_CHARS}-character "
                f"display limit: {_quote(title)}",
                f"Shorten the title to {TITLE_MAX_CHARS} characters or fewer so it is not "
                "truncated in search results.",
                "SEO-TITLE-004",
            )
        )
    elif length < TITLE_MIN_CHARS:
        findings.append(
            _finding(
                "title_too_short",
                page,
                "info",
                f"<title> is only {length} characters: {_quote(title)}",
                f"Expand the title towards {TITLE_MIN_CHARS}-{TITLE_MAX_CHARS} characters "
                "with distinguishing detail.",
                "SEO-TITLE-005",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Meta description
# ---------------------------------------------------------------------------


def check_meta_description(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """Missing, empty, repeated, over-long or over-short meta description.

    All ``info``: Google has stated meta description is not a ranking factor and
    its absence never affects indexability. It governs only the SERP snippet, so
    every finding here is advisory rather than a defect.
    """
    description = page.meta_description

    if description is None:
        return [
            _finding(
                "missing_meta_description",
                page,
                "info",
                'No <meta name="description"> element is present in the document <head>.',
                f"Add a unique meta description of about "
                f"{META_DESCRIPTION_MIN_CHARS}-{META_DESCRIPTION_MAX_CHARS} characters.",
                "SEO-META-001",
            )
        ]

    if not description.strip():
        return [
            _finding(
                "empty_meta_description",
                page,
                "info",
                'A <meta name="description"> element is present but its content is empty.',
                "Write a description summarising this page's content.",
                "SEO-META-002",
            )
        ]

    findings: list[Finding] = []

    if extras.get("meta_description_count", 1) > 1:
        findings.append(
            _finding(
                "multiple_meta_descriptions",
                page,
                "info",
                f"The document contains {extras['meta_description_count']} description "
                "meta tags.",
                "Keep exactly one meta description and remove the others.",
                "SEO-META-003",
            )
        )

    length = len(description)
    if length > META_DESCRIPTION_MAX_CHARS:
        findings.append(
            _finding(
                "meta_description_too_long",
                page,
                "info",
                f"Meta description is {length} characters, over the "
                f"{META_DESCRIPTION_MAX_CHARS}-character display limit: {_quote(description)}",
                f"Trim the description to {META_DESCRIPTION_MAX_CHARS} characters or fewer.",
                "SEO-META-004",
            )
        )
    elif length < META_DESCRIPTION_MIN_CHARS:
        findings.append(
            _finding(
                "meta_description_too_short",
                page,
                "info",
                f"Meta description is only {length} characters: {_quote(description)}",
                f"Expand the description towards {META_DESCRIPTION_MAX_CHARS} characters.",
                "SEO-META-005",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Canonical
# ---------------------------------------------------------------------------


def check_canonical(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """Missing canonical, several canonicals, or one pointing off-domain.

    Severities diverge sharply within this one check, on purpose. A single missing
    canonical is only ``info`` — a canonical only matters once duplicate URLs
    exist, and a lone page has nothing to consolidate. *Multiple* canonicals is
    ``critical`` — that is an active, conflicting signal search engines may
    respond to by ignoring all of them. An off-domain canonical is ``warning`` — a
    real, verifiable indexing risk (it tells engines to index a different site
    instead), but plausibly deliberate (syndicated content), so short of critical.
    """
    count = extras.get("canonical_count", 0)

    if count == 0:
        return [
            _finding(
                "missing_canonical",
                page,
                "info",
                'No <link rel="canonical"> element is present in the document <head>.',
                "Add a self-referencing canonical link to consolidate duplicate URLs.",
                "SEO-CANON-001",
            )
        ]

    findings: list[Finding] = []

    if count > 1:
        findings.append(
            _finding(
                "multiple_canonicals",
                page,
                "critical",
                f'The document contains {count} <link rel="canonical"> elements; '
                f"the first points to {page.canonical}.",
                "Keep exactly one canonical link. Search engines may ignore all of them "
                "when several conflict.",
                "SEO-CANON-002",
            )
        )

    canonical = page.canonical
    if canonical and not is_same_domain(canonical, page.final_url):
        findings.append(
            _finding(
                "canonical_off_domain",
                page,
                "warning",
                f"Canonical points to {canonical}, which is on a different domain from "
                f"the page itself ({page.final_url}).",
                "Confirm this cross-domain canonical is intentional; if not, point it at "
                "this page's own URL. It tells search engines to index the other site "
                "instead of this one.",
                "SEO-CANON-003",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Indexability
# ---------------------------------------------------------------------------


def check_robots_meta(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """``noindex`` / ``nofollow`` from an HTML ``<meta name="robots">`` tag.

    Reported, not assumed wrong. A staging or thank-you page is *supposed* to carry
    ``noindex``, so the fix text asks for confirmation rather than instructing
    removal — the check can see the directive but not the intent.

    Reads only ``extras["robots_meta"]`` (recomputed straight from ``page.html``),
    not ``page.robots_meta`` (which also carries any ``X-Robots-Tag`` response
    header directive merged in by Phase 3) — a header-sourced directive is a
    materially different, more surprising situation for a site owner, and is
    reported separately by :func:`check_http_header_robots` with evidence that
    says so explicitly, rather than being folded into this HTML-only finding.
    """
    directives = [d.lower() for d in extras.get("robots_meta", [])]
    findings: list[Finding] = []

    if "noindex" in directives:
        findings.append(
            _finding(
                "noindex_directive",
                page,
                "critical",
                f'Robots meta directives on this page are: {", ".join(directives)}. '
                '"noindex" excludes it from search results.',
                "Confirm this page is meant to be hidden from search. If it should be "
                'indexed, remove "noindex" from the robots meta tag.',
                "SEO-INDEX-001",
            )
        )

    if "nofollow" in directives:
        findings.append(
            _finding(
                "nofollow_directive",
                page,
                "warning",
                f'Robots meta directives on this page are: {", ".join(directives)}. '
                '"nofollow" stops link equity flowing to linked pages.',
                'Confirm this is intentional; otherwise remove "nofollow" so internal '
                "links are followed.",
                "SEO-INDEX-002",
            )
        )

    return findings


def check_http_header_robots(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """``noindex`` / ``nofollow`` sent only via the ``X-Robots-Tag`` HTTP response header.

    A real, not hypothetical, crawler concern: a server can direct a page to be
    ``noindex`` purely at the HTTP layer, with nothing in the page's own HTML to
    show for it — the site owner looking at "view source" would see nothing wrong.
    This is exactly the case ``check_robots_meta`` cannot see, since it reads only
    HTML. The two checks are complementary and deliberately non-overlapping: a
    directive present in the HTML meta tag is reported by that check; a directive
    present *only* via the header is reported here, with evidence that names the
    header explicitly so the finding doesn't read as a duplicate or a mystery.

    ``page.robots_meta`` (the Phase 3 merge of HTML + header directives) minus
    ``extras["robots_meta"]`` (HTML only, recomputed fresh from ``page.html``) is
    exactly the set of directives that could only have come from the header.
    """
    html_directives = {d.lower() for d in extras.get("robots_meta", [])}
    header_only = [d.lower() for d in page.robots_meta if d.lower() not in html_directives]
    findings: list[Finding] = []

    if "noindex" in header_only:
        findings.append(
            _finding(
                "noindex_via_http_header",
                page,
                "critical",
                'The X-Robots-Tag HTTP response header includes "noindex" for this page, '
                "though nothing in the page's own HTML says so. This excludes the page "
                "from search results just as effectively as an HTML robots meta tag would.",
                "Confirm this is intentional. Search-engine crawlers respect this header "
                "even though it is invisible in the page source — check the server or CDN "
                "configuration if it should not be there.",
                "SEO-INDEX-003",
            )
        )

    if "nofollow" in header_only:
        findings.append(
            _finding(
                "nofollow_via_http_header",
                page,
                "warning",
                'The X-Robots-Tag HTTP response header includes "nofollow" for this page, '
                "though nothing in the page's own HTML says so.",
                "Confirm this is intentional; otherwise remove it from the server or CDN "
                "configuration so internal links are followed.",
                "SEO-INDEX-004",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


def check_headings(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """No H1, several H1s, an empty heading, or a skipped level.

    All ``info``: search engines do not require an H1 to index or rank a page, and
    HTML5 explicitly permits more than one in sectioning content. These are
    semantic-authoring and legacy-SEO-advice concerns, not indexability defects.
    """
    headings = page.headings or {}
    h1s = headings.get("h1", [])
    findings: list[Finding] = []

    if not h1s:
        findings.append(
            _finding(
                "missing_h1",
                page,
                "info",
                "The page contains no <h1> element.",
                "Add a single <h1> naming the page's main topic.",
                "SEO-HEAD-001",
            )
        )
    elif len(h1s) > 1:
        findings.append(
            _finding(
                "multiple_h1",
                page,
                "info",
                f"The page contains {len(h1s)} <h1> elements: "
                + ", ".join(_quote(h) for h in h1s[:4]),
                "Keep one <h1> as the page's main heading and demote the others to <h2>.",
                "SEO-HEAD-002",
            )
        )

    for level in sorted(headings):
        empties = sum(1 for text in headings[level] if not text.strip())
        if empties:
            findings.append(
                _finding(
                    "empty_heading",
                    page,
                    "info",
                    f"The page contains {empties} empty <{level}> element(s).",
                    f"Remove the empty <{level}> element(s) or give them text.",
                    "SEO-HEAD-003",
                )
            )

    present = sorted(int(level[1]) for level, texts in headings.items() if texts)
    for previous, current in zip(present, present[1:]):
        if current - previous > 1:
            findings.append(
                _finding(
                    "skipped_heading_level",
                    page,
                    "info",
                    f"Heading levels jump from <h{previous}> to <h{current}> with no "
                    f"<h{previous + 1}> in between.",
                    f"Insert an <h{previous + 1}> or promote the <h{current}> so the "
                    "outline is contiguous.",
                    "SEO-HEAD-004",
                )
            )
            break

    return findings


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def check_image_alt(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """Images with no ``alt`` attribute at all.

    ``alt=""`` is deliberately **not** reported. It is the correct, specified way to
    mark a decorative image, and flagging it would tell a site to undo work it has
    already done properly. Only a genuinely absent attribute is a finding — which is
    why Phase 3 preserves ``None`` and ``""`` as distinct values.
    """
    missing = [image for image in page.images if image.alt is None]
    if not missing:
        return []

    listed = ", ".join(image.src for image in missing[:5] if image.src)
    suffix = f" and {len(missing) - 5} more" if len(missing) > 5 else ""

    return [
        _finding(
            "image_missing_alt",
            page,
            "warning",
            f"{len(missing)} of {len(page.images)} <img> elements have no alt attribute: "
            f"{listed}{suffix}",
            'Add descriptive alt text to each image, or alt="" if it is purely decorative.',
            "SEO-IMG-001",
        )
    ]


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def check_thin_content(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """Very low word count. ``info`` only — this is a threshold, not a rule."""
    words = len(page.text.split())
    if words >= THIN_CONTENT_WORD_COUNT:
        return []

    return [
        _finding(
            "thin_content",
            page,
            "info",
            f"The page has {words} words of visible text, below the "
            f"{THIN_CONTENT_WORD_COUNT}-word threshold used here.",
            "Confirm this page has enough substance to rank. Short utility pages such as "
            "contact or login pages are often legitimately brief.",
            "SEO-CONTENT-001",
        )
    ]


def check_open_graph(page: PageData, extras: dict[str, Any]) -> list[Finding]:
    """Missing Open Graph title/description/image, which govern social previews."""
    soup = parse_html(page.html)
    present = {
        (tag.get("property") or "").strip().lower()
        for tag in soup.find_all("meta")
        if tag.get("property")
    }

    missing = sorted({"og:title", "og:description", "og:image"} - present)
    if not missing:
        return []

    return [
        _finding(
            "missing_open_graph",
            page,
            "info",
            f"Open Graph tags absent from <head>: {', '.join(missing)}.",
            "Add the missing og: meta tags so shared links render a title, summary and "
            "image.",
            "SEO-SOCIAL-001",
        )
    ]


# ---------------------------------------------------------------------------
# Cross-page checks
# ---------------------------------------------------------------------------


def check_duplicate_titles(pages: list[PageData]) -> list[Finding]:
    """The same non-empty ``<title>`` on more than one page."""
    return _duplicate_field(
        pages,
        value_of=lambda p: (p.title or "").strip(),
        metric="duplicate_title",
        check_id="SEO-DUP-001",
        label="<title>",
        fix="Give each page a distinct title describing its own content.",
    )


def check_duplicate_meta_descriptions(pages: list[PageData]) -> list[Finding]:
    """The same non-empty meta description on more than one page."""
    return _duplicate_field(
        pages,
        value_of=lambda p: (p.meta_description or "").strip(),
        metric="duplicate_meta_description",
        check_id="SEO-DUP-002",
        label="meta description",
        fix="Write a distinct meta description for each page.",
    )


def check_duplicate_content(pages: list[PageData]) -> list[Finding]:
    """Identical visible text on more than one URL, via ``content_hash``.

    Empty pages are excluded: several blank pages share a hash trivially, and that
    is a different defect already covered by the thin-content check.
    """
    by_hash: dict[str, list[PageData]] = defaultdict(list)
    for page in pages:
        if page.text.strip():
            by_hash[page.content_hash].append(page)

    findings: list[Finding] = []
    for group in by_hash.values():
        if len(group) < 2:
            continue
        urls = [p.final_url for p in group]
        for page in group:
            others = [u for u in urls if u != page.final_url]
            findings.append(
                _finding(
                    "duplicate_content",
                    page,
                    "warning",
                    f"This page's visible text is identical to {len(others)} other "
                    f"page(s): {', '.join(others[:3])}",
                    "Consolidate these URLs, or set a canonical link pointing at the one "
                    "version that should be indexed.",
                    "SEO-DUP-003",
                )
            )

    return findings


def check_broken_internal_links(
    pages: list[PageData], fetch_results: Iterable[Any] | None = None
) -> list[Finding]:
    """Internal links whose target returned 4xx or 5xx **during this crawl**.

    Only targets actually fetched are judged. A link to a page the crawl never
    reached is not evidence of breakage — it may simply have fallen past
    ``max_pages`` — and reporting it would be a finding the evidence does not
    support.

    Args:
        pages: HTML pages the audit ran against.
        fetch_results: Optional ``FetchResult`` objects from Phase 2, covering
            **every** URL the crawl fetched — including non-HTML ones (a PDF, a
            markdown file) that never became a ``PageData`` at all. Without these,
            a genuinely broken link to a non-HTML resource looks identical to a
            link the crawl simply never reached, and is silently under-reported.
            ``pages`` alone is enough to catch a broken link between two ordinary
            pages, which is why this argument is optional.
    """
    status_by_key: dict[str, tuple[int, str]] = {}
    for page in pages:
        for key in {_link_key(page.url), _link_key(page.final_url)}:
            status_by_key[key] = (page.status_code, page.final_url)
    for result in fetch_results or []:
        url = getattr(result, "url", None)
        final_url = getattr(result, "final_url", None)
        status_code = getattr(result, "status_code", None)
        if not final_url or status_code is None:
            continue
        for key in {_link_key(u) for u in (url, final_url) if u}:
            status_by_key.setdefault(key, (status_code, final_url))

    findings: list[Finding] = []
    for page in pages:
        broken: list[tuple[str, int]] = []
        for link in page.links:
            if not link.is_internal:
                continue
            entry = status_by_key.get(_link_key(link.href))
            if entry and entry[0] >= 400:
                broken.append((link.href, entry[0]))

        if not broken:
            continue

        listed = ", ".join(f"{href} ({status})" for href, status in broken[:5])
        suffix = f" and {len(broken) - 5} more" if len(broken) > 5 else ""
        findings.append(
            _finding(
                "broken_internal_link",
                page,
                "critical",
                f"{len(broken)} internal link(s) on this page returned an error status "
                f"during the crawl: {listed}{suffix}",
                "Repair or remove these links, or redirect the targets to live URLs.",
                "SEO-LINK-001",
            )
        )

    return findings


def check_redirect_chains(fetch_results: Iterable[Any]) -> list[Finding]:
    """Redirect chains longer than :data:`MAX_REDIRECT_HOPS`.

    Needs Phase 2's ``FetchResult``: ``PageData`` records where a page ended up, not
    how many hops it took to get there.
    """
    findings: list[Finding] = []

    for result in fetch_results:
        chain = list(getattr(result, "redirect_chain", []) or [])
        if len(chain) <= MAX_REDIRECT_HOPS:
            continue

        hops = " -> ".join(chain + [result.final_url])
        findings.append(
            Finding(
                metric="redirect_chain",
                page=result.final_url,
                severity="warning",
                evidence=f"{len(chain)} redirect hops before the final URL: {hops}",
                suggested_fix="Point the first URL directly at the final destination so "
                "the chain is a single hop.",
                check_id="SEO-REDIR-001",
            )
        )

    return findings


#: Per-page checks, each ``(PageData, extras) -> list[Finding]``.
PAGE_CHECKS: tuple[Callable[[PageData, dict[str, Any]], list[Finding]], ...] = (
    check_title,
    check_meta_description,
    check_canonical,
    check_robots_meta,
    check_http_header_robots,
    check_headings,
    check_image_alt,
    check_thin_content,
    check_open_graph,
)

#: Cross-page checks, each ``(list[PageData]) -> list[Finding]``.
#: check_broken_internal_links is deliberately absent from this tuple — it takes
#: an optional fetch_results argument the other site checks don't, so
#: run_seo_audit calls it explicitly instead of through this uniform-arity loop.
SITE_CHECKS: tuple[Callable[[list[PageData]], list[Finding]], ...] = (
    check_duplicate_titles,
    check_duplicate_meta_descriptions,
    check_duplicate_content,
)


# ---------------------------------------------------------------------------
# The LLM boundary
# ---------------------------------------------------------------------------

#: Default prompt for the ``suggested_fix`` rewrite. It forbids restating the
#: evidence, which the model cannot alter in any case.
DEFAULT_SUGGESTED_FIX_PROMPT = (
    "A deterministic SEO check found this issue. Write one or two plain sentences "
    "telling the site owner how to fix it.\n\n"
    "Issue: {metric}\n"
    "Severity: {severity}\n"
    "Page: {page}\n"
    "Evidence: {evidence}\n\n"
    "Reply with the fix instruction only. Do not restate the evidence, do not add "
    "caveats, and do not describe any issue other than this one."
)


def apply_llm_suggested_fixes(
    findings: list[Finding],
    generate: Callable[[str], str] | None,
    prompt_template: str | None = None,
) -> list[Finding]:
    """Let an LLM rewrite ``suggested_fix`` — and nothing else.

    The immutability is structural, not instructional. Each finding is rebuilt from
    the *original* object with only ``suggested_fix`` substituted, so a model that
    tries to alter ``metric``, ``page``, ``evidence`` or ``check_id`` has no channel
    to reach them. The deterministic text is kept whenever the replacement is empty,
    over :data:`MAX_SUGGESTED_FIX_CHARS`, or merely echoes the evidence.

    Any exception from ``generate`` is swallowed per finding: a flaky free-tier LLM
    must degrade the phrasing of a fix, never the audit itself.

    Args:
        findings: Completed deterministic findings.
        generate: Callable taking a prompt and returning text. ``None`` returns the
            findings unchanged, which is the offline default.
        prompt_template: Format string with ``{metric}``, ``{page}``, ``{severity}``
            and ``{evidence}`` placeholders. Defaults to
            :data:`DEFAULT_SUGGESTED_FIX_PROMPT`.

    Returns:
        A new list. Input findings are never mutated.
    """
    if generate is None:
        return list(findings)

    template = prompt_template or DEFAULT_SUGGESTED_FIX_PROMPT
    rewritten: list[Finding] = []

    for finding in findings:
        try:
            candidate = generate(
                template.format(
                    metric=finding.metric,
                    page=finding.page,
                    severity=finding.severity,
                    evidence=finding.evidence,
                )
            )
            replacement = _acceptable_fix(candidate, finding)
        except Exception:
            replacement = None

        rewritten.append(
            finding.model_copy(update={"suggested_fix": replacement})
            if replacement
            else finding
        )

    return rewritten


def _acceptable_fix(candidate: str | None, finding: Finding) -> str | None:
    """Return a usable ``suggested_fix``, or ``None`` to keep the deterministic one."""
    if not candidate:
        return None

    cleaned = " ".join(str(candidate).split())
    if not cleaned or len(cleaned) > MAX_SUGGESTED_FIX_CHARS:
        return None
    if cleaned == finding.evidence.strip():
        return None

    return cleaned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finding(
    metric: str,
    page: PageData,
    severity: str,
    evidence: str,
    suggested_fix: str,
    check_id: str,
) -> Finding:
    """Build a Finding, citing the page by its post-redirect URL.

    ``final_url`` is used deliberately: it is where the evidence was actually
    observed, and Phase 7's validator re-reads the page from there.
    """
    return Finding(
        metric=metric,
        page=page.final_url,
        severity=severity,
        evidence=evidence,
        suggested_fix=suggested_fix,
        check_id=check_id,
    )


def _duplicate_field(
    pages: list[PageData],
    value_of: Callable[[PageData], str],
    metric: str,
    check_id: str,
    label: str,
    fix: str,
) -> list[Finding]:
    """Report a field shared by two or more pages. Empty values are never duplicates."""
    groups: dict[str, list[PageData]] = defaultdict(list)
    for page in pages:
        value = value_of(page)
        if value:
            groups[value].append(page)

    findings: list[Finding] = []
    for value, group in groups.items():
        if len(group) < 2:
            continue
        urls = [p.final_url for p in group]
        for page in group:
            others = [u for u in urls if u != page.final_url]
            findings.append(
                _finding(
                    metric,
                    page,
                    "warning",
                    f"This page's {label} {_quote(value)} is also used on "
                    f"{len(others)} other page(s): {', '.join(others[:3])}",
                    fix,
                    check_id,
                )
            )

    return findings


def _link_key(url: str) -> str:
    """Compare link targets ignoring a trailing slash, which servers treat alike."""
    return url.rstrip("/") or url


def _quote(value: str) -> str:
    """Quote a verbatim snippet, truncating very long ones for readability."""
    text = (
        value if len(value) <= EVIDENCE_SNIPPET_CHARS else value[:EVIDENCE_SNIPPET_CHARS] + "..."
    )
    return f'"{text}"'
