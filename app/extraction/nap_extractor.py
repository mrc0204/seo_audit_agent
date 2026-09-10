"""Name, Address, and Phone (NAP) candidate extractor.

Candidates are gathered in the plan's priority order — JSON-LD, then microdata, then
``tel:`` links, then visible text — and every one is tagged with the ``source`` it
came from, so a verdict can say not just *what* disagreed but *where* each value was
written.

The governing rule: **a candidate must appear literally on the page.** Nothing here
infers, completes or corrects a value. A phone number with a missing digit stays
missing; an address split across two elements is joined only by whitespace that was
already between them. That is what lets Phase 7's ``nap_validator`` re-find every
``raw_value`` in the page's own text or structured data.

Visible text is the loosest source and therefore the most tightly gated: phone
numbers must match a phone-shaped pattern *and* pass a digit-count test, addresses
must contain a house number followed by a street-type word, and names are read only
from a copyright line. Anything vaguer would put guesses into evidence.

A fifth, optional source — :func:`extract_nap_candidates_via_llm` — exists for real
sites where the value is genuinely stated but in a shape none of the four regex/schema
tiers above recognize (a plain header/logo name with no copyright line or schema; a
"City, State" address with no street or generic suffix word at all). It is never a
replacement for the tiers above and the same governing rule still applies, enforced
mechanically rather than by trusting the model: a claimed value is only kept if it is
a literal, whitespace-normalized substring of the page's own text — the identical
check :mod:`app.validation.qa_validator` uses for Q3. See
:func:`app.agents.nap_agent.run_nap_check` for when this source actually runs (only
for a field with zero candidates from every other source, and only against a small,
targeted subset of pages — never every page in a large crawl).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from app.extraction.html_parser import element_text, normalize_whitespace, parse_html
from app.extraction.normalize import (
    normalize_address,
    normalize_name,
    normalize_phone,
    strip_legal_suffix,
)
from app.extraction.schema_extractor import extract_microdata
from app.models.nap import NAPValue

#: JSON-LD / microdata ``@type`` values that describe the site's own business. A
#: ``Product`` or ``Article`` block may carry a phone number belonging to someone
#: else entirely, so only these types are trusted.
BUSINESS_TYPES: frozenset[str] = frozenset(
    {
        "localbusiness",
        "organization",
        "corporation",
        "store",
        "restaurant",
        "cafe",
        "hotel",
        "medicalbusiness",
        "dentist",
        "physician",
        "professionalservice",
        "homeandconstructionbusiness",
        "automotivebusiness",
        "foodestablishment",
        "healthandbeautybusiness",
        "legalservice",
        "financialservice",
        "educationalorganization",
        "ngo",
        "sportsactivitylocation",
        "entertainmentbusiness",
        "lodgingbusiness",
        "travelagency",
        "realestateagent",
        "childcare",
        "emergencyservice",
        "employmentagency",
        "insuranceagency",
        "library",
        "selfstorage",
        "shoppingcenter",
        "touristinformationcenter",
    }
)

#: Ordered fields of a schema.org ``PostalAddress``. Joined in this order so two
#: pages listing the same address produce the same string regardless of key order.
ADDRESS_PARTS: tuple[str, ...] = (
    "streetAddress",
    "addressLocality",
    "addressRegion",
    "postalCode",
    "addressCountry",
)

#: Phone-shaped runs in visible text: an optional ``+``, then digits separated by
#: spaces, dots, hyphens or brackets. Requires at least one of ``()``, ``-``, ``.``
#: or a leading ``+`` (:data:`_PHONE_PUNCTUATION`) — plain space-separated digit
#: runs are too weak a signal on their own and false-positive on dates ("2026-09-17"
#: passes the digit-count gate) and numeric sequences in code samples or tables,
#: which real pages contain far more often than a bare unformatted phone number.
PHONE_PATTERN = re.compile(r"\+?\d[\d\s().\-]{6,20}\d")
_PHONE_PUNCTUATION = re.compile(r"[()\-.]|^\+")
#: ISO-shaped dates (2026-09-17, with or without stray internal spaces from the
#: whitespace-normalized text) pass both the digit-count gate and the punctuation
#: gate above, since a hyphen is phone-typical too. Excluded explicitly rather than
#: tightening the phone pattern further, which would also start rejecting real
#: hyphenated numbers.
_DATE_SHAPE = re.compile(r"^\d{4}\s*-\s*\d{2}\s*-\s*\d{2}$")
_DIGIT_GROUP = re.compile(r"\d+")

#: An address-like run: a house number, up to five words, then a street-type word.
#: Requiring the street-type word keeps prices, dates and bare "Suite 4" out.
ADDRESS_PATTERN = re.compile(
    # Up to 3 filler words, not 5: real addresses rarely put more than a couple of
    # directional/name words between the house number and the street type ("100
    # North Main Street"), and every additional word the gap allows is another
    # chance to match ordinary prose that happens to end near a street-type word.
    #
    # The leading number requires at least 2 digits, not 1: a single-digit
    # "house number" is real but rare, and several street-type abbreviations
    # here double as everyday short units elsewhere — "ct" is also retail
    # shorthand for "count". Found live on a real e-commerce grocery site: every
    # product page's "4 ct" (a 4-pack quantity label) matched as if it were
    # "4 Court", producing a confidently wrong address at 1.0 confidence, not
    # merely noisy evidence — worse than a value sitting in an already-uncertain
    # field. Requiring 2+ digits costs real single-digit-numbered addresses
    # (uncommon for the commercial properties this tool targets) in exchange for
    # closing off the much more frequent single-digit-quantity-label collision.
    r"\b\d{2,6}[a-zA-Z]?\s+(?:[A-Za-z0-9.'\-]+\s+){0,3}"
    r"(?:street|st|road|rd|avenue|ave|boulevard|blvd|drive|dr|lane|ln|court|ct|"
    r"place|pl|square|sq|terrace|ter|parkway|pkwy|highway|hwy|way|close|cl|"
    r"crescent|cres|gardens|gdns)\b",
    re.IGNORECASE,
)

#: A copyright line — the one place a business name reliably appears in visible text.
#: The name tokens exclude ``.`` so the match stops at the end of the sentence:
#: "Copyright 2026 Ridgeline Coffee Roasters. All rights reserved" must capture the
#: business, not run on into the boilerplate that follows it.
#: The ``(?i:...)`` scope is deliberate: only the "copyright"/"by" markers are
#: case-insensitive. Making the whole pattern ignore case would let ``[A-Z]`` match
#: lowercase words, so the capture would run on into ordinary prose.
#:
#: A trailing year is effectively required before the name capture, and this is
#: load-bearing, not cosmetic. The bare word "copyright" appears constantly in
#: legal-boilerplate footers ("...pursuant to the Digital Millennium Copyright
#: Act...", "report a copyright infringement claim...") with no notice intended at
#: all, and without the year gate the capture grabbed whatever capitalized phrase
#: happened to follow — "Act", "Infringement Dispute Resolution" — as if it were
#: the business name. Found live on a real Shopify store's footer during the Phase
#: 10 sweep, contaminating the Q2 name comparison with garbage. A 4-digit year is
#: what distinguishes an actual "© 2026 Business Name" notice from that noise; the
#: bare ``©`` glyph is kept as an alternate, year-optional trigger since the symbol
#: itself is unambiguous in a way the word "copyright" is not.
COPYRIGHT_PATTERN = re.compile(
    r"(?:"
    r"©\s*(?:\d{4}\s*(?:[-–]\s*\d{4})?\s*)?"
    r"|(?i:copyright|\(c\))\s+\d{4}\s*(?:[-–]\s*\d{4})?\s*"
    r")"
    # ’ is the typographic "smart quote" apostrophe (’) — real sites overwhelmingly
    # use it rather than a straight ASCII apostrophe for a possessive business name in
    # body copy ("Zabar’s", "McDonald’s"). Without it here, the name is silently
    # truncated right before the apostrophe ("Zabar's" -> "Zabar") since the character
    # class simply stops matching — found live on a real site's product pages.
    r"(?:(?i:by)\s+)?([A-Z][\w&'’\-]*(?:\s+[A-Z][\w&'’\-]*){0,5})"
)

#: The standard rights-reservation boilerplate that follows a copyright year on
#: countless sites ("© 2026 All Rights Reserved", "All Rights Reserved
#: Worldwide", "Some Rights Reserved" for Creative Commons). Every one of these
#: is capitalized exactly like a real business name and sits in exactly the
#: position COPYRIGHT_PATTERN captures, with no textual signal to tell them
#: apart other than that they aren't actually a name — a bare regex has no way
#: to know that, so this is an explicit denylist, checked word-by-word rather
#: than as a fixed phrase so it also catches variants (trailing "Worldwide",
#: "Globally", a leading "All"/"Some"/"No").
_RIGHTS_BOILERPLATE_WORDS: frozenset[str] = frozenset(
    {"all", "some", "no", "rights", "reserved", "worldwide", "globally", "international"}
)


def _is_rights_boilerplate(candidate: str) -> bool:
    """True when a copyright-line capture is rights-reservation text, not a name.

    Rejects only when *every* word in the capture is one of the boilerplate
    words above — "All Rights Reserved" is rejected, but a real name that merely
    contains one of these words as part of something larger would not be (none
    of the words are common in business names on their own, and requiring all
    of them to match keeps this from over-triggering).
    """
    words = candidate.lower().split()
    return bool(words) and all(word in _RIGHTS_BOILERPLATE_WORDS for word in words)


#: URL path segments that reliably identify a legal/policy document page across
#: real sites — Terms of Service, Privacy Policy, cookie/legal notices, and their
#: many naming variants. Deliberately a path-based check rather than a content
#: sniff: it is cheap, doesn't depend on the page actually saying "Terms of
#: Service" verbatim, and generalizes across sites since this URL convention is
#: extremely common (found live matching stripe.com/in/legal/... during a real
#: NAP consistency check).
_LEGAL_PAGE_PATH_MARKERS: frozenset[str] = frozenset(
    {"legal", "terms", "tos", "privacy", "cookie-policy", "cookies", "gdpr", "dpa",
     "acceptable-use", "eula"}
)

#: Matches one path segment that either *is* a legal-document marker exactly, or
#: begins with one followed by a hyphen (so "terms-of-service" and
#: "privacy-policy-2026" both match, but "legality-of-vpns" and "termstartup" do
#: not — the hyphen boundary is what a naive substring check got wrong).
_LEGAL_SEGMENT_RE = re.compile(
    r"^(?:" + "|".join(re.escape(m) for m in _LEGAL_PAGE_PATH_MARKERS) + r")(?:-.*)?$"
)


def _is_legal_document_url(url: str) -> bool:
    """True when a URL's path names it as a legal/policy document page.

    Args:
        url: The page's URL (its final, post-redirect form).

    Returns:
        Whether any path segment matches a known legal-document convention. Segment
        boundaries matter: ``/legality-of-vpns`` is an ordinary page about a topic
        that happens to start with "legal", not a legal document, and must not
        match — only a segment that *is* one of the known markers, optionally with
        a hyphenated suffix (``terms-of-service``), counts.
    """
    segments = urlsplit(url).path.lower().strip("/").split("/")
    return any(_LEGAL_SEGMENT_RE.match(segment) for segment in segments if segment)


#: Cap per field per page, so one pathological page cannot flood the comparison.
MAX_CANDIDATES_PER_FIELD = 12


def identify_target_business(pages: list[Any]) -> str | None:
    """Identify which business a site's NAP data should belong to, before extracting it.

    This is the missing step that let unrelated schema — a payment processor's
    ``Organization`` block, a sister brand mentioned in a footer, or (as found live
    during the Phase 10 cross-site sweep, after fixing a separate false-positive
    regex bug) any other business-typed block anywhere on the site — get pooled
    into the same NAP comparison as the site's own business, producing a verdict
    about an entity nobody actually asked about.

    Reads only schema (JSON-LD and microdata) business names — never visible-text
    guesses — across every page, and picks one by:

    1. **Homepage priority.** If the site's actual root page (``https://x.example/``
       — path is exactly empty or ``/``, nothing shorter or nothing else) declares
       a business name via schema, that name wins outright. A site's own homepage
       is the single most authoritative statement of who it is; nothing else on
       the site should get to overrule it. This is deliberately a literal check
       against the root path, not "whichever crawled page happens to have the
       shortest path" — if the crawl seed was ``/shop`` and root was never
       fetched, ``/shop`` is not the homepage and must not be treated as one; a
       genuinely shorter-but-still-non-root page (say ``/about`` shorter than
       ``/about/team``) has no more claim to being "the front door" than any
       other non-root page does.
    2. **Otherwise, majority vote.** The most common normalized name across every
       page's schema, ties broken by first-seen order for determinism.

    Args:
        pages: ``PageData`` objects from Phase 3.

    Returns:
        The normalized target business name, or ``None`` when no page declares one
        via schema at all — in which case entity filtering is simply not applied
        (see :func:`extract_nap_candidates`), which is exactly today's behaviour
        for a site with no identifiable schema entity.
    """
    homepage_names: list[str] = []
    all_names: list[str] = []

    for page in pages:
        try:
            url = getattr(page, "final_url", "") or getattr(page, "url", "")
            structured = list(getattr(page, "structured_data", []) or [])
            html = getattr(page, "html", "") or ""

            names_on_page = _business_names_in_schema(structured)
            names_on_page.extend(_business_names_in_schema(extract_microdata(parse_html(html))))
        except Exception:
            # One page that can't be read must not cost identification of the
            # target business for every other page — matches the same tolerance
            # extract_nap_candidates' own caller (run_nap_check) already has.
            continue

        if not names_on_page:
            continue

        all_names.extend(names_on_page)

        if _is_root_path(url):
            homepage_names.extend(names_on_page)

    if homepage_names:
        # A genuine root-page statement of identity wins outright, never merely
        # contributing a vote alongside every other page.
        return Counter(homepage_names).most_common(1)[0][0]

    if not all_names:
        return None

    return _most_common_stable(all_names)


def _is_root_path(url: str) -> bool:
    """True when ``url``'s path is the site's actual root — exactly empty or ``/``.

    Deliberately not "the shortest path seen among crawled pages": a crawl that
    never fetched root at all has no homepage among its pages, and the shortest
    available non-root page (``/shop``, ``/about``) is not entitled to stand in
    for one.
    """
    path = urlsplit(url).path
    return path in ("", "/")


def _business_names_in_schema(blocks: Any) -> list[str]:
    """Return every normalized business name found in a list of schema blocks."""
    names: list[str] = []

    def walk(block: Any) -> None:
        if isinstance(block, list):
            for entry in block:
                walk(entry)
            return
        if not isinstance(block, dict):
            return
        if _is_business_type(block.get("@type")):
            name = block.get("name") or block.get("legalName")
            if isinstance(name, str):
                normalized = normalize_name(name)
                if normalized:
                    names.append(normalized)
        for value in block.values():
            if isinstance(value, (dict, list)):
                walk(value)

    walk(blocks)
    return names


def _most_common_stable(values: list[str]) -> str:
    """Mode of ``values``, ties broken by first-seen order rather than left to chance."""
    counts = Counter(values)
    best_count = max(counts.values())
    for value in values:
        if counts[value] == best_count:
            return value
    return values[0]  # unreachable, but keeps this total


def extract_nap_candidates(
    page_data: Any, target_business: str | None = None
) -> dict[str, list[NAPValue]]:
    """Extract candidate Name, Address, and Phone (NAP) values from parsed page data.

    Args:
        page_data: A ``PageData`` from Phase 3.
        target_business: The site's identified business name, already normalized
            (see :func:`identify_target_business`). When given, schema blocks
            (JSON-LD or microdata) whose own name identifies a *different*
            business are excluded — see :func:`_collect_from_schema` for exactly
            what that does and doesn't catch. ``None`` (the default) disables
            entity filtering, matching this function's original behaviour.

    Returns:
        ``{"name": [...], "address": [...], "phone": [...]}``. Each ``NAPValue``
        carries the verbatim ``raw_value``, its normalized comparison form, and the
        ``source`` it came from. Candidates whose normalized form is empty (an
        implausible phone number, say) are dropped. Empty lists are normal — that is
        ``insufficient_data``, not an error.
    """
    page_url = getattr(page_data, "final_url", "") or getattr(page_data, "url", "")
    html = getattr(page_data, "html", "") or ""
    text = getattr(page_data, "text", "") or ""
    structured = list(getattr(page_data, "structured_data", []) or [])

    candidates: dict[str, list[NAPValue]] = {"name": [], "address": [], "phone": []}
    seen: set[tuple[str, str, str]] = set()

    def add(field: str, raw: str, source: str) -> None:
        raw_value = normalize_whitespace(raw)
        if not raw_value:
            return

        normalized = _NORMALIZERS[field](raw_value)
        if not normalized:
            return

        key = (field, normalized, source)
        if key in seen or len(candidates[field]) >= MAX_CANDIDATES_PER_FIELD:
            return
        seen.add(key)

        candidates[field].append(
            NAPValue(
                page=page_url,
                raw_value=raw_value,
                normalized_value=normalized,
                source=source,
            )
        )

    # 1. JSON-LD — most authoritative when present.
    _collect_from_schema(structured, add, "json_ld", target_business)

    soup = parse_html(html)

    # 2. Microdata — the pre-JSON-LD equivalent.
    _collect_from_schema(extract_microdata(soup), add, "microdata", target_business)

    # 3. tel: links — an explicit, machine-readable phone number.
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if href.lower().startswith("tel:"):
            # Prefer the visible label; fall back to the href for icon-only links.
            add("phone", element_text(anchor) or href[4:], "tel_link")

    # 4. Visible text — loosest source, tightest gates. Skipped entirely on a
    # legal/policy document page (see _is_legal_document_url): dense numbered
    # legal prose ("11 Severability. If any court...", "Section 25.10.3-101")
    # is adversarial to every one of these shape heuristics at once, and real
    # NAP data is essentially never meaningfully stated in Terms of Service
    # boilerplate anyway — a business's actual phone/address lives on its
    # Contact page or footer, not buried in a numbered clause. json_ld,
    # microdata and tel_link above are unaffected: those are deliberate
    # structured statements even when they happen to sit on a legal page.
    if not _is_legal_document_url(page_url):
        for match in PHONE_PATTERN.findall(text):
            if _looks_phone_shaped(match):
                add("phone", match, "visible_text")

        for match in ADDRESS_PATTERN.findall(text):
            if _looks_address_shaped(match):
                add("address", match, "visible_text")

        for match in COPYRIGHT_PATTERN.findall(text):
            if not _is_rights_boilerplate(match):
                add("name", match, "visible_text")

    return {field: _prefer_precise_sources(found) for field, found in candidates.items()}


#: How much of a page's text is shown to the LLM NAP fallback. NAP information
#: typically sits in a header/logo (near the start) or a footer (near the end),
#: so a long page is bounded by keeping both ends rather than just truncating
#: from the front, which would silently drop a footer entirely.
_LLM_NAP_MAX_CHARS = 6000


def extract_nap_candidates_via_llm(
    page_data: Any, fields: Iterable[str], generate: Callable[..., str]
) -> dict[str, list[NAPValue]]:
    """Ask an LLM to find NAP values none of the tiers above could recognize.

    A last-resort supplementary source, never a replacement for the tiers above:
    :func:`app.agents.nap_agent.run_nap_check` calls this only for a field that
    already has zero candidates from every other source across the whole site,
    and only against a small, targeted subset of pages it selects (home/contact/
    about-like) — never against every page in a large crawl, which both bounds
    the cost and matches how a human would actually go looking for this
    information.

    The same governing rule as every other source in this module still applies,
    enforced mechanically rather than by trusting the prompt: a claimed value is
    only accepted if it is a literal, whitespace-normalized substring of this
    page's own text — the identical check
    :func:`app.validation.qa_validator.find_source_span` uses for Q3. The model
    may point at real text on the page; it may never compose or paraphrase one.

    Args:
        page_data: A ``PageData`` from Phase 3.
        fields: Which of "name"/"address"/"phone" to ask about — normally only
            the fields the rest of this module found nothing for, site-wide.
        generate: A ``generate(prompt, **kwargs) -> str`` callable.

    Returns:
        ``{field: [NAPValue]}`` for whichever requested fields the model found
        AND that passed the literal-substring check — often empty. Any parse
        failure, exception, or invented (non-literal) value degrades silently to
        no candidate for that field, never a guess.
    """
    requested = [f for f in fields if f in _NORMALIZERS]
    if not requested:
        return {}

    page_url = getattr(page_data, "final_url", "") or getattr(page_data, "url", "")
    text = getattr(page_data, "text", "") or ""

    if not text.strip() or _is_legal_document_url(page_url):
        return {}

    try:
        raw = generate(
            _llm_nap_prompt(text, requested),
            max_tokens=400,
            reasoning_effort="low",
            temperature=0,
        )
    except Exception:
        return {}

    claimed = _parse_llm_nap_response(raw, requested)
    if not claimed:
        return {}

    found: dict[str, list[NAPValue]] = {}
    for field, value in claimed.items():
        raw_value = normalize_whitespace(value)
        if not raw_value or raw_value not in text:
            # Not literally on the page -- the model composed or paraphrased
            # instead of pointing at real text. Discarded, never "fixed".
            continue

        normalized = _NORMALIZERS[field](raw_value)
        if not normalized:
            continue

        found[field] = [
            NAPValue(
                page=page_url,
                raw_value=raw_value,
                normalized_value=normalized,
                source="llm",
            )
        ]

    return found


def _bound_text_for_llm(text: str, max_chars: int = _LLM_NAP_MAX_CHARS) -> str:
    """Keep both ends of a long page's text, since NAP data lives at either end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _llm_nap_prompt(text: str, fields: list[str]) -> str:
    """Build the prompt asking the model to quote NAP values verbatim, or say none."""
    field_list = ", ".join(fields)
    example = json.dumps({field: "<exact text from the page, or null>" for field in fields})
    return (
        "You are given the visible text of one web page. Find the business's own "
        f"{field_list} on this page, if stated.\n\n"
        "Rules:\n"
        "- Copy each value EXACTLY as it appears below -- the same characters, "
        "spacing, and punctuation. Do not paraphrase, reformat, translate, "
        "abbreviate, or invent anything.\n"
        "- Only report a field the page actually states. Use null for a field "
        "that is not present -- guessing is worse than leaving it blank.\n"
        "- An address does not need a street name or number to count: a city "
        "and state/region is a real, reportable address if that is all the page "
        "states.\n\n"
        f"Page text:\n{_bound_text_for_llm(text)}\n\n"
        f"Reply with a single JSON object using exactly these keys: {example}"
    )


def _parse_llm_nap_response(raw: Any, fields: list[str]) -> dict[str, str]:
    """Parse the model's JSON reply into ``{field: claimed_text}``, dropping nulls/junk."""
    if not raw:
        return {}
    try:
        payload = json.loads(str(raw).strip())
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    result: dict[str, str] = {}
    for field in fields:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value
    return result


def _prefer_precise_sources(values: list[NAPValue]) -> list[NAPValue]:
    """Drop ``visible_text`` candidates when a precise source already supplied one.

    Without this, one page states its address twice — once in full in JSON-LD, once
    as the fragment the visible-text regex can see — and the two normalize
    differently, producing an ``inconsistent`` verdict for a site that is perfectly
    consistent. That is the single most likely false positive in this phase, and a
    false "inconsistent" is worse than a missed one: it is a confident claim the
    evidence does not support.

    Only the regex tier is suppressed. JSON-LD, microdata and ``tel:`` links are all
    deliberate machine-readable statements of a complete value, so they stay mixed
    and a genuine disagreement between them is still caught.

    The fallback is preserved: on a page with no structured markup at all, the
    visible-text candidates are the only ones there are and every one is kept.
    """
    precise = [value for value in values if value.source != "visible_text"]
    return precise if precise else values


#: A leading number that could plausibly be a calendar year rather than a house
#: number. House numbers in this exact 4-digit range exist but are far rarer than
#: a copyright year or a "last updated" year sitting near ordinary prose that
#: happens to end in a street-type word.
_LEADING_NUMBER = re.compile(r"^\d+")
_PLAUSIBLE_YEAR_RANGE = range(1900, 2100)


def _looks_address_shaped(candidate: str) -> bool:
    """Reject an address match whose leading number is more likely a year than a house number.

    ``ADDRESS_PATTERN`` requires a street-type word (``street``, ``way``,
    ``close``, ...), but several of those are also ordinary English words, and a
    stray 4-digit copyright year sitting a few words before one of them is enough
    to produce a match that is not an address at all. Found live on a real
    Shopify store's footer: "2026 Terms of service Close" (the page's copyright
    year, followed unrelatedly by a "Close" button label) and "32 Better things in
    a better way" (a marketing tagline ending in "way") both matched
    ``ADDRESS_PATTERN`` and polluted the Q2 address comparison with garbage.

    This mirrors the same precedent already applied to phone numbers
    (:func:`_looks_phone_shaped`'s date-shape exclusion): a bare number that looks
    like a year is treated as unlikely to be the start of a real address.
    """
    match = _LEADING_NUMBER.match(candidate.strip())
    if not match:
        return True  # ADDRESS_PATTERN always starts with digits; stay permissive if not.

    number = match.group()
    if len(number) == 4 and int(number) in _PLAUSIBLE_YEAR_RANGE:
        return False

    return True


def _looks_phone_shaped(candidate: str) -> bool:
    """True when a digit-punctuation run is shaped like a formatted phone number.

    A real phone number is chunked into groups of 2-4 digits ("503-555-0147"), and
    a leading punctuation mark (parens, a "+") is what marks a country/area code.
    This rejects two shapes the digit-count gate alone lets through: a year range
    ("2001-2026", two 4-digit groups) and a numeric sequence embedded in a code
    sample or table ("1 1 2 3 5 8", all single-digit groups). It is intentionally
    permissive elsewhere — the goal is filtering out obvious non-phone noise, not
    validating a specific numbering plan.

    One shape is deliberately exempted from the "at least 2 groups" requirement: a
    solid, unbroken digit run prefixed with "+" ("+919876543210", with zero
    internal spacing). This is an extremely common real-world format — WhatsApp
    contact numbers and mobile numbers across South Asia and elsewhere are
    routinely written exactly this way — and it collapses to a single digit group
    since there is nothing to split it on. Found live: a real site's WhatsApp
    number, repeated identically and correctly on 18 of 20 pages, was silently
    rejected everywhere except the one page that happened to add a space after the
    country code, turning a clean "consistent" result into a false
    "insufficient_data". The leading "+" is itself the phone signal here — nobody
    prefixes an arbitrary numeric sequence with it — so it is trusted the same way
    parens already are elsewhere in this function.
    """
    if not _PHONE_PUNCTUATION.search(candidate):
        return False
    if _DATE_SHAPE.match(candidate.strip()):
        return False

    groups = _DIGIT_GROUP.findall(candidate)
    if len(groups) < 2 and not candidate.strip().startswith("+"):
        return False

    # Two or more 4-digit groups is a year range or a copyright/date run bleeding
    # together ("2001-2026", or a mangled "2026 2026-09-17" from concatenated
    # copyright-year and last-updated-date text) rather than a phone number. Real
    # formatted numbers have at most one 4-digit block (the subscriber number);
    # everything else in a typical grouping is 1-3 digits. Found live on
    # python.org: this exact shape survived every earlier check and produced a
    # false "phone: consistent" verdict on a site with no phone number at all —
    # not merely noisy evidence, a wrong top-level claim. Parens or a leading "+"
    # are still trusted as a deliberate area-code marker and exempt this.
    four_digit_groups = sum(1 for g in groups if len(g) == 4)
    if four_digit_groups >= 2:
        if "(" not in candidate and not candidate.strip().startswith("+"):
            return False

    # Real numbers are chunked in groups of 2-4 digits. More than one lone-digit
    # group (as opposed to a single leading country-code digit) means this is a
    # digit list, not a phone number.
    single_digit_groups = sum(1 for g in groups if len(g) == 1)
    if single_digit_groups > 1:
        return False

    return True


def _collect_from_schema(
    block: Any, add, source: str, target_business: str | None = None
) -> None:
    """Pull name/address/phone out of one JSON-LD or microdata object.

    Recurses into nested objects and lists, since a ``@graph`` entry or an
    ``Organization`` nested inside a ``WebSite`` is often where the real business
    data sits. Only objects whose ``@type`` names a business are read, so a
    ``Product`` block quoting a manufacturer's phone number is skipped.

    Args:
        block: A parsed JSON-LD object/array, or a microdata item dict.
        add: The candidate-recording closure from :func:`extract_nap_candidates`.
        source: ``"json_ld"`` or ``"microdata"``.
        target_business: The site's identified business name (already
            normalized), from :func:`identify_target_business`. When given, a
            business-typed block whose own ``name``/``legalName`` is present and
            does not match this — after stripping legal suffixes from both sides,
            so "Ridgeline Coffee Roasters" and "Ridgeline Coffee Roasters LLC"
            still count as the same business — is skipped entirely: its
            phone/address describe a different entity (a shipping partner, a
            payment processor, a sister brand referenced in passing) and do not
            belong in this site's own NAP comparison.

            The suffix-tolerant comparison is deliberate, not an approximation for
            convenience: an exact-string match would silently defeat Phase 5's own
            legal-suffix-disagreement detection (:func:`app.extraction.normalize.differs_only_by_legal_suffix`)
            by filtering the differently-suffixed variant out as "a different
            business" before comparison ever saw it, rather than keeping it and
            flagging the disagreement.

            A block with no name field at all cannot be checked and is kept,
            matching this function's existing conservative default of accepting
            what it cannot disprove. ``None`` (the default) disables this filter
            entirely — every business-typed block is accepted, exactly as before
            this argument existed, so every existing call site is unaffected
            unless it opts in.
    """
    if isinstance(block, list):
        for entry in block:
            _collect_from_schema(entry, add, source, target_business)
        return

    if not isinstance(block, dict):
        return

    if _is_business_type(block.get("@type")):
        name = block.get("name") or block.get("legalName")
        block_name = name if isinstance(name, str) else None

        block_matches_target = (
            not target_business
            or not block_name
            or strip_legal_suffix(normalize_name(block_name)) == strip_legal_suffix(target_business)
        )

        if not block_matches_target:
            # A different, identifiable entity — not the site's own business.
            # Still recurse below in case a correctly-named block is nested
            # inside this one (e.g. the target Organization listed as a
            # "publisher" of an unrelated Article).
            pass
        else:
            if block_name:
                add("name", block_name, source)

            for key in ("telephone", "phone"):
                value = block.get(key)
                if isinstance(value, str):
                    add("phone", value, source)
                elif isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, str):
                            add("phone", entry, source)

            address = _format_address(block.get("address"))
            if address:
                add("address", address, source)

    for value in block.values():
        if isinstance(value, (dict, list)):
            _collect_from_schema(value, add, source, target_business)


def _is_business_type(raw_type: Any) -> bool:
    """True when a schema ``@type`` names the site's own business."""
    if isinstance(raw_type, str):
        return raw_type.strip().lower().rstrip("/").rsplit("/", 1)[-1] in BUSINESS_TYPES
    if isinstance(raw_type, list):
        return any(_is_business_type(entry) for entry in raw_type)
    return False


def _format_address(address: Any) -> str:
    """Render a schema address as one comma-joined string.

    Parts are emitted in :data:`ADDRESS_PARTS` order rather than dict order, so two
    pages whose JSON-LD lists the same fields differently still produce the same
    string — a difference of serialization, not of location.
    """
    if isinstance(address, str):
        return normalize_whitespace(address)

    if isinstance(address, list):
        for entry in address:
            rendered = _format_address(entry)
            if rendered:
                return rendered
        return ""

    if not isinstance(address, dict):
        return ""

    parts: list[str] = []
    for key in ADDRESS_PARTS:
        value = address.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("addressCountry")
        if isinstance(value, str) and value.strip():
            parts.append(normalize_whitespace(value))

    return ", ".join(parts)


#: Field name to its normalization rule. A candidate whose normalized form is empty
#: is not usable and is dropped at extraction time.
_NORMALIZERS = {
    "name": normalize_name,
    "address": normalize_address,
    "phone": normalize_phone,
}
