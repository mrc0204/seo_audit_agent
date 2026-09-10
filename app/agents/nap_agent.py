"""NAP (Name, Address, Phone) consistency checking agent.

Takes the candidates :mod:`app.extraction.nap_extractor` found on each page and
produces one :class:`~app.models.nap.NAPComparison` per field.

The verdict rule is the plan's, unchanged:

* fewer than two **pages** contributed a value → ``insufficient_data``
* every value normalizes to one form → ``consistent``
* two or more distinct normalized forms → ``inconsistent``

Note that the page count, not the value count, gates ``insufficient_data``. Three
values from one page say nothing about consistency *across* the site — they are one
page agreeing with itself, and calling that "consistent" would be a verdict the
evidence does not support.

``confidence`` is deterministic and stated plainly: the share of collected values
that agree with the majority normalized form. It is never model-estimated, and it
measures *agreement among what was found* — not the probability that the values are
correct, which no amount of on-page evidence can establish.

Where an LLM could help, per the plan, is filtering candidates: deciding whether a
phone-shaped string is the business's number or one quoted in a case study.
:func:`filter_candidates_with_llm` provides that hook with the same structural
guarantee as Phase 4 — a model may only *reject* candidates, never introduce or edit
one, so it cannot put a value into evidence that is not literally on the page.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from app.extraction.nap_extractor import extract_nap_candidates, identify_target_business
from app.extraction.normalize import address_subsumes, differs_only_by_legal_suffix
from app.models.nap import NAPComparison, NAPValue

#: The fields compared, and the order they appear in ``nap_report.json``.
NAP_FIELDS: tuple[str, ...] = ("name", "address", "phone")

#: Minimum number of distinct pages that must contribute a value before a verdict
#: other than ``insufficient_data`` is possible.
MIN_PAGES_FOR_VERDICT = 2

#: Ranking used to pick which raw spelling represents a normalized value. Structured
#: markup is a deliberate statement of the value; visible text is an inference from
#: layout.
SOURCE_PRIORITY: dict[str, int] = {
    "json_ld": 0,
    "microdata": 1,
    "tel_link": 2,
    "visible_text": 3,
}


def run_nap_check(pages: list[Any]) -> list[Any]:
    """Audit and compare Name, Address, and Phone consistency across crawled web pages.

    Two passes, per the plan's own diagram (target business identification ->
    extraction -> normalization -> comparison): first
    :func:`~app.extraction.nap_extractor.identify_target_business` reads every
    page's schema to decide which business this audit is actually about, then
    extraction runs a second time with that identity, so a payment processor's or
    a sister brand's schema block never gets pooled into this site's own NAP
    comparison. See :func:`identify_target_business` for exactly how the target is
    chosen and what "no identifiable target" degrades to.

    Args:
        pages: ``PageData`` objects from Phase 3.

    Returns:
        One :class:`NAPComparison` per field in :data:`NAP_FIELDS`, always three,
        even when a field had no candidates at all — a field that is absent
        everywhere is itself a reportable result (``insufficient_data``), not an
        omission from the report.
    """
    target_business = identify_target_business(pages)
    collected: dict[str, list[NAPValue]] = defaultdict(list)

    for page in pages:
        try:
            found = extract_nap_candidates(page, target_business=target_business)
        except Exception:
            # One unparseable page must not cost the whole site's NAP verdict.
            continue
        for field in NAP_FIELDS:
            collected[field].extend(found.get(field, []))

    return [compare_field(field, collected.get(field, [])) for field in NAP_FIELDS]


def compare_field(field: str, values: Iterable[NAPValue]) -> NAPComparison:
    """Build the consistency verdict for one NAP field.

    Args:
        field: ``"name"``, ``"address"`` or ``"phone"``.
        values: Every candidate found for that field, across all pages.

    Returns:
        A populated :class:`NAPComparison`. ``values`` and ``normalized_values`` are
        the distinct forms found, in first-seen order, so the report shows what
        differed rather than repeating one value once per page.
    """
    values = list(values)
    pages_compared = _unique([value.page for value in values])

    if not values or len(pages_compared) < MIN_PAGES_FOR_VERDICT:
        return NAPComparison(
            field=field,
            pages_compared=pages_compared,
            values=_unique([value.raw_value for value in values]),
            normalized_values=_unique([value.normalized_value for value in values]),
            confidence=0.0,
            verdict="insufficient_data",
            evidence=values,
        )

    canonical_of = _canonical_forms(field, [value.normalized_value for value in values])

    counts = Counter(canonical_of[value.normalized_value] for value in values)
    distinct = list(counts)

    _, majority_count = counts.most_common(1)[0]
    confidence = round(majority_count / len(values), 4)

    verdict = "consistent" if len(distinct) == 1 else "inconsistent"

    return NAPComparison(
        field=field,
        pages_compared=pages_compared,
        # One representative raw spelling per distinct value, so the report reads as
        # "these forms were found" rather than repeating one value once per page.
        values=[
            _representative_raw(values, canonical_of, canonical) for canonical in distinct
        ],
        normalized_values=distinct,
        confidence=confidence,
        verdict=verdict,
        evidence=values,
    )


def _canonical_forms(field: str, normalized_values: Iterable[str]) -> dict[str, str]:
    """Map each normalized value to the form it is counted as.

    For every field but ``address`` this is the identity: two values agree only when
    they are byte-identical, which is what makes ``Acme Inc`` and ``Acme LLC`` stay
    distinct.

    Addresses get one narrow extra rule. A value that is a truncation of a fuller
    one — the street line alone versus street plus locality and postcode — is
    counted as that fuller form, because the two are the same place written at
    different levels of detail. :func:`app.extraction.normalize.address_subsumes`
    defines "truncation" strictly enough that a differing suite number, house
    number, street or town never qualifies.

    Args:
        field: The NAP field being compared.
        normalized_values: Every normalized value collected for it.

    Returns:
        ``{normalized_value: canonical_value}`` covering every input.
    """
    distinct = _unique(normalized_values)

    if field != "address":
        return {value: value for value in distinct}

    # Longest first, so shorter truncations attach to the fullest form available.
    ordered = sorted(distinct, key=lambda value: len(value.split()), reverse=True)

    canonical_of: dict[str, str] = {}
    canonicals: list[str] = []

    for value in ordered:
        match = next(
            (existing for existing in canonicals if address_subsumes(existing, value)),
            None,
        )
        if match is None:
            canonicals.append(value)
            canonical_of[value] = value
        else:
            canonical_of[value] = match

    return canonical_of


def describe_disagreement(comparison: NAPComparison) -> str:
    """Explain an ``inconsistent`` verdict in one sentence.

    Distinguishes the cases a bare "inconsistent" flattens together — most usefully
    a legal-suffix difference (``Acme Inc`` vs ``Acme LLC``), which normalization
    deliberately refuses to merge because it may be a real signal rather than a typo.

    Args:
        comparison: A completed comparison.

    Returns:
        A human-readable explanation, or ``""`` when there is nothing to explain.
    """
    if comparison.verdict == "insufficient_data":
        found = len(comparison.evidence)
        pages = len(comparison.pages_compared)
        return (
            f"Only {found} {comparison.field} value(s) across {pages} page(s); at least "
            f"{MIN_PAGES_FOR_VERDICT} pages must carry a value before consistency can "
            "be judged."
        )

    if comparison.verdict == "consistent":
        return (
            f"All {len(comparison.evidence)} {comparison.field} value(s) normalize to "
            f"{comparison.normalized_values[0]!r}."
        )

    normalized = comparison.normalized_values
    if (
        comparison.field == "name"
        and len(normalized) == 2
        and differs_only_by_legal_suffix(normalized[0], normalized[1])
    ):
        return (
            f"The names {normalized[0]!r} and {normalized[1]!r} differ only by legal "
            "suffix. These are distinct legal entities, so they are reported rather "
            "than merged — confirm which is correct."
        )

    return (
        f"{len(normalized)} distinct {comparison.field} values were found across "
        f"{len(comparison.pages_compared)} page(s): "
        + ", ".join(repr(value) for value in normalized[:4])
    )


def filter_candidates_with_llm(
    values: list[NAPValue],
    decide: Callable[[str], str] | None,
    context: str = "",
) -> list[NAPValue]:
    """Let an LLM *reject* candidates that are not the business's own details.

    The optional filter the plan allows: a phone-shaped string on a page may be a
    number quoted in a case study rather than the business's own. A model can judge
    that from context; a regex cannot.

    The guarantee mirrors Phase 4's. The model is only ever asked a yes/no question
    about a candidate that already exists, and its answer can only *remove* items
    from the list. It has no channel to add a value or edit one, so it cannot put a
    number into evidence that is not literally on the page. Anything other than a
    clear rejection keeps the candidate — the default favours reporting a real value
    over silently dropping one.

    Args:
        values: Candidates for one field.
        decide: Callable taking a prompt and returning text. ``None`` returns the
            list unchanged, which is the offline default.
        context: Optional extra detail for the prompt, such as the business name.

    Returns:
        A new list containing a subset of ``values``, in the original order.
    """
    if decide is None or not values:
        return list(values)

    kept: list[NAPValue] = []
    for value in values:
        try:
            answer = decide(
                CANDIDATE_FILTER_PROMPT.format(
                    raw_value=value.raw_value,
                    source=value.source,
                    page=value.page,
                    context=context or "(none)",
                )
            )
            rejected = str(answer or "").strip().lower().startswith("no")
        except Exception:
            # A failing filter must not silently empty the evidence list.
            rejected = False

        if not rejected:
            kept.append(value)

    return kept


#: Prompt for the optional candidate filter. Phrased so the model answers about one
#: existing candidate and cannot supply a value of its own.
CANDIDATE_FILTER_PROMPT = (
    "A value was extracted from a business website and may or may not be the "
    "business's own contact detail.\n\n"
    "Value: {raw_value}\n"
    "Found in: {source}\n"
    "Page: {page}\n"
    "Business context: {context}\n\n"
    'Answer "yes" if this is plausibly the business\'s own detail, or "no" if it '
    "clearly belongs to someone else (a customer quoted in a case study, a partner, "
    "an emergency service). Answer with one word only."
)


def _representative_raw(
    values: list[NAPValue], canonical_of: dict[str, str], canonical: str
) -> str:
    """Pick the raw spelling that best represents a canonical value.

    Prefers the most authoritative source, so evidence quotes the site's structured
    markup rather than a stray copy in a footer when both exist.
    """
    matching = [
        value for value in values if canonical_of.get(value.normalized_value) == canonical
    ]
    if not matching:
        return canonical
    return min(matching, key=lambda v: SOURCE_PRIORITY.get(v.source, 99)).raw_value


def _unique(items: Iterable[str]) -> list[str]:
    """De-duplicate while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
