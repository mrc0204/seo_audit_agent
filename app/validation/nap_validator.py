"""Validator for NAP audit outputs.

Three independent re-checks per comparison, each catching a different failure mode:

1. **Evidence re-derivation** — every ``NAPValue`` in ``evidence`` must reappear
   when :func:`app.extraction.nap_extractor.extract_nap_candidates` is run again on
   its cited page. This is the plan's explicit requirement ("re-confirm each
   raw_value literally appears in the cited page's text or structured data"), and
   re-running the extractor rather than doing a bare substring search also catches
   a value attributed to the wrong ``source`` (claimed as ``json_ld`` when the page
   only has it in visible text, say).
2. **Normalization drift** — each value's ``normalized_value`` is recomputed
   independently from its ``raw_value`` and must match what is stored. Catches a
   comparison built with one version of :mod:`app.extraction.normalize` being
   checked against markup that would normalize differently today.
3. **Verdict re-derivation** — the stored ``verdict``, ``confidence``,
   ``pages_compared``, ``values`` and ``normalized_values`` must equal what
   :func:`app.agents.nap_agent.compare_field` produces from the *same* evidence
   list. Catches a verdict that does not actually follow from its own evidence —
   the class of bug an evidence-level check alone cannot see.
"""

from __future__ import annotations

from typing import Any

from app.agents.nap_agent import compare_field
from app.extraction.nap_extractor import extract_nap_candidates, identify_target_business
from app.extraction.normalize import normalize_address, normalize_name, normalize_phone
from app.models.nap import NAPComparison

_NORMALIZERS = {
    "name": normalize_name,
    "address": normalize_address,
    "phone": normalize_phone,
}


def validate_nap(output: Any, pages: list[Any]) -> bool:
    """Validate NAP audit comparison outputs against page evidence.

    Args:
        output: A single ``NAPComparison``, or a list of them.
        pages: The ``PageData`` objects the evidence claims to come from — the
            same crawl snapshot the comparison was built from.

    Returns:
        ``True`` only if every comparison passes all three re-checks described in
        the module docstring. ``False`` the moment any evidence value fails to
        reappear, any normalization has drifted, or the stored verdict does not
        match what the evidence itself produces.
    """
    comparisons = _as_list(output)
    if comparisons is None:
        return False

    # Computed once and reused for every field: identify_target_business scans
    # schema across all `pages`, which is the same regardless of which field is
    # being checked, so recomputing it per-comparison would be pure waste.
    target_business = identify_target_business(pages)

    for comparison in comparisons:
        if not isinstance(comparison, NAPComparison):
            return False
        if not _validate_one(comparison, pages, target_business):
            return False

    return True


def _validate_one(
    comparison: NAPComparison, pages: list[Any], target_business: str | None = None
) -> bool:
    """Run all three re-checks against one comparison. Short-circuits on failure.

    Args:
        comparison: The comparison to re-derive and check.
        pages: The crawl snapshot it claims to come from.
        target_business: The site's identified business name — see
            :func:`app.extraction.nap_extractor.identify_target_business`.
            Re-deriving *with* this filter (rather than the unfiltered form) is
            what catches a comparison whose evidence smuggled in a foreign
            entity's NAP value: without it, re-extraction would still find that
            value present on the page and wrongly accept it.
    """
    field = comparison.field
    if field not in _NORMALIZERS:
        return False

    pages_by_url = _index_pages(pages)

    for value in comparison.evidence:
        page = pages_by_url.get(value.page)
        if page is None:
            return False

        # Check 1: re-derive the candidate from the page and require an exact
        # (raw_value, source) match — not merely "this text is somewhere on the
        # page", which would let a phone number attributed to the wrong source
        # (or the wrong business entity) through.
        recomputed = extract_nap_candidates(page, target_business=target_business).get(field, [])
        if not any(
            c.raw_value == value.raw_value and c.source == value.source for c in recomputed
        ):
            return False

        # Check 2: normalization must not have drifted from what the rule
        # currently produces for this raw value.
        if _NORMALIZERS[field](value.raw_value) != value.normalized_value:
            return False

    # Check 3: the verdict must actually follow from its own evidence.
    expected = compare_field(field, comparison.evidence)
    if (
        expected.verdict != comparison.verdict
        or expected.confidence != comparison.confidence
        or expected.pages_compared != comparison.pages_compared
        or expected.values != comparison.values
        or expected.normalized_values != comparison.normalized_values
    ):
        return False

    return True


def filter_valid_comparisons(
    comparisons: list[NAPComparison], pages: list[Any]
) -> list[NAPComparison]:
    """Return only the comparisons that pass :func:`validate_nap`, individually.

    The gate Phase 8's CLI should call before writing ``nap_report.json``.

    Args:
        comparisons: Candidate comparisons, typically one per NAP field.
        pages: The crawl snapshot to validate against.

    Returns:
        The subset of ``comparisons`` that pass every re-check, in original order.
    """
    target_business = identify_target_business(pages)
    return [
        c
        for c in comparisons
        if isinstance(c, NAPComparison) and _validate_one(c, pages, target_business)
    ]


def _as_list(output: Any) -> list[Any] | None:
    """Normalize a single comparison or a list into a list; ``None`` on a bad shape."""
    if output is None:
        return None
    if isinstance(output, NAPComparison):
        return [output]
    if isinstance(output, list):
        return output
    return None


def _index_pages(pages: list[Any]) -> dict[str, Any]:
    """Map both ``final_url`` and ``url`` to their page, so either lookup works."""
    index: dict[str, Any] = {}
    for page in pages:
        final_url = getattr(page, "final_url", None)
        url = getattr(page, "url", None)
        if final_url:
            index.setdefault(final_url, page)
        if url:
            index.setdefault(url, page)
    return index
