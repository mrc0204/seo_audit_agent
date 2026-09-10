"""Validator for evidence-grounded QA outputs and citations.

The critical gate named in the plan (Phase 6, step 4): a claimed excerpt is
accepted only if it is a **literal substring** of the cited page's own text, after
whitespace normalization and nothing else. No fuzzy matching, no trimming a
near-miss into shape. A near-miss is a failure, full stop — the agent must return
null rather than "fix" it.

:func:`app.agents.qa_agent.answer_question` calls this before it will accept an
LLM's claimed excerpt. Phase 7's cross-cutting validation pass re-runs the same
function against the finished ``outputs/answer.json``, so both call sites share
this one gate rather than two drifting implementations.
"""

from __future__ import annotations

from typing import Any

from app.extraction.html_parser import normalize_whitespace

#: A well-formed "no answer" state: neither field claims anything.
_NULL_MATCH_TYPES = {"none", None}


def validate_qa(output: Any, pages: list[Any]) -> bool:
    """Validate QA agent answers for ground truth support and citation validity.

    Args:
        output: A ``QAAnswer``, or any object/dict exposing ``url`` and
            ``excerpt`` (and optionally ``match_type``).
        pages: ``PageData`` objects the excerpt is checked against.

    Returns:
        ``True`` when the output is grounded: either it correctly declines
        (``url`` and ``excerpt`` both absent) or its ``excerpt`` is a literal,
        whitespace-normalized substring of the cited page's own text. ``False``
        for every other case, including a well-formed excerpt whose page cannot be
        found, or a partial claim (one of ``url``/``excerpt`` set without the
        other).
    """
    url = _get(output, "url")
    excerpt = _get(output, "excerpt")

    if url is None and excerpt is None:
        # Correctly declining is a valid, grounded outcome — the point of this
        # whole phase is that null beats a guess.
        return True

    if url is None or excerpt is None:
        # A citation with only half the claim present cannot be checked and is
        # not a state the pipeline should ever produce.
        return False

    if not str(excerpt).strip():
        return False

    return find_source_span(url, excerpt, pages) is not None


def find_source_span(url: str, excerpt: str, pages: list[Any]) -> tuple[int, int] | None:
    """Locate a claimed excerpt in its cited page's text, if it is really there.

    Args:
        url: The page the excerpt is claimed to come from.
        excerpt: The claimed text.
        pages: Candidate ``PageData`` objects.

    Returns:
        The ``(char_start, char_end)`` span of the excerpt within that page's
        ``PageData.text``, or ``None`` if the page cannot be found or the excerpt,
        after whitespace normalization, is not a literal substring of it. Only
        whitespace is normalized — no case-folding, no punctuation stripping, no
        fuzzy matching.
    """
    page = _find_page(url, pages)
    if page is None:
        return None

    page_text = getattr(page, "text", "") or ""
    normalized_excerpt = normalize_whitespace(excerpt)
    if not normalized_excerpt:
        return None

    start = page_text.find(normalized_excerpt)
    if start == -1:
        return None

    return start, start + len(normalized_excerpt)


def is_well_formed_null(output: Any) -> bool:
    """True when an output is a correctly-declined answer, not merely absent fields.

    Args:
        output: A ``QAAnswer`` or equivalent.

    Returns:
        Whether ``url`` and ``excerpt`` are both null and ``match_type`` (when
        present) says so too.
    """
    return (
        _get(output, "url") is None
        and _get(output, "excerpt") is None
        and _get(output, "match_type", default="none") in _NULL_MATCH_TYPES
    )


def filter_valid_answer(answer: Any, pages: list[Any]) -> Any | None:
    """Return ``answer`` if it passes :func:`validate_qa`, else the correct null answer.

    The gate Phase 8's CLI should call before writing ``answer.json``, mirroring
    :func:`app.validation.seo_validator.filter_valid_findings` and
    :func:`app.validation.nap_validator.filter_valid_comparisons`. An answer's
    single-item nature (one question, one answer) means "filter" here means
    "replace with null" rather than "drop from a list" — the query itself must
    still be reported as asked, just answered honestly as unsupported.

    Args:
        answer: A ``QAAnswer`` or equivalent.
        pages: The crawl snapshot to validate against.

    Returns:
        ``answer`` unchanged if valid; otherwise a well-formed null answer
        constructed from its own ``query`` field, or ``None`` when even that could
        not be determined.
    """
    if validate_qa(answer, pages):
        return answer

    query = _get(answer, "query")
    if query is None:
        return None

    try:
        from app.models.answer import QAAnswer

        return QAAnswer(query=query, url=None, excerpt=None, match_type="none")
    except Exception:
        return {"query": query, "url": None, "excerpt": None, "match_type": "none"}


def _find_page(url: str, pages: list[Any]) -> Any | None:
    """Find the page whose ``final_url`` (or ``url``) matches exactly.

    Exact match only — no normalization, no trailing-slash folding. The excerpt
    must have come from the URL actually cited, not a URL that merely resembles it.
    """
    for page in pages:
        if getattr(page, "final_url", None) == url or getattr(page, "url", None) == url:
            return page
    return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or dict key, tolerating either shape."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
