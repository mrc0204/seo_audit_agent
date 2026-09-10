"""Validator for verifying SEO audit finding outputs against evidence.

The plan's cross-cutting safety net: even though every check in
:mod:`app.agents.seo_agent` is already a pure function over evidence, a second,
independent pass catches bugs in the first — a check that used stale data, a
finding hand-edited after the fact, or (structurally impossible today, but not
provable from this file alone) an LLM step that reached past ``suggested_fix``.

The re-confirmation is literal, not approximate: **recompute the entire audit from
the cited pages and require an identical finding to reappear** — same ``metric``,
``page``, ``severity``, ``evidence`` and ``check_id``. ``suggested_fix`` is excluded
from the comparison on purpose; it is the one field Phase 4's LLM step is allowed to
rewrite, so a validator that required it to match byte-for-byte would reject
legitimate output.

A ``Finding`` with no ``check_id`` is rejected outright. There is no rule to
re-run without knowing which one produced it, and "cannot be traced" is exactly the
failure mode this validator exists to catch, not something to wave through.
"""

from __future__ import annotations

from typing import Any

from app.agents.seo_agent import run_seo_audit
from app.models.finding import Finding

#: Fields compared when confirming a finding still holds. ``suggested_fix`` is
#: deliberately absent — the LLM is allowed to have rewritten it.
_IMMUTABLE_FIELDS: tuple[str, ...] = ("metric", "page", "severity", "evidence", "check_id")


def validate_seo(output: Any, pages: list[Any]) -> bool:
    """Validate SEO audit agent findings against page evidence and report schema rules.

    Args:
        output: A single ``Finding``, or a list of them (dict-shaped findings with
            every immutable field present also work).
        pages: The ``PageData`` objects the findings claim to be about — the same
            crawl snapshot the audit ran against.

    Returns:
        ``True`` only if every finding re-derives exactly from ``pages``: its page
        exists in the snapshot, and re-running the deterministic checks over that
        snapshot reproduces an identical finding (every field but
        ``suggested_fix``). ``False`` the moment any finding fails to reappear —
        one bad finding invalidates the batch, since a report is only as
        trustworthy as its least-verified line.
    """
    findings = _as_list(output)
    if findings is None:
        return False

    if not findings:
        return True

    pages = list(pages)
    page_urls = {getattr(p, "final_url", None) for p in pages} | {
        getattr(p, "url", None) for p in pages
    }

    recomputed_keys = {_key(f) for f in run_seo_audit(pages)}

    for finding in findings:
        if not _has_all_fields(finding):
            return False
        if _get(finding, "page") not in page_urls:
            return False
        if _key(finding) not in recomputed_keys:
            return False

    return True


def filter_valid_findings(findings: list[Any], pages: list[Any]) -> list[Finding]:
    """Return only the findings that pass :func:`validate_seo`, individually.

    This is the gate Phase 8's CLI should call before writing ``audit.json``: it
    filters out any finding that cannot be re-derived rather than either trusting
    the whole batch or rejecting it wholesale for one bad entry.

    Args:
        findings: Candidate findings.
        pages: The crawl snapshot to validate against.

    Returns:
        The subset of ``findings`` that independently re-derive from ``pages``, in
        their original order.
    """
    pages = list(pages)
    page_urls = {getattr(p, "final_url", None) for p in pages} | {
        getattr(p, "url", None) for p in pages
    }
    recomputed_keys = {_key(f) for f in run_seo_audit(pages)}

    kept = []
    for finding in findings:
        if (
            _has_all_fields(finding)
            and _get(finding, "page") in page_urls
            and _key(finding) in recomputed_keys
        ):
            kept.append(finding)
    return kept


def _as_list(output: Any) -> list[Any] | None:
    """Normalize a single finding or a list into a list; ``None`` on a bad shape."""
    if output is None:
        return None
    if isinstance(output, (Finding, dict)):
        return [output]
    if isinstance(output, list):
        return output
    return None


def _has_all_fields(finding: Any) -> bool:
    """True when every immutable field is present and non-empty."""
    return all(str(_get(finding, field) or "").strip() for field in _IMMUTABLE_FIELDS)


def _key(finding: Any) -> tuple:
    """The tuple of immutable fields a finding is re-confirmed against."""
    return tuple(_get(finding, field) for field in _IMMUTABLE_FIELDS)


def _get(obj: Any, name: str) -> Any:
    """Read an attribute or dict key, tolerating either shape."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
