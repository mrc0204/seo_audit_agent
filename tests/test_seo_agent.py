"""Tests for the Phase 4 deterministic SEO auditor (``app/agents/seo_agent.py``).

The suite has two jobs, and the second matters as much as the first:

1. **Every injected defect is found.** ``tests/fixtures/messy_page.html`` carries a
   known set of issues and the expected ``check_id`` set below was verified by hand
   against that markup, element by element.
2. **Nothing is reported that the evidence does not support.** The clean fixture
   must produce no ``critical`` or ``warning`` findings at all, ``alt=""`` must never
   be flagged, and a link to a page the crawl never fetched must never be called
   broken. A false finding is worse than a missed one here — the brief grades on
   whether findings hold up under manual review.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import pytest

from app.agents.seo_agent import (
    DEFAULT_SUGGESTED_FIX_PROMPT,
    MAX_SUGGESTED_FIX_CHARS,
    THIN_CONTENT_WORD_COUNT,
    TITLE_MAX_CHARS,
    apply_llm_suggested_fixes,
    check_broken_internal_links,
    check_duplicate_content,
    check_duplicate_meta_descriptions,
    check_duplicate_titles,
    check_redirect_chains,
    page_extras,
    run_seo_audit,
)
from app.crawler.crawler import FetchResult
from app.extraction.seo_extractor import build_page_data
from app.models.finding import Finding
from app.models.page import ImageRef, LinkRef, PageData

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

MESSY_URL = "https://messy.example/products"
CLEAN_URL = "https://ridgeline.example/"


def fixture_page(name: str, url: str, status_code: int = 200) -> PageData:
    html = (FIXTURES / name).read_text(encoding="utf-8")
    return build_page_data(url, url, status_code, html)


def make_page(url: str, **overrides) -> PageData:
    """A minimal, defect-free page, so a test's overrides are the only trigger."""
    base = dict(
        url=url,
        final_url=url,
        status_code=200,
        fetched_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        html="<html><head><title>A perfectly reasonable page title here</title></head></html>",
        text="word " * (THIN_CONTENT_WORD_COUNT + 10),
        title="A perfectly reasonable page title here",
        meta_description="d" * 100,
        canonical=url,
        robots_meta=[],
        headings={"h1": ["Heading"], "h2": [], "h3": [], "h4": [], "h5": [], "h6": []},
        images=[],
        links=[],
        structured_data=[],
        content_hash=f"hash-of-{url}",
    )
    base.update(overrides)
    return PageData(**base)


def ids(findings) -> set[str]:
    return {f.check_id for f in findings}


# ---------------------------------------------------------------------------
# Fixture-site snapshot — hand-verified against the markup
# ---------------------------------------------------------------------------

#: Verified by reading messy_page.html element by element:
#: empty <title>; two <link rel=canonical>; meta ROBOTS "NOINDEX, NOFOLLOW";
#: two <h1>; an <h3> with no <h2>; one of four <img> with no alt attribute;
#: no meta description; no og: tags; 44 words of visible text.
EXPECTED_MESSY_CHECK_IDS = {
    "SEO-TITLE-002",  # empty title
    "SEO-META-001",  # missing meta description
    "SEO-CANON-002",  # two canonicals
    "SEO-INDEX-001",  # noindex
    "SEO-INDEX-002",  # nofollow
    "SEO-HEAD-002",  # multiple h1
    "SEO-HEAD-004",  # skipped heading level
    "SEO-IMG-001",  # image missing alt
    "SEO-CONTENT-001",  # thin content
    "SEO-SOCIAL-001",  # missing open graph
}


def test_messy_fixture_produces_exactly_the_expected_findings():
    findings = run_seo_audit([fixture_page("messy_page.html", MESSY_URL)])
    assert ids(findings) == EXPECTED_MESSY_CHECK_IDS


def test_clean_fixture_produces_no_critical_or_warning_findings():
    # The whole point of the audit: a well-formed page must not be nagged at.
    findings = run_seo_audit([fixture_page("clean_page.html", CLEAN_URL)])
    serious = [f for f in findings if f.severity in {"critical", "warning"}]
    assert serious == [], [f"{f.check_id}: {f.evidence}" for f in serious]


# ---------------------------------------------------------------------------
# Priority 3 fix: severity reflects real indexability impact, not "worth fixing"
#
# Locks in the recalibration so a future edit can't silently drift a real
# indexability defect and an advisory best-practice recommendation back onto
# the same severity. Each assertion here was a deliberate, individual decision,
# not a batch relabel — see the module docstring's "Severity is calibrated..."
# section for the criterion applied to every one of them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("check_id", "page_kwargs"),
    [
        # None of these stop a page from being crawled, indexed, or ranked —
        # they affect only the SERP snippet, social preview, or authoring
        # hygiene. Google has stated meta description isn't a ranking factor;
        # a single missing canonical only matters once duplicates exist; H1
        # count is not required by any real search engine to index a page.
        ("SEO-TITLE-004", dict(title="x" * (TITLE_MAX_CHARS + 10))),
        ("SEO-META-001", dict(meta_description=None, html="<html></html>")),
        ("SEO-META-002", dict(meta_description="", html='<html><head><meta name="description" content=""></head></html>')),
        ("SEO-CANON-001", dict(canonical=None, html="<html></html>")),
        ("SEO-HEAD-001", dict(headings={"h1": []})),
        ("SEO-HEAD-002", dict(headings={"h1": ["One", "Two"]})),
    ],
)
def test_advisory_findings_are_info_not_warning(check_id, page_kwargs):
    page = make_page("https://x.example/a", **page_kwargs)
    finding = next(f for f in run_seo_audit([page]) if f.check_id == check_id)
    assert finding.severity == "info", f"{check_id} should be advisory (info), not {finding.severity}"


@pytest.mark.parametrize(
    ("check_id", "expected_severity", "page_kwargs"),
    [
        # These retain real crawler-visible or ranking consequences and must NOT
        # have been swept into the advisory recalibration above.
        ("SEO-TITLE-001", "critical", dict(title=None, html="<html></html>")),
        (
            "SEO-INDEX-001",
            "critical",
            dict(
                robots_meta=["noindex"],
                html='<html><head><meta name="robots" content="noindex"></head></html>',
            ),
        ),
        ("SEO-CANON-002", "critical", dict(
            canonical="https://x.example/a",
            html='<html><head><link rel="canonical" href="https://x.example/a">'
            '<link rel="canonical" href="https://x.example/a2"></head></html>',
        )),
        (
            "SEO-INDEX-002",
            "warning",
            dict(
                robots_meta=["nofollow"],
                html='<html><head><meta name="robots" content="nofollow"></head></html>',
            ),
        ),
        ("SEO-IMG-001", "warning", dict(images=[ImageRef(src="/a.png", alt=None)])),
    ],
)
def test_real_defects_kept_their_original_severity(check_id, expected_severity, page_kwargs):
    page = make_page("https://x.example/a", **page_kwargs)
    finding = next(f for f in run_seo_audit([page]) if f.check_id == check_id)
    assert finding.severity == expected_severity


def test_decorative_alt_is_never_reported_as_missing():
    # messy_page.html has alt="" on a spacer gif. That is correct markup for a
    # decorative image; flagging it would tell the site to undo correct work.
    findings = run_seo_audit([fixture_page("messy_page.html", MESSY_URL)])
    alt_finding = next(f for f in findings if f.check_id == "SEO-IMG-001")
    assert "1 of 4" in alt_finding.evidence
    assert "spacer.gif" not in alt_finding.evidence


def test_every_finding_carries_non_empty_evidence_and_a_valid_severity():
    findings = run_seo_audit(
        [fixture_page("messy_page.html", MESSY_URL), fixture_page("clean_page.html", CLEAN_URL)]
    )
    assert findings
    for finding in findings:
        assert finding.evidence.strip()
        assert finding.suggested_fix.strip()
        assert finding.check_id.startswith("SEO-")
        assert finding.severity in {"critical", "warning", "info"}


def test_findings_are_sorted_most_serious_first():
    findings = run_seo_audit([fixture_page("messy_page.html", MESSY_URL)])
    rank = {"critical": 0, "warning": 1, "info": 2}
    severities = [rank[f.severity] for f in findings]
    assert severities == sorted(severities)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------


def test_missing_title_is_critical():
    findings = run_seo_audit([make_page("https://x.example/a", title=None, html="<html></html>")])
    assert "SEO-TITLE-001" in ids(findings)


def test_empty_title_is_reported_separately_from_missing_title():
    # Distinct defects with distinct fixes; Phase 3 preserves the difference.
    page = make_page("https://x.example/a", title="", html="<html><title></title></html>")
    assert "SEO-TITLE-002" in ids(run_seo_audit([page]))
    assert "SEO-TITLE-001" not in ids(run_seo_audit([page]))


def test_long_title_reports_its_actual_length():
    title = "x" * (TITLE_MAX_CHARS + 15)
    findings = run_seo_audit([make_page("https://x.example/a", title=title)])
    finding = next(f for f in findings if f.check_id == "SEO-TITLE-004")
    assert str(len(title)) in finding.evidence
    assert str(TITLE_MAX_CHARS) in finding.evidence


def test_short_title_is_info_not_warning():
    findings = run_seo_audit([make_page("https://x.example/a", title="Home")])
    finding = next(f for f in findings if f.check_id == "SEO-TITLE-005")
    assert finding.severity == "info"


def test_multiple_titles_detected_from_html_not_page_data():
    # PageData holds one title; the count is re-derived from the markup.
    html = "<html><head><title>First title of the page</title>" "<title>Second</title></head></html>"
    page = make_page("https://x.example/a", html=html, title="First title of the page")
    assert page_extras(page)["title_count"] == 2
    assert "SEO-TITLE-003" in ids(run_seo_audit([page]))


# ---------------------------------------------------------------------------
# Canonical, indexability, headings
# ---------------------------------------------------------------------------


def test_missing_canonical_reported():
    page = make_page("https://x.example/a", canonical=None, html="<html></html>")
    assert "SEO-CANON-001" in ids(run_seo_audit([page]))


def test_off_domain_canonical_is_reported():
    page = make_page(
        "https://x.example/a",
        canonical="https://other.example/a",
        html='<html><head><link rel="canonical" href="https://other.example/a"></head></html>',
    )
    findings = run_seo_audit([page])
    assert "SEO-CANON-003" in ids(findings)


def test_same_domain_canonical_is_not_reported():
    page = make_page(
        "https://x.example/a",
        canonical="https://www.x.example/a",  # www is not a different domain
        html='<html><head><link rel="canonical" href="https://www.x.example/a"></head></html>',
    )
    assert "SEO-CANON-003" not in ids(run_seo_audit([page]))


def test_noindex_is_critical_but_its_fix_asks_for_confirmation():
    page = make_page(
        "https://x.example/a",
        robots_meta=["noindex"],
        html='<html><head><meta name="robots" content="noindex"></head></html>',
    )
    finding = next(f for f in run_seo_audit([page]) if f.check_id == "SEO-INDEX-001")
    assert finding.severity == "critical"
    # The check can see the directive but not the intent; a thank-you page is
    # supposed to be noindex, so the fix must not simply say "remove it".
    assert "confirm" in finding.suggested_fix.lower()


# ---------------------------------------------------------------------------
# X-Robots-Tag HTTP header — Priority 4 fix
# ---------------------------------------------------------------------------


def test_header_only_noindex_is_reported_separately_from_html_noindex():
    # robots_meta carries the Phase-3-merged directive (as a real crawl would
    # produce it), but the HTML itself has no meta tag at all — exactly the
    # "invisible in view-source" scenario this check exists to catch.
    page = make_page(
        "https://x.example/a",
        robots_meta=["noindex"],
        html="<html><head><title>Nothing about robots in this HTML at all</title></head></html>",
    )
    findings = run_seo_audit([page])
    assert "SEO-INDEX-003" in ids(findings)
    assert "SEO-INDEX-001" not in ids(findings)  # not also reported as HTML-sourced

    finding = next(f for f in findings if f.check_id == "SEO-INDEX-003")
    assert finding.severity == "critical"
    assert "X-Robots-Tag" in finding.evidence
    assert "HTTP" in finding.evidence


def test_html_sourced_noindex_does_not_trigger_the_header_specific_finding():
    # The reverse case: a directive that genuinely IS in the HTML must not also be
    # reported as header-only — the two checks must be non-overlapping.
    page = make_page(
        "https://x.example/a",
        robots_meta=["noindex"],
        html='<html><head><meta name="robots" content="noindex"></head></html>',
    )
    findings = run_seo_audit([page])
    assert "SEO-INDEX-001" in ids(findings)
    assert "SEO-INDEX-003" not in ids(findings)


def test_header_only_nofollow_is_warning_and_names_the_header():
    page = make_page(
        "https://x.example/a",
        robots_meta=["nofollow"],
        html="<html><head><title>No robots meta tag here either</title></head></html>",
    )
    finding = next(f for f in run_seo_audit([page]) if f.check_id == "SEO-INDEX-004")
    assert finding.severity == "warning"
    assert "X-Robots-Tag" in finding.evidence


def test_a_page_with_no_header_directives_at_all_triggers_neither_header_check():
    page = make_page("https://x.example/a", robots_meta=[], html="<html></html>")
    findings = run_seo_audit([page])
    assert "SEO-INDEX-003" not in ids(findings)
    assert "SEO-INDEX-004" not in ids(findings)


def test_missing_h1_reported_and_multiple_h1_reported():
    empty = make_page("https://x.example/a", headings={"h1": []})
    assert "SEO-HEAD-001" in ids(run_seo_audit([empty]))

    several = make_page("https://x.example/b", headings={"h1": ["One", "Two"]})
    assert "SEO-HEAD-002" in ids(run_seo_audit([several]))


def test_contiguous_heading_levels_are_not_flagged():
    page = make_page(
        "https://x.example/a",
        headings={"h1": ["A"], "h2": ["B"], "h3": ["C"], "h4": [], "h5": [], "h6": []},
    )
    assert "SEO-HEAD-004" not in ids(run_seo_audit([page]))


def test_thin_content_is_info_and_quotes_the_threshold():
    page = make_page("https://x.example/a", text="only a few words here")
    finding = next(f for f in run_seo_audit([page]) if f.check_id == "SEO-CONTENT-001")
    assert finding.severity == "info"
    assert str(THIN_CONTENT_WORD_COUNT) in finding.evidence
    assert "5 words" in finding.evidence


# ---------------------------------------------------------------------------
# Cross-page checks
# ---------------------------------------------------------------------------


def test_duplicate_titles_reported_on_every_affected_page():
    pages = [
        make_page("https://x.example/a", title="Exactly the same title text"),
        make_page("https://x.example/b", title="Exactly the same title text"),
        make_page("https://x.example/c", title="A completely different title"),
    ]
    findings = check_duplicate_titles(pages)
    assert {f.page for f in findings} == {"https://x.example/a", "https://x.example/b"}


def test_empty_titles_are_not_treated_as_duplicates_of_each_other():
    # Two pages with no title share a "value", but that is the missing-title
    # finding, not a duplicate-title one. Reporting both would double-count.
    pages = [
        make_page("https://x.example/a", title=""),
        make_page("https://x.example/b", title=""),
    ]
    assert check_duplicate_titles(pages) == []


def test_duplicate_meta_descriptions_reported():
    pages = [
        make_page("https://x.example/a", meta_description="Same description on both."),
        make_page("https://x.example/b", meta_description="Same description on both."),
    ]
    assert len(check_duplicate_meta_descriptions(pages)) == 2


def test_duplicate_content_uses_the_content_hash():
    pages = [
        make_page("https://x.example/a", text="Identical prose.", content_hash="same"),
        make_page("https://x.example/b", text="Identical prose.", content_hash="same"),
    ]
    findings = check_duplicate_content(pages)
    assert len(findings) == 2
    assert all(f.check_id == "SEO-DUP-003" for f in findings)


def test_blank_pages_are_not_reported_as_duplicate_content():
    pages = [
        make_page("https://x.example/a", text="", content_hash="empty"),
        make_page("https://x.example/b", text="", content_hash="empty"),
    ]
    assert check_duplicate_content(pages) == []


# ---------------------------------------------------------------------------
# Broken links — the strongest "no unsupported findings" case
# ---------------------------------------------------------------------------


def test_broken_internal_link_reported_when_the_target_was_actually_fetched():
    home = make_page(
        "https://x.example/",
        links=[LinkRef(href="https://x.example/gone", anchor_text="Gone", is_internal=True)],
    )
    gone = make_page("https://x.example/gone", status_code=404)

    findings = check_broken_internal_links([home, gone])
    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert "404" in findings[0].evidence


def test_broken_link_to_a_non_html_resource_is_still_caught_via_fetch_results():
    # Regression companion to main.py's non-HTML exclusion: a broken PDF or
    # markdown link is real evidence of breakage even though that resource never
    # becomes a PageData at all. Without fetch_results, this status is invisible
    # to the check and the link looks merely "uncrawled" rather than broken.
    home = make_page(
        "https://x.example/",
        links=[LinkRef(href="https://x.example/manifest.md", anchor_text="Docs", is_internal=True)],
    )
    manifest_fetch = _fetch("https://x.example/manifest.md", chain=[])
    manifest_fetch = manifest_fetch.model_copy(update={"status_code": 404})

    assert check_broken_internal_links([home]) == []  # invisible without fetch_results
    findings = check_broken_internal_links([home], fetch_results=[manifest_fetch])
    assert len(findings) == 1
    assert "404" in findings[0].evidence


def test_link_to_an_uncrawled_page_is_never_called_broken():
    # The crawl may simply have stopped at max_pages. Calling this broken would be
    # a finding with no evidence behind it.
    home = make_page(
        "https://x.example/",
        links=[LinkRef(href="https://x.example/never-fetched", anchor_text="X", is_internal=True)],
    )
    assert check_broken_internal_links([home]) == []


def test_external_links_are_not_judged_by_the_internal_link_check():
    home = make_page(
        "https://x.example/",
        links=[LinkRef(href="https://other.example/gone", anchor_text="X", is_internal=False)],
    )
    other = make_page("https://other.example/gone", status_code=404)
    assert check_broken_internal_links([home, other]) == []


def test_healthy_internal_links_produce_nothing():
    home = make_page(
        "https://x.example/",
        links=[LinkRef(href="https://x.example/about", anchor_text="About", is_internal=True)],
    )
    about = make_page("https://x.example/about", status_code=200)
    assert check_broken_internal_links([home, about]) == []


# ---------------------------------------------------------------------------
# Redirect chains
# ---------------------------------------------------------------------------


def _fetch(final_url: str, chain: list[str]) -> FetchResult:
    return FetchResult(
        url=chain[0] if chain else final_url,
        final_url=final_url,
        status_code=200,
        fetched_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        redirect_chain=chain,
    )


def test_single_hop_redirect_is_not_reported():
    # http->https or a trailing-slash fix is normal and healthy.
    results = [_fetch("https://x.example/a", ["http://x.example/a"])]
    assert check_redirect_chains(results) == []


def test_multi_hop_redirect_chain_is_reported_with_every_hop():
    results = [_fetch("https://x.example/c", ["http://x.example/a", "https://x.example/b"])]
    findings = check_redirect_chains(results)
    assert len(findings) == 1
    assert findings[0].check_id == "SEO-REDIR-001"
    for hop in ("http://x.example/a", "https://x.example/b", "https://x.example/c"):
        assert hop in findings[0].evidence


def test_redirect_check_is_skipped_when_fetch_results_are_absent():
    # PageData cannot express hop count, so the check declines rather than guesses.
    page = make_page("https://x.example/a")
    assert "SEO-REDIR-001" not in ids(run_seo_audit([page]))
    assert "SEO-REDIR-001" not in ids(run_seo_audit([page], fetch_results=[]))


# ---------------------------------------------------------------------------
# The LLM boundary — plan risk #3
# ---------------------------------------------------------------------------


def _finding() -> Finding:
    return Finding(
        metric="missing_meta_description",
        page="https://x.example/a",
        severity="warning",
        evidence='No <meta name="description"> element is present.',
        suggested_fix="Deterministic fix text.",
        check_id="SEO-META-001",
    )


def test_llm_can_replace_only_the_suggested_fix():
    original = _finding()
    updated = apply_llm_suggested_fixes([original], lambda _: "Write a 150-character summary.")[0]

    assert updated.suggested_fix == "Write a 150-character summary."
    # Everything else is copied from the original, not from the model.
    assert updated.metric == original.metric
    assert updated.page == original.page
    assert updated.severity == original.severity
    assert updated.evidence == original.evidence
    assert updated.check_id == original.check_id


def test_llm_cannot_reach_evidence_even_when_it_tries():
    # A model returning a JSON object that reassigns evidence/metric has no channel
    # to those fields: only the returned string is used, and only for suggested_fix.
    hostile = '{"evidence": "FABRICATED", "metric": "invented_metric", "suggested_fix": "x"}'
    original = _finding()
    updated = apply_llm_suggested_fixes([original], lambda _: hostile)[0]

    assert updated.evidence == original.evidence
    assert updated.metric == original.metric
    assert "FABRICATED" not in updated.evidence
    assert "invented_metric" not in updated.metric


def test_input_findings_are_never_mutated():
    original = _finding()
    apply_llm_suggested_fixes([original], lambda _: "A replacement fix.")
    assert original.suggested_fix == "Deterministic fix text."


@pytest.mark.parametrize(
    "response",
    ["", "   ", None, "x" * (MAX_SUGGESTED_FIX_CHARS + 1)],
)
def test_unusable_llm_output_falls_back_to_the_deterministic_fix(response):
    updated = apply_llm_suggested_fixes([_finding()], lambda _: response)[0]
    assert updated.suggested_fix == "Deterministic fix text."


def test_llm_echoing_the_evidence_is_rejected():
    original = _finding()
    updated = apply_llm_suggested_fixes([original], lambda _: original.evidence)[0]
    assert updated.suggested_fix == "Deterministic fix text."


def test_llm_failure_degrades_phrasing_not_the_audit():
    def explode(_prompt: str) -> str:
        raise RuntimeError("free tier rate limit")

    updated = apply_llm_suggested_fixes([_finding()], explode)
    assert len(updated) == 1
    assert updated[0].suggested_fix == "Deterministic fix text."
    assert updated[0].evidence == _finding().evidence


def test_no_llm_configured_returns_findings_unchanged():
    findings = [_finding()]
    assert apply_llm_suggested_fixes(findings, None) == findings


def test_prompt_carries_the_evidence_and_forbids_restating_it():
    seen: list[str] = []
    apply_llm_suggested_fixes([_finding()], lambda p: seen.append(p) or "A fix.")
    assert len(seen) == 1
    assert 'No <meta name="description"> element is present.' in seen[0]
    assert "missing_meta_description" in seen[0]
    assert "Do not restate the evidence" in DEFAULT_SUGGESTED_FIX_PROMPT


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_audit_of_no_pages_returns_no_findings():
    assert run_seo_audit([]) == []


def test_non_page_data_entries_are_ignored():
    assert run_seo_audit(["not a page", None, 42]) == []


def test_audit_survives_a_page_with_empty_html():
    page = build_page_data("https://x.example/", "https://x.example/", 200, "")
    findings = run_seo_audit([page])
    assert "SEO-TITLE-001" in ids(findings)
    assert all(f.evidence for f in findings)


def test_image_only_page_reports_alt_once_not_per_image():
    page = make_page(
        "https://x.example/a",
        images=[ImageRef(src=f"/img/{i}.png", alt=None) for i in range(12)],
    )
    alt_findings = [f for f in run_seo_audit([page]) if f.check_id == "SEO-IMG-001"]
    assert len(alt_findings) == 1
    assert "12 of 12" in alt_findings[0].evidence
    assert "and 7 more" in alt_findings[0].evidence
