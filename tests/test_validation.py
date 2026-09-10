"""Tests for the Phase 7 cross-cutting validation layer.

The plan's definition of done, verbatim: "deliberately inject one bad
(hallucinated) finding/answer in a test and confirm the validator rejects it
before it reaches outputs/." Every ``test_hallucinated_*`` test below does exactly
that — takes a genuine, agent-produced output and tampers one field, then asserts
the validator catches it. The genuine-output tests exist so a validator that
rejects everything can't pass by accident.

``qa_validator`` was built and already tested in Phase 6 (``tests/test_qa.py``);
this file adds only the Phase 7 addition, :func:`filter_valid_answer`, for parity
with the other two.
"""

from __future__ import annotations

import pathlib

from app.agents.nap_agent import run_nap_check
from app.agents.seo_agent import run_seo_audit
from app.extraction.seo_extractor import build_page_data
from app.models.answer import QAAnswer
from app.models.finding import Finding
from app.models.nap import NAPValue
from app.validation.nap_validator import filter_valid_comparisons, validate_nap
from app.validation.qa_validator import filter_valid_answer, validate_qa
from app.validation.seo_validator import filter_valid_findings, validate_seo

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture_page(name: str, url: str):
    return build_page_data(url, url, 200, (FIXTURES / name).read_text(encoding="utf-8"))


def messy_page():
    return fixture_page("messy_page.html", "https://messy.example/products")


def clean_page():
    return fixture_page("clean_page.html", "https://ridgeline.example/")


def contact_page():
    return fixture_page("nap_contact_page.html", "https://ridgeline.example/contact")


# ---------------------------------------------------------------------------
# seo_validator — genuine output
# ---------------------------------------------------------------------------


def test_genuine_seo_findings_validate():
    page = messy_page()
    findings = run_seo_audit([page])
    assert findings  # the fixture is meant to trip several checks
    assert validate_seo(findings, [page]) is True


def test_genuine_single_finding_validates():
    page = messy_page()
    finding = run_seo_audit([page])[0]
    assert validate_seo(finding, [page]) is True


def test_llm_rewritten_suggested_fix_still_validates():
    # suggested_fix is the one field Phase 4's LLM step may rewrite. A validator
    # that required it to match verbatim would reject legitimate output.
    page = messy_page()
    finding = run_seo_audit([page])[0]
    rewritten = finding.model_copy(update={"suggested_fix": "A completely different phrasing."})
    assert validate_seo(rewritten, [page]) is True


# ---------------------------------------------------------------------------
# seo_validator — hallucination injection (the plan's required test)
# ---------------------------------------------------------------------------


def test_hallucinated_evidence_is_rejected():
    page = messy_page()
    finding = run_seo_audit([page])[0]
    hallucinated = finding.model_copy(
        update={"evidence": "This sentence never appeared anywhere on the page."}
    )
    assert validate_seo(hallucinated, [page]) is False


def test_hallucinated_finding_on_a_clean_page_is_rejected():
    # A finding fabricated for a page that has no such defect at all.
    page = clean_page()
    fake = Finding(
        metric="missing_title",
        page=page.final_url,
        severity="critical",
        evidence="No <title> element is present in the document <head>.",
        suggested_fix="Add a title.",
        check_id="SEO-TITLE-001",
    )
    assert validate_seo(fake, [page]) is False


def test_finding_citing_an_uncrawled_page_is_rejected():
    page = messy_page()
    finding = run_seo_audit([page])[0]
    elsewhere = finding.model_copy(update={"page": "https://never-crawled.example/"})
    assert validate_seo(elsewhere, [page]) is False


def test_finding_with_severity_upgraded_is_rejected():
    # Changing severity without changing evidence is its own kind of fabrication.
    page = messy_page()
    finding = next(f for f in run_seo_audit([page]) if f.severity == "info")
    upgraded = finding.model_copy(update={"severity": "critical"})
    assert validate_seo(upgraded, [page]) is False


def test_finding_missing_check_id_is_rejected():
    # Nothing to re-run without knowing which rule produced it.
    page = messy_page()
    finding = run_seo_audit([page])[0]
    untraceable = finding.model_copy(update={"check_id": ""})
    assert validate_seo(untraceable, [page]) is False


def test_filter_valid_findings_drops_only_the_bad_one():
    page = messy_page()
    genuine = run_seo_audit([page])
    hallucinated = genuine[0].model_copy(update={"evidence": "FABRICATED"})

    kept = filter_valid_findings([hallucinated] + genuine, [page])
    assert hallucinated not in kept
    assert len(kept) == len(genuine)
    assert set(id(f) for f in kept) == set(id(f) for f in genuine)


def test_validate_seo_of_empty_list_is_true():
    assert validate_seo([], [messy_page()]) is True


# ---------------------------------------------------------------------------
# nap_validator — genuine output
# ---------------------------------------------------------------------------


def test_genuine_nap_comparisons_validate():
    pages = [clean_page(), contact_page()]
    comparisons = run_nap_check(pages)
    assert validate_nap(comparisons, pages) is True


# ---------------------------------------------------------------------------
# nap_validator — hallucination injection
# ---------------------------------------------------------------------------


def test_hallucinated_nap_raw_value_is_rejected():
    pages = [clean_page(), contact_page()]
    phone = next(c for c in run_nap_check(pages) if c.field == "phone")

    fabricated = phone.evidence[0].model_copy(update={"raw_value": "999-999-9999"})
    tampered = phone.model_copy(update={"evidence": [fabricated] + phone.evidence[1:]})

    assert validate_nap(tampered, pages) is False


def test_nap_value_attributed_to_the_wrong_source_is_rejected():
    # The raw text is genuinely on the page, but claimed from a source that did
    # not actually produce it — must be caught even though a bare substring search
    # of the page would find the text.
    pages = [clean_page(), contact_page()]
    phone = next(c for c in run_nap_check(pages) if c.field == "phone")

    real_value = next(v for v in phone.evidence if v.source == "json_ld")
    relabeled = real_value.model_copy(update={"source": "visible_text"})
    tampered = phone.model_copy(
        update={"evidence": [relabeled] + [v for v in phone.evidence if v is not real_value]}
    )

    assert validate_nap(tampered, pages) is False


def test_nap_normalized_value_drift_is_rejected():
    pages = [clean_page(), contact_page()]
    phone = next(c for c in run_nap_check(pages) if c.field == "phone")

    drifted = phone.evidence[0].model_copy(update={"normalized_value": "0000000000"})
    tampered = phone.model_copy(update={"evidence": [drifted] + phone.evidence[1:]})

    assert validate_nap(tampered, pages) is False


def test_nap_verdict_not_matching_its_own_evidence_is_rejected():
    pages = [clean_page(), contact_page()]
    phone = next(c for c in run_nap_check(pages) if c.field == "phone")
    assert phone.verdict == "consistent"

    lied = phone.model_copy(update={"verdict": "inconsistent"})
    assert validate_nap(lied, pages) is False


def test_nap_confidence_not_matching_its_own_evidence_is_rejected():
    pages = [clean_page(), contact_page()]
    phone = next(c for c in run_nap_check(pages) if c.field == "phone")

    lied = phone.model_copy(update={"confidence": 0.1})
    assert validate_nap(lied, pages) is False


def test_nap_comparison_citing_an_uncrawled_page_is_rejected():
    pages = [clean_page(), contact_page()]
    phone = next(c for c in run_nap_check(pages) if c.field == "phone")

    elsewhere = phone.evidence[0].model_copy(update={"page": "https://never-crawled.example/"})
    tampered = phone.model_copy(update={"evidence": [elsewhere] + phone.evidence[1:]})

    assert validate_nap(tampered, pages) is False


def test_filter_valid_comparisons_drops_only_the_bad_one():
    pages = [clean_page(), contact_page()]
    genuine = run_nap_check(pages)
    phone = next(c for c in genuine if c.field == "phone")
    tampered = phone.model_copy(update={"verdict": "inconsistent"})

    replaced = [tampered if c.field == "phone" else c for c in genuine]
    kept = filter_valid_comparisons(replaced, pages)

    assert len(kept) == len(genuine) - 1
    assert all(c.field != "phone" or c.verdict != "inconsistent" for c in kept)


def test_validate_nap_rejects_a_non_comparison_object():
    assert validate_nap({"field": "phone"}, [clean_page()]) is False
    assert validate_nap(None, [clean_page()]) is False


# ---------------------------------------------------------------------------
# qa_validator — Phase 7 addition (filter_valid_answer)
# ---------------------------------------------------------------------------


def test_filter_valid_answer_passes_through_a_genuine_answer():
    page = clean_page()
    answer = QAAnswer(
        query="When are the cupping sessions?",
        url=page.final_url,
        excerpt="Free cupping sessions every Saturday at ten in the morning.",
        match_type="exact_substring",
    )
    assert filter_valid_answer(answer, [page]) is answer


def test_filter_valid_answer_nulls_a_hallucinated_excerpt():
    page = clean_page()
    hallucinated = QAAnswer(
        query="When are the cupping sessions?",
        url=page.final_url,
        excerpt="Cuppings happen every single Saturday morning around 10am.",
        match_type="exact_substring",
    )
    result = filter_valid_answer(hallucinated, [page])
    assert result.url is None
    assert result.excerpt is None
    assert result.match_type == "none"
    assert result.query == hallucinated.query  # the question itself is preserved


def test_filter_valid_answer_passes_through_a_correct_null():
    answer = QAAnswer(query="Anything?", url=None, excerpt=None, match_type="none")
    assert filter_valid_answer(answer, [clean_page()]) is answer


def test_validate_qa_still_works_as_the_phase6_gate(clean_page_fixture=None):
    page = clean_page()
    genuine = QAAnswer(
        query="q",
        url=page.final_url,
        excerpt="Free cupping sessions every Saturday at ten in the morning.",
        match_type="exact_substring",
    )
    assert validate_qa(genuine, [page]) is True
