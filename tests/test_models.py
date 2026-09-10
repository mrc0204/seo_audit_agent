"""Tests for the Phase 1 data-contract models (``app/models/*``).

Two layers of assertion here:

1. The *model* layer, locked by EXECUTION_PLAN.md Phase 1 — exact field sets,
   exact ``Literal`` membership, rejection of invalid enum values, and round-trip
   fidelity through both Python-mode and JSON-mode serialization.
2. The *deliverable* layer, locked by the assignment brief — the exact JSON shapes
   of ``audit.json`` (Q1), ``nap_report.json`` (Q2) and ``answer.json`` (Q3). The
   models carry three fields the brief's shapes do not (``Finding.check_id``,
   ``NAPComparison.evidence``, ``QAAnswer.match_type``) because the plan requires
   them for validation and traceability, so each model exposes a serializer that
   emits the brief's shape exactly. Those serializers are tested by exact
   dict-equality, not subset matching, so an accidental extra key fails loudly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.answer import QAAnswer
from app.models.finding import Finding
from app.models.nap import NAPComparison, NAPValue
from app.models.page import ImageRef, LinkRef, PageData

# ---------------------------------------------------------------------------
# PageData (the crawl/extraction seam)
# ---------------------------------------------------------------------------


def make_page_data(**overrides) -> PageData:
    base = dict(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        fetched_at=datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc),
        html="<html><body><h1>Hi</h1></body></html>",
        text="Hi",
        title="Example Domain",
        meta_description="An example page.",
        canonical="https://example.com/",
        robots_meta=["index", "follow"],
        headings={"h1": ["Hi"], "h2": []},
        images=[ImageRef(src="/logo.png", alt="Logo")],
        links=[LinkRef(href="https://example.com/about", anchor_text="About", is_internal=True)],
        structured_data=[{"@type": "Organization", "name": "Example"}],
        content_hash="abc123",
    )
    base.update(overrides)
    return PageData(**base)


def test_image_ref_shape():
    assert ImageRef(src="/logo.png", alt="Logo").model_dump() == {
        "src": "/logo.png",
        "alt": "Logo",
    }


def test_image_ref_alt_is_optional():
    img = ImageRef(src="/logo.png")
    assert img.alt is None
    assert img.model_dump() == {"src": "/logo.png", "alt": None}


def test_link_ref_shape():
    link = LinkRef(href="https://example.com/about", anchor_text="About", is_internal=True)
    assert link.model_dump() == {
        "href": "https://example.com/about",
        "anchor_text": "About",
        "is_internal": True,
    }


def test_page_data_field_set_matches_contract():
    assert set(make_page_data().model_dump().keys()) == {
        "url",
        "final_url",
        "status_code",
        "fetched_at",
        "html",
        "text",
        "title",
        "meta_description",
        "canonical",
        "robots_meta",
        "headings",
        "images",
        "links",
        "structured_data",
        "content_hash",
    }


def test_page_data_python_mode_serialization():
    dumped = make_page_data().model_dump()
    assert dumped["url"] == "https://example.com/"
    assert dumped["final_url"] == "https://example.com/"
    assert dumped["status_code"] == 200
    assert dumped["fetched_at"] == datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
    assert dumped["title"] == "Example Domain"
    assert dumped["robots_meta"] == ["index", "follow"]
    assert dumped["headings"] == {"h1": ["Hi"], "h2": []}
    assert dumped["images"] == [{"src": "/logo.png", "alt": "Logo"}]
    assert dumped["links"] == [
        {"href": "https://example.com/about", "anchor_text": "About", "is_internal": True}
    ]
    assert dumped["structured_data"] == [{"@type": "Organization", "name": "Example"}]
    assert dumped["content_hash"] == "abc123"


def test_page_data_json_mode_serializes_datetime_to_iso8601():
    page = make_page_data()
    assert page.model_dump(mode="json")["fetched_at"] == "2026-01-15T12:30:00Z"
    parsed = json.loads(page.model_dump_json())
    assert parsed["fetched_at"] == "2026-01-15T12:30:00Z"
    assert parsed["images"] == [{"src": "/logo.png", "alt": "Logo"}]


def test_page_data_nullable_fields_accept_none():
    dumped = make_page_data(title=None, meta_description=None, canonical=None).model_dump()
    assert dumped["title"] is None
    assert dumped["meta_description"] is None
    assert dumped["canonical"] is None


def test_page_data_round_trip():
    page = make_page_data()
    assert PageData.model_validate(page.model_dump()) == page


def test_page_data_requires_all_fields():
    with pytest.raises(ValidationError):
        PageData(url="https://example.com/")


# ---------------------------------------------------------------------------
# Q1 output — Finding
# ---------------------------------------------------------------------------


def test_q1_finding_serializes_to_expected_json_shape():
    finding = Finding(
        metric="missing_meta_description",
        page="https://example.com/",
        severity="warning",
        evidence='<meta name="description"> tag absent from <head>.',
        suggested_fix="Add a unique meta description between 120-160 characters.",
        check_id="SEO-META-DESC-001",
    )
    assert json.loads(finding.model_dump_json()) == {
        "metric": "missing_meta_description",
        "page": "https://example.com/",
        "severity": "warning",
        "evidence": '<meta name="description"> tag absent from <head>.',
        "suggested_fix": "Add a unique meta description between 120-160 characters.",
        "check_id": "SEO-META-DESC-001",
    }


@pytest.mark.parametrize("severity", ["critical", "warning", "info"])
def test_finding_accepts_each_valid_severity(severity):
    finding = Finding(
        metric="m", page="p", severity=severity, evidence="e", suggested_fix="f", check_id="c"
    )
    assert finding.severity == severity


def test_finding_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        Finding(
            metric="m",
            page="p",
            severity="urgent",
            evidence="e",
            suggested_fix="f",
            check_id="c",
        )


def test_finding_round_trip_json():
    finding = Finding(
        metric="m", page="p", severity="critical", evidence="e", suggested_fix="f", check_id="c"
    )
    assert Finding.model_validate_json(finding.model_dump_json()) == finding


# ---------------------------------------------------------------------------
# Q2 output — NAPValue / NAPComparison
# ---------------------------------------------------------------------------


def test_nap_value_shape():
    value = NAPValue(
        page="https://example.com/contact",
        raw_value="+91 98765 43210",
        normalized_value="9876543210",
        source="tel_link",
    )
    assert value.model_dump() == {
        "page": "https://example.com/contact",
        "raw_value": "+91 98765 43210",
        "normalized_value": "9876543210",
        "source": "tel_link",
    }


@pytest.mark.parametrize("source", ["visible_text", "json_ld", "microdata", "tel_link"])
def test_nap_value_accepts_each_valid_source(source):
    assert NAPValue(page="p", raw_value="r", normalized_value="n", source=source).source == source


def test_nap_value_rejects_invalid_source():
    with pytest.raises(ValidationError):
        NAPValue(page="p", raw_value="r", normalized_value="n", source="og_tag")


def test_q2_comparison_serializes_to_expected_json_shape():
    # Phone example taken from EXECUTION_PLAN.md Phase 5: "+91 98765 43210" and
    # "919876543210" must normalize to the same value.
    comparison = NAPComparison(
        field="phone",
        pages_compared=["https://example.com/", "https://example.com/contact"],
        values=["+91 98765 43210", "919876543210"],
        normalized_values=["9876543210", "9876543210"],
        confidence=1.0,
        verdict="consistent",
        evidence=[
            NAPValue(
                page="https://example.com/",
                raw_value="+91 98765 43210",
                normalized_value="9876543210",
                source="visible_text",
            ),
            NAPValue(
                page="https://example.com/contact",
                raw_value="919876543210",
                normalized_value="9876543210",
                source="tel_link",
            ),
        ],
    )
    assert json.loads(comparison.model_dump_json()) == {
        "field": "phone",
        "pages_compared": ["https://example.com/", "https://example.com/contact"],
        "values": ["+91 98765 43210", "919876543210"],
        "normalized_values": ["9876543210", "9876543210"],
        "confidence": 1.0,
        "verdict": "consistent",
        "evidence": [
            {
                "page": "https://example.com/",
                "raw_value": "+91 98765 43210",
                "normalized_value": "9876543210",
                "source": "visible_text",
            },
            {
                "page": "https://example.com/contact",
                "raw_value": "919876543210",
                "normalized_value": "9876543210",
                "source": "tel_link",
            },
        ],
    }


@pytest.mark.parametrize("field", ["name", "address", "phone"])
def test_nap_comparison_accepts_each_valid_field(field):
    comparison = NAPComparison(
        field=field,
        pages_compared=["p1"],
        values=["v1"],
        normalized_values=["v1"],
        confidence=0.5,
        verdict="insufficient_data",
        evidence=[],
    )
    assert comparison.field == field


@pytest.mark.parametrize("verdict", ["consistent", "inconsistent", "insufficient_data"])
def test_nap_comparison_accepts_each_valid_verdict(verdict):
    comparison = NAPComparison(
        field="name",
        pages_compared=["p1"],
        values=["v1"],
        normalized_values=["v1"],
        confidence=0.5,
        verdict=verdict,
        evidence=[],
    )
    assert comparison.verdict == verdict


def test_nap_comparison_rejects_invalid_field_and_verdict():
    with pytest.raises(ValidationError):
        NAPComparison(
            field="email",
            pages_compared=[],
            values=[],
            normalized_values=[],
            confidence=0.0,
            verdict="consistent",
            evidence=[],
        )
    with pytest.raises(ValidationError):
        NAPComparison(
            field="name",
            pages_compared=[],
            values=[],
            normalized_values=[],
            confidence=0.0,
            verdict="maybe",
            evidence=[],
        )


def test_nap_comparison_round_trip_json():
    comparison = NAPComparison(
        field="address",
        pages_compared=["https://example.com/", "https://example.com/contact"],
        values=["123 Main St", "123 Main Street"],
        normalized_values=["123 main street", "123 main street"],
        confidence=1.0,
        verdict="consistent",
        evidence=[
            NAPValue(
                page="https://example.com/",
                raw_value="123 Main St",
                normalized_value="123 main street",
                source="visible_text",
            )
        ],
    )
    assert NAPComparison.model_validate_json(comparison.model_dump_json()) == comparison


# ---------------------------------------------------------------------------
# Q3 output — QAAnswer
# ---------------------------------------------------------------------------


def test_q3_answer_serializes_to_expected_json_shape_when_answerable():
    answer = QAAnswer(
        query="What are your business hours?",
        url="https://example.com/contact",
        excerpt="Open Monday to Friday, 9am to 6pm.",
        match_type="exact_substring",
    )
    assert json.loads(answer.model_dump_json()) == {
        "query": "What are your business hours?",
        "url": "https://example.com/contact",
        "excerpt": "Open Monday to Friday, 9am to 6pm.",
        "match_type": "exact_substring",
    }


def test_q3_answer_serializes_to_null_shape_when_unanswerable():
    # Per EXECUTION_PLAN.md Phase 6: when nothing qualifies, or the excerpt fails
    # the exact-substring gate, the answer must be an explicit null result.
    answer = QAAnswer(
        query="What is your refund policy?",
        url=None,
        excerpt=None,
        match_type="none",
    )
    assert json.loads(answer.model_dump_json()) == {
        "query": "What is your refund policy?",
        "url": None,
        "excerpt": None,
        "match_type": "none",
    }


def test_q3_answer_match_type_is_itself_nullable():
    # match_type is Literal[...] | None — a null match_type is distinct from "none".
    answer = QAAnswer(query="Anything?", url=None, excerpt=None, match_type=None)
    assert answer.match_type is None
    assert answer.model_dump()["match_type"] is None


def test_q3_answer_rejects_invalid_match_type():
    with pytest.raises(ValidationError):
        QAAnswer(query="q", url=None, excerpt=None, match_type="fuzzy_match")


def test_q3_answer_round_trip_json():
    answer = QAAnswer(
        query="q", url="https://example.com/", excerpt="e", match_type="exact_substring"
    )
    assert QAAnswer.model_validate_json(answer.model_dump_json()) == answer


# ---------------------------------------------------------------------------
# Deliverable JSON shapes (assignment brief)
#
# Brief:
#   audit.json      -> one entry per finding:
#                      {metric, page, severity, evidence, suggested_fix}
#   nap_report.json -> {field, pages_compared, values, normalized_values,
#                       confidence, verdict}
#   answer.json     -> {query, url, excerpt}; url and excerpt null when the site
#                      does not support an answer
# ---------------------------------------------------------------------------

AUDIT_ENTRY_KEYS = {"metric", "page", "severity", "evidence", "suggested_fix"}
NAP_REPORT_KEYS = {
    "field",
    "pages_compared",
    "values",
    "normalized_values",
    "confidence",
    "verdict",
}
ANSWER_KEYS = {"query", "url", "excerpt"}


def test_audit_json_entry_matches_brief_shape_exactly():
    finding = Finding(
        metric="missing_meta_description",
        page="https://example.com/",
        severity="warning",
        evidence='<meta name="description"> tag absent from <head>.',
        suggested_fix="Add a unique meta description between 120-160 characters.",
        check_id="SEO-META-DESC-001",
    )
    entry = finding.to_audit_entry()
    assert set(entry.keys()) == AUDIT_ENTRY_KEYS
    assert entry == {
        "metric": "missing_meta_description",
        "page": "https://example.com/",
        "severity": "warning",
        "evidence": '<meta name="description"> tag absent from <head>.',
        "suggested_fix": "Add a unique meta description between 120-160 characters.",
    }
    # check_id stays on the model for validator traceability, just not in the output.
    assert finding.check_id == "SEO-META-DESC-001"


def test_audit_json_entry_is_json_serializable():
    finding = Finding(
        metric="m", page="p", severity="info", evidence="e", suggested_fix="f", check_id="c"
    )
    assert json.loads(json.dumps(finding.to_audit_entry())) == finding.to_audit_entry()


def test_audit_json_is_a_list_of_entries():
    # "audit.json — one entry per finding"
    findings = [
        Finding(
            metric="missing_title",
            page="https://example.com/a",
            severity="critical",
            evidence="<title> element is absent from <head>.",
            suggested_fix="Add a descriptive <title>.",
            check_id="SEO-TITLE-001",
        ),
        Finding(
            metric="image_missing_alt",
            page="https://example.com/b",
            severity="warning",
            evidence='<img src="/logo.png"> has no alt attribute.',
            suggested_fix="Add descriptive alt text.",
            check_id="SEO-IMG-ALT-001",
        ),
    ]
    payload = [f.to_audit_entry() for f in findings]
    assert isinstance(payload, list) and len(payload) == 2
    assert all(set(entry.keys()) == AUDIT_ENTRY_KEYS for entry in payload)


def test_nap_report_json_matches_brief_shape_exactly():
    comparison = NAPComparison(
        field="phone",
        pages_compared=["https://example.com/", "https://example.com/contact"],
        values=["+91 98765 43210", "919876543210"],
        normalized_values=["9876543210", "9876543210"],
        confidence=1.0,
        verdict="consistent",
        evidence=[
            NAPValue(
                page="https://example.com/",
                raw_value="+91 98765 43210",
                normalized_value="9876543210",
                source="visible_text",
            )
        ],
    )
    entry = comparison.to_report_entry()
    assert set(entry.keys()) == NAP_REPORT_KEYS
    assert entry == {
        "field": "phone",
        "pages_compared": ["https://example.com/", "https://example.com/contact"],
        "values": ["+91 98765 43210", "919876543210"],
        "normalized_values": ["9876543210", "9876543210"],
        "confidence": 1.0,
        "verdict": "consistent",
    }
    # Evidence stays on the model so nap_validator.py can re-verify it later.
    assert comparison.evidence[0].raw_value == "+91 98765 43210"


def test_nap_report_json_is_json_serializable():
    comparison = NAPComparison(
        field="address",
        pages_compared=["https://example.com/"],
        values=["123 Main St"],
        normalized_values=["123 main street"],
        confidence=0.5,
        verdict="insufficient_data",
        evidence=[],
    )
    assert json.loads(json.dumps(comparison.to_report_entry())) == comparison.to_report_entry()


def test_answer_json_matches_brief_shape_exactly_when_answerable():
    answer = QAAnswer(
        query="What are your business hours?",
        url="https://example.com/contact",
        excerpt="Open Monday to Friday, 9am to 6pm.",
        match_type="exact_substring",
    )
    payload = answer.to_answer_json()
    assert set(payload.keys()) == ANSWER_KEYS
    assert payload == {
        "query": "What are your business hours?",
        "url": "https://example.com/contact",
        "excerpt": "Open Monday to Friday, 9am to 6pm.",
    }


@pytest.mark.parametrize("match_type", ["none", None])
def test_answer_json_is_null_when_site_does_not_support_an_answer(match_type):
    # Brief: "url and excerpt null when the site does not support an answer".
    # Both internal encodings of "no answer" must produce the same nulled payload.
    answer = QAAnswer(
        query="What is your refund policy?",
        url=None,
        excerpt=None,
        match_type=match_type,
    )
    payload = answer.to_answer_json()
    assert set(payload.keys()) == ANSWER_KEYS
    assert payload == {
        "query": "What is your refund policy?",
        "url": None,
        "excerpt": None,
    }
    assert json.loads(json.dumps(payload))["url"] is None


def test_answer_json_is_json_serializable():
    answer = QAAnswer(
        query="q", url="https://example.com/", excerpt="e", match_type="exact_substring"
    )
    assert json.loads(json.dumps(answer.to_answer_json())) == answer.to_answer_json()
