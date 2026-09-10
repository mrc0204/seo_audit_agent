"""Tests for the web UI's HTTP layer (``app/web.py``).

``app.web`` contains no audit logic of its own — it only calls
:func:`app.main.run_pipeline` and serializes the result — so these tests patch
``run_pipeline`` with a canned :class:`~app.main.PipelineResult` rather than
crawling a real (or even mocked-transport) site. Phase 8's own tests
(``tests/test_main.py``) already cover the pipeline itself end-to-end offline;
this file's job is only to confirm the HTTP contract: request validation, status
codes, and that the response is exactly the three brief-shaped deliverables the
CLI also writes to disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import PipelineResult
from app.models.answer import QAAnswer
from app.models.finding import Finding
from app.models.nap import NAPComparison, NAPValue
from app.web import app

client = TestClient(app)


def _finding(metric="missing_canonical", severity="info") -> Finding:
    return Finding(
        metric=metric,
        page="https://example.com/",
        severity=severity,
        evidence="No <link rel=\"canonical\"> element is present in the document <head>.",
        suggested_fix="Add a self-referencing canonical link.",
        check_id="SEO-CANON-001",
    )


def _nap_comparison(field="phone", verdict="consistent") -> NAPComparison:
    value = NAPValue(
        page="https://example.com/",
        raw_value="+1 503-555-0147",
        normalized_value="5035550147",
        source="json_ld",
    )
    return NAPComparison(
        field=field,
        pages_compared=["https://example.com/"],
        values=["+1 503-555-0147"],
        normalized_values=["5035550147"],
        confidence=1.0,
        verdict=verdict,
        evidence=[value],
    )


def _canned_result(with_answer: bool = False) -> PipelineResult:
    answer = None
    if with_answer:
        answer = QAAnswer(
            query="What are your hours?",
            url="https://example.com/",
            excerpt="Open Monday to Friday, 9am to 5pm.",
            match_type="exact_substring",
        )
    return PipelineResult(
        start_url="https://example.com/",
        pages=[object()],  # only len() is used by app.web
        findings=[_finding()],
        nap_comparisons=[_nap_comparison()],
        answer=answer,
        crawl_notes=["robots.txt status: missing"],
        pages_fetched=1,
    )


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_serves_the_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Evidence-Grounded SEO Audit Agent" in response.text


# ---------------------------------------------------------------------------
# POST /api/audit — request validation
# ---------------------------------------------------------------------------


def test_audit_rejects_a_non_crawlable_url():
    response = client.post("/api/audit", json={"url": "javascript:void(0)"})
    assert response.status_code == 422


def test_audit_rejects_a_missing_url():
    response = client.post("/api/audit", json={})
    assert response.status_code == 422


@pytest.mark.parametrize("field,value", [("max_pages", 0), ("max_pages", 500), ("max_depth", -1), ("timeout", 0)])
def test_audit_rejects_out_of_range_options(field, value):
    payload = {"url": "https://example.com/", field: value}
    response = client.post("/api/audit", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/audit — success path (run_pipeline patched, no network)
# ---------------------------------------------------------------------------


def test_audit_returns_the_briefs_three_deliverable_shapes():
    with patch("app.web.run_pipeline", return_value=_canned_result()) as mock_run:
        response = client.post("/api/audit", json={"url": "https://example.com/"})

    assert response.status_code == 200
    data = response.json()

    assert data["start_url"] == "https://example.com/"
    assert data["pages_crawled"] == 1
    assert data["pages_fetched"] == 1
    assert data["crawl_notes"] == ["robots.txt status: missing"]

    # audit.json shape
    assert data["findings"] == [
        {
            "metric": "missing_canonical",
            "page": "https://example.com/",
            "severity": "info",
            "evidence": 'No <link rel="canonical"> element is present in the document <head>.',
            "suggested_fix": "Add a self-referencing canonical link.",
        }
    ]

    # nap_report.json shape
    assert data["nap_report"] == [
        {
            "field": "phone",
            "pages_compared": ["https://example.com/"],
            "values": ["+1 503-555-0147"],
            "normalized_values": ["5035550147"],
            "confidence": 1.0,
            "verdict": "consistent",
            "evidence": [
                {
                    "page": "https://example.com/",
                    "raw_value": "+1 503-555-0147",
                    "normalized_value": "5035550147",
                    "source": "json_ld",
                }
            ],
        }
    ]

    assert data["answer"] is None

    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["url"] == "https://example.com/"


def test_audit_includes_answer_json_shape_when_a_question_was_asked():
    with patch("app.web.run_pipeline", return_value=_canned_result(with_answer=True)):
        response = client.post(
            "/api/audit", json={"url": "https://example.com/", "question": "What are your hours?"}
        )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer == {
        "query": "What are your hours?",
        "url": "https://example.com/",
        "excerpt": "Open Monday to Friday, 9am to 5pm.",
    }
    assert "match_type" not in answer  # internal field, not part of the deliverable


def test_audit_passes_options_through_to_run_pipeline():
    with patch("app.web.run_pipeline", return_value=_canned_result()) as mock_run:
        client.post(
            "/api/audit",
            json={
                "url": "https://example.com/",
                "max_pages": 5,
                "max_depth": 1,
                "timeout": 20,
            },
        )

    kwargs = mock_run.call_args.kwargs
    assert kwargs["max_pages"] == 5
    assert kwargs["max_depth"] == 1
    assert kwargs["timeout"] == 20


def test_audit_defaults_llm_provider_to_none_and_passes_no_generate_callable():
    with patch("app.web.run_pipeline", return_value=_canned_result()) as mock_run:
        client.post("/api/audit", json={"url": "https://example.com/"})

    assert mock_run.call_args.kwargs["llm_generate"] is None


# ---------------------------------------------------------------------------
# POST /api/audit — failure handling
# ---------------------------------------------------------------------------


def test_audit_returns_500_not_a_crash_when_the_pipeline_raises():
    with patch("app.web.run_pipeline", side_effect=RuntimeError("boom")):
        response = client.post("/api/audit", json={"url": "https://example.com/"})

    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


def test_audit_never_reaches_run_pipeline_for_an_invalid_url():
    with patch("app.web.run_pipeline") as mock_run:
        client.post("/api/audit", json={"url": "not a url at all"})

    mock_run.assert_not_called()


def test_audit_forwards_question_scope_flags():
    with patch("app.web.run_pipeline", return_value=_canned_result()) as mock_run:
        client.post(
            "/api/audit",
            json={
                "url": "https://example.com/",
                "run_q1": False,
                "run_q2": True,
                "run_q3": False,
            },
        )

    kwargs = mock_run.call_args.kwargs
    assert kwargs["run_q1"] is False
    assert kwargs["run_q2"] is True
    assert kwargs["run_q3"] is False
