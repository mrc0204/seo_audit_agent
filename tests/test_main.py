"""Tests for the Phase 8 CLI/orchestration layer (``app/main.py``).

The pipeline is exercised end-to-end against an in-memory fixture site served by
``httpx.MockTransport`` — the same offline pattern as Phase 2's crawler tests — so
this file never touches the network. The plan's Phase 0 definition of done
("``python -m app.main --help`` runs without import errors") and Phase 8's
("single command runs the pipeline end-to-end and produces all three JSON files
without crashing") are both directly tested here.
"""

from __future__ import annotations

import json
import subprocess
import sys

import httpx
import pytest

from app.main import PipelineResult, main, run_pipeline, write_outputs


def _page(*hrefs: str, body: str) -> str:
    links = "".join(f'<a href="{h}">link</a>' for h in hrefs)
    return f"<html><head><title>Fixture Business</title></head><body><main><p>{body}</p>{links}</main></body></html>"


FIXTURE_SITE: dict[str, tuple[int, str, str]] = {
    "/robots.txt": (404, "text/plain", ""),
    "/": (
        200,
        "text/html",
        f"""<html>
        <head>
          <title>Fixture Coffee Roasters</title>
          <meta name="description" content="Small-batch coffee roasted fresh in Portland every week.">
          <script type="application/ld+json">
          {{"@context": "https://schema.org", "@type": "LocalBusiness",
            "name": "Fixture Coffee Roasters", "telephone": "+1 503-555-0147",
            "address": {{"@type": "PostalAddress", "streetAddress": "1 Test St",
            "addressLocality": "Portland", "addressRegion": "OR", "postalCode": "97201"}}}}
          </script>
        </head>
        <body><main>
          <h1>Fixture Coffee Roasters</h1>
          <p>We roast single-origin coffee beans fresh every single week for local cafes.</p>
          <a href="/about">About</a>
        </main></body></html>""",
    ),
    "/about": (
        200,
        "text/html",
        """<html><head><title>About Fixture Coffee Roasters</title>
        <meta name="description" content="Learn about our small-batch roasting process and our story.">
        </head><body><main>
          <h1>About us</h1>
          <p>Fixture Coffee Roasters has roasted coffee in Portland since 2015 for local cafes.</p>
        </main></body></html>""",
    ),
}


def _site_transport(site: dict[str, tuple[int, str, str]] | None = None) -> httpx.MockTransport:
    site = site or FIXTURE_SITE

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path not in site:
            return httpx.Response(404, text="not found")
        status, content_type, body = site[path]
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def test_pipeline_runs_end_to_end_and_returns_populated_result():
    result = run_pipeline("https://fixture.example/", max_pages=10, max_depth=2, transport=_site_transport())

    assert isinstance(result, PipelineResult)
    assert result.start_url == "https://fixture.example/"
    assert result.pages_fetched >= 2
    assert len(result.pages) >= 2
    assert result.findings  # the fixture has real, findable defects (no canonical, etc.)
    assert len(result.nap_comparisons) == 3
    assert result.answer is None  # no --question was given


def test_pipeline_finds_the_consistent_nap_across_the_two_pages():
    # Only the homepage carries JSON-LD NAP; /about has none. That is
    # insufficient_data by design (fewer than 2 pages contributed a value), not a
    # false consistency claim — confirms the pipeline doesn't invent agreement.
    result = run_pipeline("https://fixture.example/", max_pages=10, max_depth=2, transport=_site_transport())
    phone = next(c for c in result.nap_comparisons if c.field == "phone")
    assert phone.verdict == "insufficient_data"


def test_pipeline_answers_a_question_when_one_is_given():
    result = run_pipeline(
        "https://fixture.example/",
        question="What has Fixture Coffee Roasters done since 2015?",
        max_pages=10,
        max_depth=2,
        transport=_site_transport(),
    )
    assert result.answer is not None
    assert result.answer.query == "What has Fixture Coffee Roasters done since 2015?"
    # Grounded: whatever came back must be real, validated text or an honest null.
    assert (result.answer.url is None) == (result.answer.excerpt is None)


def test_pipeline_returns_null_answer_for_an_unanswerable_question():
    result = run_pipeline(
        "https://fixture.example/",
        question="What is your policy on international shipping insurance claims?",
        max_pages=10,
        max_depth=2,
        transport=_site_transport(),
    )
    assert result.answer is not None
    assert result.answer.url is None
    assert result.answer.excerpt is None
    assert result.answer.match_type == "none"


def test_pipeline_never_audits_a_non_html_resource_as_a_page():
    # Regression: found live in the Phase 10 sweep. A linked text/markdown file
    # (a real convention, e.g. an "agents.md" AI-agent manifest) was fetched
    # successfully (fr.ok is True — 2xx, no transport error) but is not a
    # webpage. Auditing it produced artefactual findings ("missing title",
    # "missing canonical") against a resource that was never going to have
    # either. It must be excluded from the audited page set entirely.
    site = dict(FIXTURE_SITE)
    site["/"] = (200, "text/html", _page("/agents.md", body="Homepage content about coffee roasting."))
    site["/agents.md"] = (200, "text/markdown", "# Agent instructions\n\nNo HTML here at all.")

    result = run_pipeline("https://fixture.example/", max_pages=10, max_depth=2, transport=_site_transport(site))

    audited_urls = {p.final_url for p in result.pages}
    assert "https://fixture.example/agents.md" not in audited_urls
    assert not any(f.page.endswith("/agents.md") for f in result.findings)


def test_pipeline_never_crashes_on_an_unreachable_site():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    result = run_pipeline("https://fixture.example/", transport=httpx.MockTransport(handler))
    assert result.pages_fetched >= 1  # attempted, even though nothing succeeded
    assert result.pages == []
    assert result.findings == []
    assert len(result.nap_comparisons) == 3
    assert all(c.verdict == "insufficient_data" for c in result.nap_comparisons)


def test_pipeline_rejects_a_non_crawlable_url():
    from app.crawler.url_utils import InvalidURLError

    with pytest.raises(InvalidURLError):
        run_pipeline("javascript:void(0)")


def test_pipeline_findings_and_comparisons_are_already_validated():
    # write_outputs must never need to re-validate; the pipeline promises this.
    from app.validation.nap_validator import validate_nap
    from app.validation.seo_validator import validate_seo

    result = run_pipeline("https://fixture.example/", max_pages=10, max_depth=2, transport=_site_transport())
    assert validate_seo(result.findings, result.pages) is True
    assert validate_nap(result.nap_comparisons, result.pages) is True


# ---------------------------------------------------------------------------
# write_outputs
# ---------------------------------------------------------------------------


def test_write_outputs_produces_the_briefs_exact_shapes(tmp_path):
    result = run_pipeline(
        "https://fixture.example/",
        question="What has Fixture Coffee Roasters done since 2015?",
        max_pages=10,
        max_depth=2,
        transport=_site_transport(),
    )
    written = write_outputs(result, str(tmp_path))

    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert isinstance(audit, list)
    for entry in audit:
        assert set(entry.keys()) == {"metric", "page", "severity", "evidence", "suggested_fix"}

    nap_report = json.loads((tmp_path / "nap_report.json").read_text(encoding="utf-8"))
    assert isinstance(nap_report, list) and len(nap_report) == 3
    for entry in nap_report:
        assert set(entry.keys()) == {
            "field",
            "pages_compared",
            "values",
            "normalized_values",
            "confidence",
            "verdict",
        }

    answer = json.loads((tmp_path / "answer.json").read_text(encoding="utf-8"))
    assert set(answer.keys()) == {"query", "url", "excerpt"}

    assert written["audit"].endswith("audit.json")
    assert written["nap_report"].endswith("nap_report.json")
    assert written["answer"].endswith("answer.json")


def test_write_outputs_skips_answer_file_when_no_question_was_asked(tmp_path):
    result = run_pipeline("https://fixture.example/", max_pages=10, max_depth=2, transport=_site_transport())
    written = write_outputs(result, str(tmp_path))

    assert written["answer"] is None
    assert not (tmp_path / "answer.json").exists()
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "nap_report.json").exists()


def test_write_outputs_creates_the_directory_if_missing(tmp_path):
    result = run_pipeline("https://fixture.example/", max_pages=10, max_depth=2, transport=_site_transport())
    target = tmp_path / "nested" / "outputs"
    write_outputs(result, str(target))
    assert (target / "audit.json").exists()


def test_write_outputs_preserves_unicode_without_escaping(tmp_path):
    site = dict(FIXTURE_SITE)
    site["/"] = (
        200,
        "text/html",
        '<html><head><title>Café Roasters — Fixture</title></head>'
        '<body><main><p>We roast coffee for the café down the road, "unicode" tested here.</p></main></body></html>',
    )
    result = run_pipeline("https://fixture.example/", max_pages=5, max_depth=1, transport=_site_transport(site))
    write_outputs(result, str(tmp_path))

    raw = (tmp_path / "audit.json").read_text(encoding="utf-8")
    # ensure_ascii=False: a real accented character, not a \uXXXX escape.
    assert "\\u" not in raw


# ---------------------------------------------------------------------------
# build_llm_generate
# ---------------------------------------------------------------------------


def test_build_llm_generate_returns_none_when_no_provider_selected():
    from app.main import build_llm_generate

    assert build_llm_generate(None, None) is None
    assert build_llm_generate("none", None) is None


def test_build_llm_generate_wraps_a_valid_provider_lazily():
    # GroqProvider/GeminiProvider construct fine without ever touching the network
    # — build_llm_generate does not call .generate() eagerly, so selecting "groq"
    # succeeds here and nothing is attempted until something actually calls it.
    from app.main import build_llm_generate

    generate = build_llm_generate("groq", "fake-key")
    assert generate is not None
    assert callable(generate)


def test_pipeline_tolerates_a_misconfigured_llm_provider_end_to_end():
    # The consumer side of the provider contract: every call site (apply_llm_
    # suggested_fixes, answer_question) already swallows a raising generate() and
    # falls back to deterministic output. api_key=None makes GroqProvider raise
    # ProviderConfigError immediately, with no network call, which is both the
    # realistic "user forgot to set a key" case and keeps this test offline.
    from app.main import build_llm_generate

    generate = build_llm_generate("groq", None)
    result = run_pipeline(
        "https://fixture.example/",
        question="What has Fixture Coffee Roasters done since 2015?",
        max_pages=10,
        max_depth=2,
        transport=_site_transport(),
        llm_generate=generate,
    )
    assert result.findings  # still produced, with the deterministic suggested_fix
    assert result.answer is not None


def test_build_llm_generate_rejects_an_unknown_provider_name(capsys):
    from app.main import build_llm_generate

    assert build_llm_generate("not-a-real-provider", None) is None
    assert "warning" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# main() — argument handling and exit codes
# ---------------------------------------------------------------------------


def test_main_rejects_a_malformed_url(capsys):
    code = main(["--url", "javascript:void(0)"])
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()


def test_main_help_runs_without_import_errors():
    # The plan's Phase 0 definition of done, run for real as a subprocess so it
    # also exercises "python -m app.main --help" exactly as a user would.
    proc = subprocess.run(
        [sys.executable, "-m", "app.main", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--url" in proc.stdout
