"""Minimal web UI for the Evidence-Grounded SEO Audit Agent.

A thin HTTP layer over :mod:`app.main` — this file contains no audit logic of its
own. It exists only to let a browser trigger :func:`app.main.run_pipeline` and
render its result, instead of going through the CLI. Every correctness guarantee
documented in ``app/main.py`` and the modules it calls (the LLM boundary, the
validation gates, the literal-substring check) applies exactly as-is here; this
module cannot weaken or bypass any of them, because it never touches a `Finding`,
`NAPComparison`, or `QAAnswer` directly — it only serializes what `run_pipeline`
already validated.

Run it with:

    uvicorn app.web:app --reload

Then open http://127.0.0.1:8000/ in a browser.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.crawler.url_utils import InvalidURLError, normalize_url
from app.main import build_llm_generate, run_pipeline

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Evidence-Grounded SEO Audit Agent")


class AuditRequest(BaseModel):
    """The form fields the frontend submits, one-to-one with the CLI flags."""

    url: str
    question: str | None = None
    max_pages: int = Field(default=20, ge=1, le=200)
    max_depth: int = Field(default=2, ge=0, le=5)
    timeout: int = Field(default=15, ge=1, le=60)
    llm_provider: str = "none"


class AuditResponse(BaseModel):
    """Exactly the three deliverable shapes from the brief, plus a run summary.

    ``findings``/``nap_report``/``answer`` are produced by the same
    ``to_audit_entry`` / ``to_report_entry`` / ``to_answer_json`` serializers the
    CLI writes to disk — the UI is shown nothing the CLI doesn't also produce.
    """

    start_url: str
    pages_crawled: int
    pages_fetched: int
    crawl_notes: list[str]
    findings: list[dict[str, Any]]
    nap_report: list[dict[str, Any]]
    answer: dict[str, Any] | None


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/api/audit", response_model=AuditResponse)
async def audit(request: AuditRequest) -> AuditResponse:
    """Run the full pipeline for one URL (and optional question) and return its result.

    The pipeline is synchronous and can take anywhere from a couple of seconds to
    a minute depending on the site and ``max_pages`` — it runs in a thread pool so
    it never blocks the server's event loop while a crawl is in flight.

    Raises:
        HTTPException: 422 if ``url`` is not a crawlable http(s) address (caught
            before any crawling starts); 500 for anything unexpected during the
            crawl itself, so a single bad site cannot take the server down.
    """
    try:
        normalize_url(request.url)
    except InvalidURLError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    llm_generate = build_llm_generate(request.llm_provider, os.environ.get("LLM_API_KEY"))

    try:
        result = await run_in_threadpool(
            run_pipeline,
            url=request.url,
            question=request.question or None,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            timeout=request.timeout,
            llm_generate=llm_generate,
        )
    except InvalidURLError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a crawl failure must not crash the server
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return AuditResponse(
        start_url=result.start_url,
        pages_crawled=len(result.pages),
        pages_fetched=result.pages_fetched,
        crawl_notes=result.crawl_notes,
        findings=[f.to_audit_entry() for f in result.findings],
        nap_report=[c.to_report_entry() for c in result.nap_comparisons],
        answer=result.answer.to_answer_json() if result.answer is not None else None,
    )
