"""Command-line interface entry point for Evidence-Grounded SEO Audit Agent.

Wiring only: crawl once, run Q1/Q2/(optionally Q3), validate everything against the
crawl snapshot before it is allowed to reach disk, then write the three JSON
deliverables the brief specifies. A plain sequential script, per the plan's own
caution against reaching for an orchestration framework before one is needed.

:func:`run_pipeline` is the whole thing minus argument parsing and file I/O, kept
separate so it is callable — and testable against a mocked transport — without
going through the CLI or touching disk.

LLM enrichment (rewriting ``suggested_fix``, and answering ``--question`` with a
model instead of the deterministic top-chunk fallback) is opt-in and best-effort:
Phase 9 owns the actual provider implementations, so this module only wires the
*selection* (``--llm-provider``, ``LLM_PROVIDER`` in ``.env``) and degrades to no
LLM — never a crash — if the selected provider is unavailable or unimplemented.
Every check that matters (audit findings, NAP verdicts, the substring gate) is
already fully deterministic and works with no LLM configured at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.nap_agent import run_nap_check
from app.agents.qa_agent import answer_question
from app.agents.seo_agent import apply_llm_suggested_fixes, run_seo_audit
from app.crawler.crawler import HTML_CONTENT_TYPES, Crawler
from app.crawler.url_utils import InvalidURLError, normalize_url
from app.extraction.seo_extractor import build_page_data
from app.models.answer import QAAnswer
from app.models.finding import Finding
from app.models.nap import NAPComparison
from app.validation.nap_validator import filter_valid_comparisons
from app.validation.qa_validator import filter_valid_answer
from app.validation.seo_validator import filter_valid_findings

#: Output filenames, fixed by the brief.
AUDIT_FILENAME = "audit.json"
NAP_REPORT_FILENAME = "nap_report.json"
ANSWER_FILENAME = "answer.json"

#: Env vars read as CLI-flag defaults, per .env.example. A flag explicitly passed
#: on the command line always wins over its env default.
_ENV_DEFAULTS = {
    "max_pages": ("MAX_PAGES", int, 100),
    "max_depth": ("MAX_DEPTH", int, 3),
    "timeout": ("TIMEOUT_SECONDS", int, 10),
}


@dataclass
class PipelineResult:
    """Everything one run produced, before it is written to disk.

    Attributes:
        start_url: The normalized seed URL that was actually crawled.
        pages: ``PageData`` objects for every page the crawl fetched successfully.
        findings: Validated Q1 findings, ready for :meth:`Finding.to_audit_entry`.
        nap_comparisons: Validated Q2 comparisons, one per field, ready for
            :meth:`NAPComparison.to_report_entry`.
        answer: The Q3 result, or ``None`` when ``--question`` was not given —
            ``QAAnswer.query`` is a required field, so there is no honest "empty"
            answer to produce without a question to attach it to.
        crawl_notes: Warnings surfaced from the crawl (robots status, JS-shell
            pages, skip reasons) — not written to any output file, but printed by
            the CLI so a run explains itself.
        pages_fetched: Total fetch attempts, including failed/non-HTML ones, for
            an honest count even when ``pages`` (successes only) is smaller.
    """

    start_url: str
    pages: list[Any] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    nap_comparisons: list[NAPComparison] = field(default_factory=list)
    answer: QAAnswer | None = None
    crawl_notes: list[str] = field(default_factory=list)
    pages_fetched: int = 0


def run_pipeline(
    url: str,
    question: str | None = None,
    max_pages: int = 100,
    max_depth: int = 3,
    timeout: int = 10,
    respect_robots: bool = True,
    use_sitemap: bool = True,
    llm_generate: Callable[[str], str] | None = None,
    transport: Any = None,
) -> PipelineResult:
    """Run the full crawl -> extract -> Q1/Q2/Q3 -> validate pipeline once.

    Args:
        url: Target site. Validated as a crawlable http(s) URL before anything
            else runs.
        question: Optional natural-language question for Q3. ``None`` skips Q3
            entirely — no ``answer.json`` is produced.
        max_pages: Crawl page budget.
        max_depth: Crawl depth budget.
        timeout: Per-request timeout in seconds.
        respect_robots: Passed through to the crawler.
        use_sitemap: Passed through to the crawler.
        llm_generate: Optional callable used for both Q1's ``suggested_fix``
            rewriting and Q3's answer selection. ``None`` (the default) runs the
            pipeline fully deterministically — no field this affects is required
            for a correct, evidence-backed result.
        transport: Optional ``httpx`` transport, for tests to serve a fixture site
            instead of the real network. Threaded straight through to
            :meth:`app.crawler.crawler.Crawler.crawl`.

    Returns:
        A :class:`PipelineResult`. Findings and NAP comparisons have already been
        through :mod:`app.validation` — nothing unvalidated reaches the caller.

    Raises:
        InvalidURLError: If ``url`` is not a crawlable http(s) URL. Everything
            after this point is expected to degrade gracefully (an unreachable
            site simply crawls zero pages) rather than raise.
    """
    seed = normalize_url(url)

    crawler = Crawler(respect_robots=respect_robots, use_sitemap=use_sitemap)
    fetch_results = crawler.crawl(
        seed, max_pages=max_pages, max_depth=max_depth, timeout=timeout, transport=transport
    )
    report = crawler.last_report

    # ``fr.ok`` only means "2xx and no transport error" — it says nothing about
    # content type. A non-HTML resource that happened to 200 (a markdown file, a
    # PDF) still passes it, but the crawler already zeroed its `html` and left its
    # real `content_type`. Auditing that as a webpage would produce SEO findings
    # (missing title, missing canonical...) against a resource that was never a
    # page — an entirely artefactual "defect" the evidence does not support. Found
    # live during the Phase 10 sweep: a Shopify store's /agents.md (text/markdown)
    # was reported as missing a <title> and canonical it was never going to have.
    pages = [
        build_page_data(fr.url, fr.final_url, fr.status_code, fr.html, x_robots_tag=fr.x_robots_tag)
        for fr in fetch_results
        if fr.ok and _is_html_fetch(fr)
    ]

    findings = run_seo_audit(pages, fetch_results=fetch_results)
    findings = apply_llm_suggested_fixes(findings, llm_generate)
    findings = filter_valid_findings(findings, pages)

    nap_comparisons = run_nap_check(pages)
    nap_comparisons = filter_valid_comparisons(nap_comparisons, pages)

    answer: QAAnswer | None = None
    if question:
        answer = answer_question(question, pages, generate=llm_generate)
        answer = filter_valid_answer(answer, pages)

    crawl_notes = list(report.notes) if report else []
    if report and report.robots_status not in ("ok", "missing"):
        crawl_notes.append(f"robots.txt status: {report.robots_status}")
    if report and report.skipped:
        by_reason: dict[str, int] = {}
        for reason in report.skipped.values():
            by_reason[reason] = by_reason.get(reason, 0) + 1
        crawl_notes.append(
            "skipped URLs: " + ", ".join(f"{count} {reason}" for reason, count in sorted(by_reason.items()))
        )

    return PipelineResult(
        start_url=seed,
        pages=pages,
        findings=findings,
        nap_comparisons=nap_comparisons,
        answer=answer,
        crawl_notes=crawl_notes,
        pages_fetched=len(fetch_results),
    )


def write_outputs(result: PipelineResult, output_dir: str) -> dict[str, str]:
    """Serialize a :class:`PipelineResult` to the brief's JSON deliverables.

    Args:
        result: A completed pipeline run.
        output_dir: Directory to write into; created if it does not exist.

    Returns:
        ``{"audit": path, "nap_report": path, "answer": path | None}`` — the paths
        actually written. ``answer`` is ``None`` when no question was asked, since
        there is nothing honest to put in that file.
    """
    os.makedirs(output_dir, exist_ok=True)
    written: dict[str, str] = {}

    audit_path = os.path.join(output_dir, AUDIT_FILENAME)
    _write_json(audit_path, [f.to_audit_entry() for f in result.findings])
    written["audit"] = audit_path

    nap_path = os.path.join(output_dir, NAP_REPORT_FILENAME)
    _write_json(nap_path, [c.to_report_entry() for c in result.nap_comparisons])
    written["nap_report"] = nap_path

    if result.answer is not None:
        answer_path = os.path.join(output_dir, ANSWER_FILENAME)
        _write_json(answer_path, result.answer.to_answer_json())
        written["answer"] = answer_path
    else:
        written["answer"] = None

    return written


def _is_html_fetch(fetch_result: Any) -> bool:
    """True when a fetched resource should be treated as an auditable HTML page.

    Matches the crawler's own definition of "HTML" (:data:`HTML_CONTENT_TYPES`) so
    the two never disagree: an unlabelled ``Content-Type`` is treated as HTML
    (optimistic default — a server that omits the header should not lose its
    pages from the audit), but a resource explicitly labelled as something else
    (``text/markdown``, ``application/pdf``, ...) is excluded, regardless of its
    status code.

    Args:
        fetch_result: A ``FetchResult``.

    Returns:
        Whether this fetch should become a ``PageData`` and be audited.
    """
    content_type = (getattr(fetch_result, "content_type", None) or "").lower()
    return not content_type or content_type.startswith(HTML_CONTENT_TYPES)


def _write_json(path: str, payload: Any) -> None:
    """Write JSON with real Unicode characters kept literal, not \\u-escaped.

    ``ensure_ascii=False`` matters here specifically: extracted page text is
    already verified to preserve exact characters (Phase 3's substring invariant
    depends on it), and escaping them on the way out would make a human review of
    the output harder for no benefit.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_llm_generate(provider_name: str | None, api_key: str | None) -> Callable[[str], str] | None:
    """Build a ``generate(prompt) -> str`` callable for the named provider, if possible.

    Best-effort and silent about failure by design: every LLM-touched code path in
    this pipeline (Phase 4's ``suggested_fix``, Phase 6's answer selection) already
    treats ``generate=None`` and a raising ``generate`` identically — a fully
    deterministic, evidence-backed result either way. So a misconfigured or
    not-yet-implemented provider degrades the pipeline's polish, never its
    correctness, and is worth a printed warning rather than a hard failure.

    Args:
        provider_name: ``"groq"``, ``"gemini"``, or ``None``/``"none"`` to disable.
        api_key: The provider's API key, if required.

    Returns:
        A callable, or ``None`` when no provider was selected or it could not be
        constructed.
    """
    name = (provider_name or "none").strip().lower()
    if name in ("", "none"):
        return None

    from app.llm.provider import GeminiProvider, GroqProvider

    provider_classes = {"groq": GroqProvider, "gemini": GeminiProvider}
    if name not in provider_classes:
        print(f"warning: unknown LLM provider {name!r}; continuing without LLM.", file=sys.stderr)
        return None

    try:
        # api_key is passed at construction, not per-call: each provider also
        # falls back to its own env var (GROQ_API_KEY / GEMINI_API_KEY) when this
        # is None, so LLM_API_KEY in .env.example is honored either way.
        provider = provider_classes[name](api_key=api_key)
    except Exception as exc:
        print(f"warning: could not initialize {name} provider ({exc}); continuing without LLM.", file=sys.stderr)
        return None

    return provider.generate


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser, with env-var defaults applied per ``.env.example``."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    defaults = {
        name: env_type(os.environ[env_name]) if os.environ.get(env_name) else fallback
        for name, (env_name, env_type, fallback) in _ENV_DEFAULTS.items()
    }

    parser = argparse.ArgumentParser(description="Evidence-Grounded SEO Audit Agent CLI")
    parser.add_argument("--url", type=str, required=True, help="Target website URL to crawl and audit")
    parser.add_argument(
        "--question", type=str, default=None, help="Optional question for evidence-grounded QA agent"
    )
    parser.add_argument(
        "--max-pages", type=int, default=defaults["max_pages"], help="Maximum number of pages to crawl"
    )
    parser.add_argument("--max-depth", type=int, default=defaults["max_depth"], help="Maximum crawl depth")
    parser.add_argument(
        "--timeout", type=int, default=defaults["timeout"], help="HTTP request timeout in seconds"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory path to write JSON outputs"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=os.environ.get("LLM_PROVIDER", "none"),
        choices=["none", "groq", "gemini"],
        help="Optional LLM provider for suggested-fix phrasing and Q&A selection",
    )
    parser.add_argument(
        "--no-robots",
        action="store_false",
        dest="respect_robots",
        default=True,
        help="Ignore robots.txt (not recommended)",
    )
    parser.add_argument(
        "--no-sitemap",
        action="store_false",
        dest="use_sitemap",
        default=True,
        help="Do not seed the crawl frontier from the site's sitemap",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint parsing flags and executing SEO audit pipeline.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``. Accepting an
            explicit list keeps this testable without spawning a subprocess.

    Returns:
        Process exit code: ``0`` on a completed run (even one that crawled zero
        pages — that is a real, reportable outcome, not a crash), ``2`` for a
        malformed ``--url``.
    """
    args = _build_arg_parser().parse_args(argv)

    try:
        normalize_url(args.url)
    except InvalidURLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    llm_generate = build_llm_generate(args.llm_provider, os.environ.get("LLM_API_KEY"))

    result = run_pipeline(
        url=args.url,
        question=args.question,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        timeout=args.timeout,
        respect_robots=args.respect_robots,
        use_sitemap=args.use_sitemap,
        llm_generate=llm_generate,
    )

    if result.pages_fetched == 0:
        print(f"warning: crawled zero pages from {result.start_url}.", file=sys.stderr)

    for note in result.crawl_notes:
        print(f"note: {note}", file=sys.stderr)

    written = write_outputs(result, args.output_dir)

    severity_counts: dict[str, int] = {}
    for finding in result.findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    print(f"Crawled {len(result.pages)} page(s) of {result.pages_fetched} fetched from {result.start_url}")
    print(f"Q1 findings: {len(result.findings)} " + ", ".join(f"{v} {k}" for k, v in sorted(severity_counts.items())))
    print(
        "Q2 NAP: "
        + ", ".join(f"{c.field}={c.verdict}" for c in result.nap_comparisons)
    )
    if args.question:
        if result.answer and result.answer.url:
            print(f"Q3 answer: found on {result.answer.url}")
        else:
            print("Q3 answer: none (question could not be grounded in the site)")
    print(f"Wrote: {written['audit']}, {written['nap_report']}" + (f", {written['answer']}" if written["answer"] else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
