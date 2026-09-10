"""Phase 10 cross-site generalization sweep.

Runs the full pipeline (crawl -> extract -> Q1 -> Q2 -> Q3) against 5 real sites
never touched by development or by the automated test suite, each on a different
CMS/stack, per EXECUTION_PLAN.md Phase 10:

    site               | stack
    -------------------|----------------------------------------------
    wptavern.com        | WordPress
    allbirds.com         | Shopify
    webflow.com           | Webflow
    motherfuckingwebsite.com | plain static HTML, zero JS
    excalidraw.com         | JS-heavy client-rendered SPA

Deliberately calls :func:`app.main.run_pipeline` — the same function
``python -m app.main`` calls — rather than re-assembling crawl -> extract -> audit
by hand. An earlier version of this script did exactly that, which meant a Phase 8
bug fix (excluding non-HTML fetches, e.g. a linked ``agents.md``, from the audited
page set) silently didn't apply here even after being fixed in ``app/main.py``: the
sweep re-tested its own parallel, stale copy of the pipeline instead of the real
one. Going through ``run_pipeline`` is what makes this sweep actually mean
something about the shipped CLI.

For each site, logs: crawl success rate and pages fetched, every Q1/Q2 finding (for
hand spot-checking against the real page — this script does not grade its own
homework), and 3 answerable + 2 unanswerable Q3 questions with their grounding
verified programmatically (a non-null answer's excerpt is asserted to be a literal
substring of the cited page's own text; every result is also printed for a human to
read).

Not a pytest suite — this makes real network calls to real third-party sites and is
meant to be run manually, per the plan's own instruction that this sweep is run
("Antigravity runs the sweep") rather than executed on every CI run. Output is both
printed and saved as JSON next to this file for the record.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agents.qa_agent import answer_question
from app.main import run_pipeline
from app.validation.nap_validator import validate_nap
from app.validation.qa_validator import filter_valid_answer, validate_qa
from app.validation.seo_validator import validate_seo

SITES: list[dict] = [
    {
        "url": "https://wptavern.com/",
        "stack": "WordPress",
        "answerable_questions": [
            "What is WordPress Tavern?",
            "Does the site cover WordPress plugins?",
            "Does the site cover WordPress themes?",
        ],
        "unanswerable_questions": [
            "What is the refund policy for a purchase on this site?",
            "What programming language is the Linux kernel written in?",
        ],
    },
    {
        "url": "https://www.allbirds.com/",
        "stack": "Shopify",
        "answerable_questions": [
            "What materials does Allbirds use in its shoes?",
            "Does Allbirds sell wool shoes?",
            "Does Allbirds ship internationally?",
        ],
        "unanswerable_questions": [
            "What is the population of Japan?",
            "Who won the most recent World Cup?",
        ],
    },
    {
        "url": "https://webflow.com/",
        "stack": "Webflow",
        "answerable_questions": [
            "What is Webflow?",
            "Can Webflow be used to build websites without code?",
            "Does Webflow offer a CMS?",
        ],
        "unanswerable_questions": [
            "What is the boiling point of water in Fahrenheit?",
            "What is the capital of Australia?",
        ],
    },
    {
        "url": "https://motherfuckingwebsite.com/",
        "stack": "Static HTML (no CSS/JS)",
        "answerable_questions": [
            "What does this website say about fonts?",
            "What does this website say about JavaScript?",
        ],
        "unanswerable_questions": [
            "What is the site's shipping policy?",
            "What programming language was used to build the site's backend API?",
            "How many employees does the company have?",
        ],
    },
    {
        "url": "https://excalidraw.com/",
        "stack": "JS-heavy SPA (client-rendered canvas app)",
        "answerable_questions": [
            "What is Excalidraw?",
        ],
        "unanswerable_questions": [
            "What is the site's return policy?",
            "What is the population of Canada?",
            "What is the site's phone number?",
        ],
    },
]

MAX_PAGES = 6
MAX_DEPTH = 1
TIMEOUT = 20


def sweep_one(site: dict) -> dict:
    url, stack = site["url"], site["stack"]
    print(f"\n{'=' * 70}\n{stack}: {url}\n{'=' * 70}")

    log: dict = {"url": url, "stack": stack, "timestamp": datetime.now(timezone.utc).isoformat()}

    t0 = time.time()
    try:
        result = run_pipeline(url, max_pages=MAX_PAGES, max_depth=MAX_DEPTH, timeout=TIMEOUT)
    except Exception as exc:
        log["crawl_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  CRAWL FAILED: {log['crawl_error']}")
        return log
    elapsed = time.time() - t0

    log["pages_fetched"] = result.pages_fetched
    log["pages_ok"] = len(result.pages)
    log["crawl_seconds"] = round(elapsed, 1)
    log["notes"] = result.crawl_notes

    print(f"  crawl: {len(result.pages)}/{result.pages_fetched} ok, {elapsed:.1f}s")
    for note in log["notes"]:
        print(f"  note: {note}")

    pages = result.pages
    if not pages:
        log["findings"] = []
        log["nap"] = []
        log["qa"] = []
        print("  no pages successfully fetched — skipping Q1/Q2/Q3")
        return log

    # --- Q1 (already validated by run_pipeline; re-confirmed here as a guard) ---
    findings = result.findings
    assert validate_seo(findings, pages), "a Q1 finding failed independent re-validation"
    log["findings"] = [f.model_dump(mode="json") for f in findings]
    sev_counts: dict[str, int] = {}
    for f in findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
    print(f"  Q1: {len(findings)} findings {sev_counts}")
    for f in findings[:5]:
        print(f"    [{f.severity}] {f.check_id} {f.metric} :: {f.evidence[:90]}")
    if len(findings) > 5:
        print(f"    ... and {len(findings) - 5} more (full list in the saved JSON)")

    # --- Q2 ---
    comparisons = result.nap_comparisons
    assert validate_nap(comparisons, pages), "a Q2 comparison failed independent re-validation"
    log["nap"] = [c.model_dump(mode="json") for c in comparisons]
    for c in comparisons:
        print(f"  Q2: {c.field:8} {c.verdict:18} confidence={c.confidence} evidence={len(c.evidence)}")

    # --- Q3 ---
    # One crawl per site (above) is reused for every question here via
    # answer_question() directly, rather than calling run_pipeline per question —
    # that would re-crawl the live site once per question, which is both slow and
    # impolite. answer_question + filter_valid_answer is exactly what run_pipeline
    # does internally for its single --question argument, so this still exercises
    # the real Q3 code path against the real pages run_pipeline produced.
    qa_results = []
    for q in site["answerable_questions"]:
        answer = filter_valid_answer(answer_question(q, pages), pages)
        assert validate_qa(answer, pages), f"an accepted answer failed independent re-validation: {q!r}"
        qa_results.append({"query": q, "expected": "answerable", **answer.model_dump()})
        status = "ANSWERED" if answer.url else "NULL"
        print(f"  Q3 [{status:8}] (expected answerable)   {q}")
        if answer.excerpt:
            print(f"      -> {answer.excerpt[:100]!r}")

    for q in site["unanswerable_questions"]:
        answer = filter_valid_answer(answer_question(q, pages), pages)
        assert validate_qa(answer, pages), f"an accepted answer failed independent re-validation: {q!r}"
        qa_results.append({"query": q, "expected": "unanswerable", **answer.model_dump()})
        status = "ANSWERED" if answer.url else "NULL"
        flag = "" if answer.url is None else "  <-- unexpected non-null, inspect"
        print(f"  Q3 [{status:8}] (expected unanswerable) {q}{flag}")
        if answer.excerpt:
            print(f"      -> {answer.excerpt[:100]!r}")

    log["qa"] = qa_results
    return log


def main() -> None:
    results = [sweep_one(site) for site in SITES]

    out_path = Path(__file__).with_name("cross_site_sweep_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for r in results:
        if "crawl_error" in r:
            print(f"  {r['stack']:35} CRASHED: {r['crawl_error']}")
            continue
        n_findings = len(r.get("findings", []))
        n_answered = sum(1 for qa in r.get("qa", []) if qa.get("url"))
        n_expected_answerable = sum(1 for qa in r.get("qa", []) if qa["expected"] == "answerable")
        n_false_positive_unanswerable = sum(
            1 for qa in r.get("qa", []) if qa["expected"] == "unanswerable" and qa.get("url")
        )
        print(
            f"  {r['stack']:35} pages={r.get('pages_ok', 0)}/{r.get('pages_fetched', 0)} "
            f"findings={n_findings} answered={n_answered}/{n_expected_answerable} "
            f"unexpected_answers={n_false_positive_unanswerable}"
        )

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
