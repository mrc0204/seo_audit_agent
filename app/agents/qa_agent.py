"""Evidence-grounded Question Answering agent.

Anti-hallucination guarantee lives here, and it is enforced structurally, not by
prompting alone. The pipeline is three deliberately separate stages:

1. **Broad lexical retrieval** (:func:`app.retrieval.retriever.retrieve_top_k`) —
   BM25's top :data:`DEFAULT_TOP_K` chunks, ranking only, no relevance judgment.
   Retrieval's only job is recall: don't let the real answer get excluded before
   anything gets a chance to judge it.
2. **Answerability judgment** — deciding whether any candidate *actually answers*
   the question, as opposed to merely sharing its vocabulary, is never BM25's
   job. It happens at exactly one of two places, matched to how much judgment is
   actually available:

   * **With an LLM configured** (:func:`_llm_answer`), the model *is* the judge —
     :data:`DEFAULT_QA_PROMPT` explicitly instructs it to decide answerability
     first, before it may select and copy a span. "Do you offer wool shoes?" and
     a paragraph that merely mentions "shoes" and "wool" in an unrelated sentence
     share enough vocabulary to both retrieve, and only a judgment step — lexical
     coverage cannot make this call — tells them apart.
   * **Without one** (:func:`_offline_answer`), there is no semantic judge
     available at all, so the deterministic fallback compensates with its own,
     much stricter coverage threshold (:data:`OFFLINE_ANSWER_MIN_COVERAGE`)
     before it will call anything "answered" — a safeguard specific to the
     no-LLM path, unrelated to how retrieval itself ranks candidates.

3. **Exact source validation** (:func:`app.validation.qa_validator.find_source_span`)
   — whatever excerpt either path lands on is independently re-checked against
   the actual page text, unconditionally. Never write new sentences, never
   mistake "topically related" for "answers it", and an answer that fails this
   check is discarded and the result is null — this module never "fixes" a
   near-miss excerpt, and a model exception or an unparseable response also
   degrades to null rather than a guess.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from app.models.answer import QAAnswer
from app.retrieval.index import build_index, tokenize
from app.retrieval.retriever import DEFAULT_TOP_K, RetrievedChunk, retrieve_top_k
from app.validation.qa_validator import find_source_span

#: Longest excerpt accepted, even after passing the substring check. An "excerpt"
#: spanning several paragraphs is not what the brief means by "the exact passage".
MAX_EXCERPT_CHARS = 600

#: The offline (no-LLM) fallback's bar for treating a retrieved chunk as an actual
#: answer, not merely a topically related passage — the fraction of the query's
#: distinct, non-stopword terms that must appear in the chunk. This is unrelated
#: to retrieval's own ranking (which applies no coverage floor at all — see
#: :mod:`app.retrieval.retriever`); it exists here, specifically, because the
#: offline path has no semantic judge to lean on the way the LLM path does, so it
#: is set high enough that most of the question's actual vocabulary has to be
#: present, not just some of it.
OFFLINE_ANSWER_MIN_COVERAGE = 0.75


def answer_question(
    query: str,
    pages: list[Any],
    generate: Callable[[str], str] | None = None,
    k: int = DEFAULT_TOP_K,
    prompt_template: str | None = None,
) -> QAAnswer:
    """Answer user questions grounded strictly in retrieved context from crawled pages.

    Args:
        query: The natural-language question.
        pages: ``PageData`` objects from Phase 3.
        generate: Callable taking a prompt and returning text. ``None`` uses the
            deterministic top-chunk fallback described in the module docstring.
        k: How many candidate chunks to retrieve and (if an LLM is configured)
            show it.
        prompt_template: Overrides :data:`DEFAULT_QA_PROMPT`.

    Returns:
        A :class:`QAAnswer`. ``url``/``excerpt`` are null with
        ``match_type="none"`` whenever nothing could be grounded — no candidates
        retrieved, the LLM found nothing, its response could not be parsed, or its
        claimed excerpt failed the literal-substring check.
    """
    index = build_index(pages)
    candidates = retrieve_top_k(query, index, k=k)

    if not candidates:
        return _null_answer(query)

    if generate is None:
        return _offline_answer(query, candidates)

    return _llm_answer(query, candidates, pages, generate, prompt_template)


def _offline_answer(query: str, candidates: list[RetrievedChunk]) -> QAAnswer:
    """The no-LLM fallback: the best-covering candidate's text, if it clears a real answerability bar.

    Deliberately re-ranks ``candidates`` by query-term coverage rather than trusting
    retrieval's BM25 order, and picks the best-covering one before applying the
    answerability threshold. BM25 score and "how completely this specific chunk
    covers this specific question's vocabulary" are not the same measure — a short,
    dense chunk (a nav label like "Download & Installation") can out-score a longer
    chunk that actually states the full answer ("...available for download for all
    versions"), simply because BM25 rewards term rarity and length-normalized term
    frequency, not recall of the query's own words. Picking the top BM25 hit and
    only *then* checking whether it clears the coverage bar — this function's first
    implementation — meant a worse-but-higher-scoring chunk could shadow a better
    one that was sitting right below it in the same candidate list. Found live
    against python.org: "How do I download Python?" failed entirely under that
    ordering despite the correct answer being the second-ranked candidate.

    Retrieval (:func:`app.retrieval.retriever.retrieve_top_k`) has already filtered
    for topical relevance before any of ``candidates`` reach here; this step filters
    again, much more strictly, for whether the *best available* one is strong enough
    to call an actual answer with no semantic judge to confirm it. See
    :data:`OFFLINE_ANSWER_MIN_COVERAGE`.
    """
    if not candidates:
        return _null_answer(query)

    best = max(candidates, key=lambda c: _query_coverage(query, c.text))

    if _query_coverage(query, best.text) < OFFLINE_ANSWER_MIN_COVERAGE:
        return _null_answer(query)

    excerpt = best.text[:MAX_EXCERPT_CHARS]
    return QAAnswer(query=query, url=best.url, excerpt=excerpt, match_type="exact_substring")


def _query_coverage(query: str, text: str) -> float:
    """Fraction of the query's distinct, non-stopword terms present in ``text``."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    return len(query_terms & set(tokenize(text))) / len(query_terms)


def _llm_answer(
    query: str,
    candidates: list[RetrievedChunk],
    pages: list[Any],
    generate: Callable[[str], str],
    prompt_template: str | None,
) -> QAAnswer:
    """Ask the LLM to select an excerpt, then independently verify it before accepting."""
    template = prompt_template or DEFAULT_QA_PROMPT

    try:
        # Both kwargs below address the same underlying issue, found live against
        # a reasoning model (Groq's openai/gpt-oss-20b): it writes an internal
        # "reasoning" pass before the final answer, drawing from the SAME
        # max_tokens budget as the answer itself. Judging up to k=15 passages
        # gives it enough to reason about that it can spiral ("But... but...
        # but...") and burn the ENTIRE budget without ever concluding --
        # reproduced directly at max_tokens=1600 alone: finish_reason="length",
        # ~1600 reasoning tokens spent, zero content. Just raising max_tokens
        # further is not reliable, since a looping model keeps expanding to fill
        # whatever room it is given. reasoning_effort="low" is the fix that
        # actually converges: identical prompt, ~39 reasoning tokens, a correct
        # verified answer, reliably across 5 repeated live calls. max_tokens is
        # still raised well above a provider's bare default (512) as a cheap
        # extra margin; both kwargs are ignored by providers/models that don't
        # recognize them (see GroqProvider.generate's docstring).
        # temperature=0: this call is a YES/NO answerability judgment plus a
        # verbatim copy, not creative writing, so there is nothing for sampling
        # randomness to usefully add. Found live: on a genuinely borderline real
        # passage (mentions "course" and "offers" but never says the business
        # name itself), the default temperature=0.2 flipped between a valid
        # answer and NONE across identical repeated calls -- true model
        # non-determinism, not a bug in this pipeline, but temperature=0
        # minimizes it since there is no reason to keep any of it here.
        raw = generate(
            template.format(query=query, candidates=_format_candidates(candidates)),
            max_tokens=1600,
            reasoning_effort="low",
            temperature=0,
        )
    except Exception:
        # A failing LLM must produce null, never a guess.
        return _null_answer(query)

    parsed = _parse_response(raw)
    if parsed is None:
        return _null_answer(query)

    url, excerpt = parsed
    excerpt = excerpt[:MAX_EXCERPT_CHARS]

    # The independent check: the model's own claim is never trusted on its say-so.
    if find_source_span(url, excerpt, pages) is None:
        return _null_answer(query)

    return QAAnswer(query=query, url=url, excerpt=excerpt, match_type="exact_substring")


def _null_answer(query: str) -> QAAnswer:
    """The explicit, honest "nothing qualifies" result."""
    return QAAnswer(query=query, url=None, excerpt=None, match_type="none")


def _format_candidates(candidates: list[RetrievedChunk]) -> str:
    """Render candidate chunks for the prompt, each labelled with its source URL."""
    lines = []
    for i, candidate in enumerate(candidates, start=1):
        lines.append(f"[{i}] URL: {candidate.url}\nTEXT: {candidate.text}")
    return "\n\n".join(lines)


def _parse_response(raw: Any) -> tuple[str, str] | None:
    """Parse the LLM's response into ``(url, excerpt)``, or ``None`` for "no answer".

    Accepts a JSON object with ``url``/``excerpt`` keys, since that is what
    :data:`DEFAULT_QA_PROMPT` asks for. Any other shape, an explicit "NONE", or
    missing fields all parse to ``None`` — the caller then returns the null answer
    rather than guessing at malformed output.
    """
    if not raw:
        return None

    text = str(raw).strip()
    if not text or text.upper() == "NONE":
        return None

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    url = payload.get("url")
    excerpt = payload.get("excerpt")

    if not isinstance(url, str) or not url.strip():
        return None
    if not isinstance(excerpt, str) or not excerpt.strip():
        return None

    return url.strip(), excerpt


#: Instructs the model to select and copy, never compose. The "exact character
#: span" requirement is unenforceable by prompt alone, which is why
#: :func:`_llm_answer` always re-verifies the response independently.
DEFAULT_QA_PROMPT = (
    "You are given passages retrieved from a website because they share vocabulary "
    "with the question below. Sharing vocabulary is not the same as answering the "
    "question, and your first job is to judge that difference, not to assume it.\n\n"
    "Question: {query}\n\n"
    "Passages:\n{candidates}\n\n"
    "Step 1 — judge: does any passage genuinely answer this specific question, as "
    "opposed to merely mentioning the same topic or sharing a few words with it? A "
    "passage that is on-topic but does not actually state the answer does NOT "
    "count, even if it is the closest match available.\n\n"
    "Step 2 — if no passage genuinely answers it, reply with exactly the word "
    "NONE. Returning NONE for a related-but-non-answering passage is the correct, "
    "expected outcome, not a fallback to avoid.\n\n"
    "Step 3 — if (and only if) one genuinely does, copy the exact span of text "
    "from that passage that answers the question, word for word. Do not write a "
    "new sentence, paraphrase, summarize, or combine text from more than one "
    "passage. Reply with a single JSON object: "
    '{{"url": "<the passage\'s URL>", "excerpt": "<the exact copied text, '
    'character for character>"}}.'
)
