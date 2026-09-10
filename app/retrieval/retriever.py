"""Context retriever fetching top-k chunks with source spans for QA and validation.

Pure lexical scoring, no rewriting: a returned chunk is exactly the
:class:`~app.retrieval.index.Chunk` the index built, carrying the same
``(url, char_start, char_end)`` span. This module's only job is **ranking by
BM25 and handing back a broad top-k** — it does not decide relevance, and does
not decide answerability. Both of those decisions belong one layer up, in
:mod:`app.agents.qa_agent`:

* an LLM, when configured, judges answerability directly over these candidates
  (:data:`app.agents.qa_agent.DEFAULT_QA_PROMPT`), then the excerpt it selects
  is independently re-verified against the source text
  (:func:`app.validation.qa_validator.find_source_span`);
* with no LLM configured, :func:`app.agents.qa_agent._offline_answer` applies
  its own, much stricter coverage threshold
  (:data:`app.agents.qa_agent.OFFLINE_ANSWER_MIN_COVERAGE`) before accepting any
  of these candidates as an actual answer.

This module previously also enforced a 50%-query-coverage floor before a chunk
could even be offered as a candidate, which coupled two different judgments —
"is this chunk worth ranking at all" and "is this chunk relevant enough to be
an answer" — into one function neither the LLM path nor the no-LLM path was
best placed to own alone. Retrieval's job is now deliberately narrower: return
broad, literal top-k-by-BM25-score, and let the layer with an actual judgment
to make (an LLM, or the offline path's own explicit threshold) make it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.index import Chunk, RetrievalIndex, tokenize

#: Default number of candidates returned to the caller (and, from
#: :mod:`app.agents.qa_agent`, on to the LLM as its answerability judgment
#: material). Deliberately wide — 10 to 15 candidates give a semantic judge real
#: material to work with, where the old 5-candidate default (paired with the
#: coverage pre-filter this module used to apply) could exclude the chunk that
#: actually answered the question before an LLM ever got a chance to see it.
DEFAULT_TOP_K = 15


@dataclass(frozen=True)
class RetrievedChunk:
    """A candidate chunk plus its retrieval score.

    Attributes:
        chunk: The underlying indexed chunk, unmodified.
        score: BM25 relevance score for the query. Higher is more relevant;
            comparable only within one retrieval call.
    """

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def url(self) -> str:
        return self.chunk.url

    @property
    def char_start(self) -> int:
        return self.chunk.char_start

    @property
    def char_end(self) -> int:
        return self.chunk.char_end


def retrieve_top_k(query: str, index: RetrievalIndex, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """Retrieve the top-k most relevant content chunks with source span annotations given a query.

    Purely a BM25 ranking step — every chunk the index holds is scored and the
    top ``k`` returned, with no relevance or coverage floor applied here. A chunk
    that shares nothing with the query naturally scores at or near the bottom and
    is excluded by the ``k`` cutoff whenever enough better candidates exist; on a
    very small index it may still appear, which is fine, since the caller (an LLM
    judging answerability, or the offline path's own stricter threshold) is
    equipped to reject it, not this function.

    Args:
        query: The user's natural-language question.
        index: A :class:`~app.retrieval.index.RetrievalIndex` from
            :func:`app.retrieval.index.build_index`.
        k: Maximum candidates to return. Defaults to :data:`DEFAULT_TOP_K` (15) —
            deliberately wide, so a genuine answer ranked just outside a narrower
            cutoff still reaches whichever layer judges answerability.

    Returns:
        Up to ``k`` :class:`RetrievedChunk` objects, highest score first. Empty
        when the query tokenizes to nothing, the index is empty, or ``k`` is not
        positive — each of those is a legitimate "nothing retrieved" outcome, not
        an error, and the caller must return a null ``QAAnswer`` rather than guess.
    """
    if index is None or index.bm25 is None or not index.chunks or k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = index.bm25.get_scores(query_tokens)

    ranked = sorted(
        (RetrievedChunk(chunk=chunk, score=float(score)) for chunk, score in zip(index.chunks, scores)),
        key=lambda r: r.score,
        reverse=True,
    )

    return ranked[:k]
