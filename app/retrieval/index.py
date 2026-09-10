"""Index builder for page content chunks (BM25 / TF-IDF).

Chunk = one block from :func:`app.extraction.content_extractor.extract_clean_content`
(one paragraph, heading, list item, etc). Phase 3 already guarantees every block is
a literal substring of that page's ``PageData.text``, so this module inherits the
guarantee free: no rewriting, no joining, no summarizing happens here. Each chunk
carries the ``(url, char_start, char_end)`` span into the page's own ``text``, which
is what lets :mod:`app.validation.qa_validator` re-verify a claimed excerpt against
the original source rather than trusting the chunker.

BM25 (via ``rank_bm25``), not embeddings — free, deterministic, no API key, per the
plan and the brief's "no paid API" constraint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from app.extraction.content_extractor import BLOCK_SEPARATOR, extract_clean_content
from app.extraction.html_parser import parse_html

#: Word-boundary tokenizer shared by indexing and querying. Must be the same
#: function on both sides or BM25 scores mean nothing.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Stopwords excluded from every token stream. Without this, two chunks can share
#: only "the"/"for"/"is" and still earn a nonzero BM25 score, which is enough to
#: rank an unrelated chunk above real "nothing matches" and make the offline
#: fallback answer a question the site never addresses. Found live: "Do you offer
#: refunds for shipments?" matched a wholesale paragraph on the shared word "for".
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
        "now", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "would", "could", "shall",
        "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those", "what", "which", "who", "whom",
    }
)

#: Chunks shorter than this are dropped before indexing — a three-word fragment is
#: retrieval noise, not a candidate answer.
MIN_CHUNK_CHARS = 15


@dataclass(frozen=True)
class Chunk:
    """One indexed unit of page text, with its span into the source.

    Attributes:
        text: The chunk's text — a literal substring of ``page.text`` at
            ``[char_start:char_end)``.
        url: The page's ``final_url``.
        char_start: Start offset into that page's ``PageData.text``.
        char_end: End offset (exclusive).
    """

    text: str
    url: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class RetrievalIndex:
    """A built BM25 index plus the chunk metadata it scores over.

    Opaque to callers beyond passing it to
    :func:`app.retrieval.retriever.retrieve_top_k` — the ``Any`` return type on
    :func:`build_index` keeps that boundary explicit.
    """

    chunks: list[Chunk]
    bm25: BM25Okapi | None


def _singularize(token: str) -> str:
    """Strip a bare trailing plural "s" so "courses" and "course" tokenize identically.

    Found live: a real question ("What courses does NxtWave offer?") shared zero
    BM25 tokens with the page's own course-listing text ("Backend Developer
    Course", "Spring Boot Course Syllabus") purely because the question used the
    plural and the page used the singular — a pure lexical mismatch, unrelated to
    whether the content actually answers the question. BM25 scores were exactly
    0.0 for those chunks, so they never even reached the LLM judge in
    :mod:`app.agents.qa_agent` to be considered.

    Deliberately narrow: only a single trailing "s" is stripped, and only when
    preceded by another letter and not itself a double-s ("class", "process"),
    so this never removes a "s" that is not plural. It can still merge a handful
    of unrelated singular words that happen to end in one "s" (e.g. "status" ->
    "statu") -- harmless, since query and document sides always stem the same
    word the same way, so it can only ever fail to distinguish two words, never
    wrongly match two that would not already share the stripped stem.

    Args:
        token: A single lowercase alphanumeric token.

    Returns:
        The token with a bare trailing "s" removed, or the token unchanged.
    """
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-token split with stopwords removed and plurals folded.

    The one tokenizer used everywhere — indexing and querying must use the same
    function or BM25 scores are meaningless, and stopwords are dropped so two
    chunks cannot appear related merely by both containing "the" or "for". A
    light plural fold (see :func:`_singularize`) runs after stopword filtering so
    a plural question term matches the site's own singular wording.

    Args:
        text: Any string.

    Returns:
        Lowercased, non-stopword alphanumeric tokens, in order, with a bare
        trailing plural "s" removed.
    """
    return [
        _singularize(t)
        for t in _TOKEN_RE.findall((text or "").lower())
        if t not in STOPWORDS
    ]


def build_index(pages: list[Any]) -> RetrievalIndex:
    """Build a searchable retrieval index (e.g. BM25 or TF-IDF) over parsed page content chunks.

    Args:
        pages: ``PageData`` objects from Phase 3.

    Returns:
        A :class:`RetrievalIndex`. ``bm25`` is ``None`` when no page produced any
        chunk — an empty site is a valid input, not an error, and
        :func:`app.retrieval.retriever.retrieve_top_k` returns no candidates for it.
    """
    chunks: list[Chunk] = []

    for page in pages:
        url = getattr(page, "final_url", "") or getattr(page, "url", "")
        html = getattr(page, "html", "") or ""
        text = getattr(page, "text", "") or ""
        if not url or not text:
            continue

        cursor = 0
        for block in extract_clean_content(parse_html(html)).split(BLOCK_SEPARATOR):
            if len(block) < MIN_CHUNK_CHARS:
                continue

            # Search from `cursor` so a block that legitimately repeats on the page
            # (a phrase used twice) is indexed as two distinct spans rather than
            # both pointing at the first occurrence.
            start = text.find(block, cursor)
            if start == -1:
                start = text.find(block)
            if start == -1:
                # Should not happen given the Phase 3 substring invariant, but a
                # chunk that cannot be located is unusable as an excerpt, so it is
                # skipped rather than indexed with a wrong span.
                continue

            chunks.append(
                Chunk(text=block, url=url, char_start=start, char_end=start + len(block))
            )
            cursor = start + len(block)

    if not chunks:
        return RetrievalIndex(chunks=[], bm25=None)

    corpus = [tokenize(chunk.text) for chunk in chunks]
    # A chunk that tokenizes to nothing (pure punctuation) would break BM25's IDF
    # computation; guard rather than assume every chunk survives tokenization.
    if not any(corpus):
        return RetrievalIndex(chunks=chunks, bm25=None)

    return RetrievalIndex(chunks=chunks, bm25=BM25Okapi(corpus))
