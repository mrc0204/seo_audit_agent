"""Index builder for page content chunks (BM25 / TF-IDF)."""

from typing import Any
from rank_bm25 import BM25Okapi


def build_index(pages: list[Any]) -> Any:
    """Build a searchable retrieval index (e.g. BM25 or TF-IDF) over parsed page content chunks."""
    raise NotImplementedError
