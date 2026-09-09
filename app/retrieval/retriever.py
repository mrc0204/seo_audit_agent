"""Context retriever fetching top-k chunks with source spans for QA and validation."""

from typing import Any


def retrieve_top_k(query: str, index: Any, k: int = 5) -> list[Any]:
    """Retrieve the top-k most relevant content chunks with source span annotations given a query."""
    raise NotImplementedError
