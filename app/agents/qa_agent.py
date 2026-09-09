"""Evidence-grounded Question Answering agent."""

from typing import Any


def answer_question(query: str, pages: list[Any]) -> Any:
    """Answer user questions grounded strictly in retrieved context from crawled pages."""
    raise NotImplementedError
