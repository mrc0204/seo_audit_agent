"""Data contract for a grounded Q&A answer.

Schema locked per EXECUTION_PLAN.md, Phase 1. Produced by the Q3 retrieval + QA
pipeline (Phase 6) after ``qa_validator.py`` confirms ``excerpt`` is a literal
substring of the cited page's text. When nothing qualifies — or the excerpt fails
the substring check — ``url`` and ``excerpt`` are null and the pipeline must never
fabricate a replacement.

Output shape: the assignment brief specifies ``answer.json`` as ``{query, url,
excerpt}``, with ``url`` and ``excerpt`` null when the site does not support an
answer. ``match_type`` is an internal record of how the excerpt was verified; it is
excluded from the delivered JSON by :meth:`QAAnswer.to_answer_json`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class QAAnswer(BaseModel):
    """A grounded answer to one question, or an explicit null result."""

    query: str
    url: str | None
    excerpt: str | None
    match_type: Literal["exact_substring", "none"] | None  # internal; not in answer.json

    def to_answer_json(self) -> dict:
        """Serialize to the exact ``answer.json`` shape from the brief.

        Drops ``match_type``, which records how the excerpt was verified rather than
        forming part of the deliverable.
        """
        return self.model_dump(mode="json", exclude={"match_type"})
