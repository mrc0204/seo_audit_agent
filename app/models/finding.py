"""Data contract for a single deterministic SEO finding.

Schema locked per EXECUTION_PLAN.md, Phase 1. Produced by the Q1 deterministic
checks (Phase 4). Per the plan's risk notes, an LLM step may only rewrite
``suggested_fix`` — ``metric`` and ``evidence`` are never LLM-writable.

Output shape: the assignment brief specifies ``audit.json`` as one entry per
finding of ``{metric, page, severity, evidence, suggested_fix}``. ``check_id`` is
an internal field the plan requires so a validator can trace which deterministic
rule fired; it is carried on the model but excluded from the delivered JSON by
:meth:`Finding.to_audit_entry`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Finding(BaseModel):
    """A single evidence-backed SEO issue found on a page."""

    metric: str
    page: str
    severity: Literal["critical", "warning", "info"]
    evidence: str  # verbatim snippet or structured fact from PageData
    suggested_fix: str
    check_id: str  # which deterministic rule fired (internal; not in audit.json)

    def to_audit_entry(self) -> dict:
        """Serialize to the exact ``audit.json`` entry shape from the brief.

        Drops ``check_id``, which is internal traceability, not a deliverable field.
        """
        return self.model_dump(mode="json", exclude={"check_id"})
