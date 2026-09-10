"""Data contracts for Name/Address/Phone (NAP) extraction and cross-page comparison.

Schema locked per EXECUTION_PLAN.md, Phase 1. Populated by the Q2 NAP pipeline
(Phase 5): ``NAPValue`` is one candidate value pulled from one page/source;
``NAPComparison`` is the cross-page verdict for one field (name/address/phone),
carrying the ``NAPValue`` evidence it was derived from.

Output shape: the assignment brief specifies ``nap_report.json`` as ``{field,
pages_compared, values, normalized_values, confidence, verdict}``. The ``evidence``
list is an internal audit trail the plan requires (``nap_validator.py`` re-checks
each ``raw_value`` against the cited page); it is excluded from the delivered JSON
by :meth:`NAPComparison.to_report_entry`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class NAPValue(BaseModel):
    """A single NAP candidate value extracted from one page."""

    page: str
    raw_value: str
    normalized_value: str
    source: Literal["visible_text", "json_ld", "microdata", "tel_link", "llm"]


class NAPComparison(BaseModel):
    """Cross-page consistency verdict for one NAP field (name/address/phone)."""

    field: Literal["name", "address", "phone"]
    pages_compared: list[str]
    values: list[str]
    normalized_values: list[str]
    confidence: float
    verdict: Literal["consistent", "inconsistent", "insufficient_data"]
    evidence: list[NAPValue]  # internal audit trail; not in nap_report.json

    def to_report_entry(self) -> dict:
        """Serialize to the exact ``nap_report.json`` entry shape from the brief.

        Drops ``evidence``, which is the internal audit trail, not a deliverable field.
        """
        return self.model_dump(mode="json", exclude={"evidence"})
