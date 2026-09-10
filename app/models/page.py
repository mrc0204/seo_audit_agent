"""Data contract for a single crawled page.

Schema locked per EXECUTION_PLAN.md, Phase 1 ("Data contracts first"). This is the
seam every later phase (crawler, extraction, Q1/Q2/Q3 agents) writes into and reads
from, so field names and types here are treated as fixed.

``ImageRef`` and ``LinkRef`` are referenced by the plan's ``PageData`` snippet
("images: list[ImageRef]  # src + alt", "links: list[LinkRef]  # href + anchor text
+ internal/external flag") but their own field layouts are not spelled out anywhere
in the plan. They are defined here to match those inline comments exactly:
``ImageRef`` = src + alt, ``LinkRef`` = href + anchor text + an internal/external
flag (named ``is_internal``). This is a documented fill-in, not part of the locked
schema — revisit if the original brief names these fields differently.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ImageRef(BaseModel):
    """An ``<img>`` reference extracted from a page: src + alt."""

    src: str
    alt: str | None = None


class LinkRef(BaseModel):
    """A hyperlink extracted from a page: href + anchor text + internal/external flag."""

    href: str
    anchor_text: str
    is_internal: bool


class PageData(BaseModel):
    """Everything extracted from a single crawled page."""

    url: str
    final_url: str
    status_code: int
    fetched_at: datetime
    html: str
    text: str  # visible text, whitespace-normalized
    title: str | None
    meta_description: str | None
    canonical: str | None
    robots_meta: list[str]  # e.g. ["noindex", "nofollow"]
    headings: dict[str, list[str]]  # {"h1": [...], "h2": [...]}
    images: list[ImageRef]  # src + alt
    links: list[LinkRef]  # href + anchor text + internal/external flag
    structured_data: list[dict]  # parsed JSON-LD blocks
    content_hash: str  # for duplicate-content detection
