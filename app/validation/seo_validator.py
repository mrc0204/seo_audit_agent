"""Validator for verifying SEO audit finding outputs against evidence."""

from typing import Any


def validate_seo(output: Any, pages: list[Any]) -> bool:
    """Validate SEO audit agent findings against page evidence and report schema rules."""
    raise NotImplementedError
