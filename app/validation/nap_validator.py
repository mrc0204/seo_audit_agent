"""Validator for NAP audit outputs."""

from typing import Any


def validate_nap(output: Any, pages: list[Any]) -> bool:
    """Validate NAP audit comparison outputs against page evidence."""
    raise NotImplementedError
