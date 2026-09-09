"""Structured schema data extractor for JSON-LD scripts and Microdata tags."""

from typing import Any
from bs4 import BeautifulSoup


def extract_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Extract JSON-LD structured data objects embedded in script tags."""
    raise NotImplementedError


def extract_microdata(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Extract Microdata structured items from HTML attributes."""
    raise NotImplementedError
