"""HTML parsing utilities for cleaning trees and extracting visible text content."""

from bs4 import BeautifulSoup


def parse_html(raw_html: str) -> BeautifulSoup:
    """Parse raw HTML string into a BeautifulSoup tree with scripts and styles stripped."""
    raise NotImplementedError


def extract_visible_text(soup: BeautifulSoup) -> str:
    """Extract clean visible textual content from a parsed BeautifulSoup object."""
    raise NotImplementedError
