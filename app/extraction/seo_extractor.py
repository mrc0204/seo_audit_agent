"""SEO metadata extractor for extracting page titles, meta tags, canonicals, headings, links, and images."""

from typing import Any
from bs4 import BeautifulSoup


def extract_seo_fields(soup: BeautifulSoup, url: str) -> dict[str, Any]:
    """Extract partial PageData SEO fields (title, meta, canonical, robots meta, headings, images, links)."""
    raise NotImplementedError
