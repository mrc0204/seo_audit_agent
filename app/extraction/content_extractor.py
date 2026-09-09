"""Clean main content extractor tailored for retrieval chunking and indexing."""

from bs4 import BeautifulSoup


def extract_clean_content(soup: BeautifulSoup) -> str:
    """Extract clean main content string from a BeautifulSoup tree for retrieval chunking."""
    raise NotImplementedError
