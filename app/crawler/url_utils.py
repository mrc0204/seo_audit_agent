"""URL utilities: url normalization, same-domain checking, and deduplication key generation."""

from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    """Normalize a given URL string."""
    raise NotImplementedError


def is_same_domain(url: str, base_domain: str) -> bool:
    """Check if the given URL belongs to the base domain."""
    raise NotImplementedError


def generate_dedup_key(url: str) -> str:
    """Generate a unique deduplication key for a given URL."""
    raise NotImplementedError
