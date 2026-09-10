"""URL utilities: url normalization, same-domain checking, and deduplication key generation.

Two distinct notions of "the same URL" live here, and keeping them separate matters:

* :func:`normalize_url` returns a URL that is still **fetchable**. It only removes
  things a server cannot see or that never change the response (the fragment, a
  default port, tracking parameters), and leaves the path byte-for-byte otherwise.
  Trailing slashes are preserved, because ``/about`` and ``/about/`` genuinely can
  be different resources and collapsing them invites a redirect or a spurious 404.
* :func:`generate_dedup_key` returns a **comparison key** that is never fetched.
  It folds the extra distinctions that are almost always cosmetic — ``www.``, a
  trailing slash, the scheme — so the frontier does not queue one page four times.
  A key is only ever compared against another key.

The crawler fetches ``normalize_url(...)`` and dedups on ``generate_dedup_key(...)``.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

#: Query parameters that identify a traffic source rather than a resource. Dropping
#: them collapses ``/page?utm_source=x`` and ``/page`` into one crawl target.
#: Deliberately an explicit list of known-cosmetic keys rather than a prefix rule —
#: a site is free to use ``?ref=`` as a real router parameter, and guessing wrong
#: silently drops real pages from the crawl.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_source_platform",
        "gclid",
        "gclsrc",
        "dclid",
        "fbclid",
        "msclkid",
        "twclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "_ga",
        "_gl",
        "yclid",
        "hsa_cam",
        "hsa_grp",
        "hsa_ad",
    }
)

#: Schemes worth crawling. Anything else (mailto:, tel:, javascript:, data:) is not
#: a page and is rejected outright by :func:`normalize_url`.
CRAWLABLE_SCHEMES: frozenset[str] = frozenset({"http", "https"})

_DEFAULT_PORTS = {"http": "80", "https": "443"}

#: Matches a URL scheme prefix, e.g. the "mailto:" in "mailto:hi@example.com".
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


class InvalidURLError(ValueError):
    """Raised when a string cannot be normalized into a crawlable http(s) URL."""


def normalize_url(url: str, base: str | None = None) -> str:
    """Normalize a given URL string.

    Applies only transformations that cannot change which resource is returned:

    * resolve against ``base`` when the URL is relative (``/about``, ``../x``)
    * lowercase the scheme and host (both are case-insensitive per RFC 3986)
    * drop a default port (``:80`` on http, ``:443`` on https)
    * drop the fragment (never sent to the server)
    * drop tracking parameters, then sort the survivors so parameter order does
      not create duplicate crawl targets
    * give an empty path an explicit ``/``

    The path's case and trailing slash are left alone — both can be significant.

    Args:
        url: An absolute or relative URL.
        base: Absolute URL to resolve against when ``url`` is relative.

    Returns:
        The normalized absolute URL.

    Raises:
        InvalidURLError: If the result is not an http(s) URL with a host — for
            example a ``mailto:`` link, a ``javascript:`` handler, or a relative
            URL supplied without a ``base``.
    """
    if url is None:
        raise InvalidURLError("URL is None")

    candidate = url.strip()
    if not candidate:
        raise InvalidURLError("URL is empty")

    if base:
        candidate = urljoin(base.strip(), candidate)

    split = urlsplit(candidate)

    scheme = split.scheme.lower()
    if scheme not in CRAWLABLE_SCHEMES:
        raise InvalidURLError(f"Not a crawlable http(s) URL: {url!r}")

    if not split.hostname:
        raise InvalidURLError(f"URL has no host: {url!r}")

    netloc = split.hostname.lower()
    if split.port is not None and str(split.port) != _DEFAULT_PORTS[scheme]:
        netloc = f"{netloc}:{split.port}"

    kept = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    path = split.path or "/"

    # Fragment dropped: it is a client-side anchor and never reaches the server.
    return urlunsplit((scheme, netloc, path, query, ""))


def is_same_domain(url: str, base_domain: str, include_subdomains: bool = False) -> bool:
    """Check if the given URL belongs to the base domain.

    ``www.`` is treated as cosmetic on both sides, so ``https://www.example.com/a``
    matches ``example.com``. Other subdomains are treated as separate sites by
    default: ``blog.example.com`` is frequently a different CMS with its own SEO
    profile, and silently pulling it into the crawl would inflate the page budget
    and mix two sites' findings into one report. Pass ``include_subdomains=True``
    to opt into crawling them.

    Args:
        url: The URL to test. A string that cannot be parsed as an http(s) URL
            returns ``False`` rather than raising.
        base_domain: A bare host (``example.com``) or any absolute URL on it.
        include_subdomains: Also match hosts ending in ``.<base_domain>``.

    Returns:
        ``True`` if the URL should be considered part of the same site.
    """
    host = _host_of(url)
    base_host = _host_of(base_domain)

    if not host or not base_host:
        return False

    if host == base_host:
        return True

    return include_subdomains and host.endswith(f".{base_host}")


def generate_dedup_key(url: str, base: str | None = None) -> str:
    """Generate a unique deduplication key for a given URL.

    Starts from :func:`normalize_url`, then additionally folds the distinctions
    that are cosmetic in practice but produce duplicate crawl targets:

    * the scheme, so ``http://`` and ``https://`` share a key
    * a leading ``www.``
    * a trailing slash on non-root paths

    Path case is deliberately *not* folded — paths are case-sensitive on most
    origins, so ``/About`` and ``/about`` stay distinct.

    The result is a comparison token only. Never fetch it; fetch the URL that
    :func:`normalize_url` returned.

    Args:
        url: An absolute or relative URL.
        base: Absolute URL to resolve against when ``url`` is relative.

    Returns:
        A stable string key for the frontier's "already seen" set.

    Raises:
        InvalidURLError: Propagated from :func:`normalize_url`.
    """
    split = urlsplit(normalize_url(url, base=base))

    host = split.netloc
    if host.startswith("www."):
        host = host[4:]

    path = split.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    key = f"{host}{path}"
    if split.query:
        key = f"{key}?{split.query}"
    return key


def _host_of(value: str) -> str:
    """Return the lowercased, ``www.``-stripped host of an absolute URL or bare host.

    Returns ``""`` for anything that is not http(s) or a bare host. The guard on
    non-crawlable schemes is load-bearing: ``urlsplit("//mailto:hi@example.com")``
    reads ``example.com`` out of what is actually the userinfo section, which would
    make every ``mailto:`` link on a page look like an on-domain crawl target.
    """
    if not value:
        return ""

    candidate = value.strip()
    if "//" not in candidate:
        if _SCHEME_RE.match(candidate):
            # A scheme-bearing but non-hierarchical URL: mailto:, tel:, javascript:.
            return ""
        # A bare host such as "example.com" or "www.example.com/".
        candidate = f"//{candidate}"
    elif urlsplit(candidate).scheme.lower() not in CRAWLABLE_SCHEMES | {""}:
        return ""

    host = (urlsplit(candidate).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host
