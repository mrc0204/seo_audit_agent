"""Tests for the Phase 2 crawler core (``app/crawler/*``).

Every HTTP interaction is served by an :class:`httpx.MockTransport` over an
in-memory fixture site, so the suite is deterministic, offline, and safe to run in
CI. Politeness delays are set to zero here; that is a test-speed concession, and
the delay logic itself is asserted separately against a robots.txt fixture.
"""

from __future__ import annotations

import httpx
import pytest

from app.crawler.crawler import Crawler, FetchResult, _extract_links, _looks_js_rendered
from app.crawler.robots import RobotsChecker
from app.crawler.sitemap import fetch_sitemap
from app.crawler.url_utils import (
    InvalidURLError,
    generate_dedup_key,
    is_same_domain,
    normalize_url,
)

# ---------------------------------------------------------------------------
# url_utils — normalization edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Fragments never reach the server.
        ("https://example.com/about#team", "https://example.com/about"),
        ("https://example.com/#", "https://example.com/"),
        # Scheme and host are case-insensitive; the path is not.
        ("HTTPS://Example.COM/About", "https://example.com/About"),
        # Default ports carry no information.
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:80/a", "http://example.com/a"),
        # A non-default port does.
        ("https://example.com:8443/a", "https://example.com:8443/a"),
        # An empty path becomes an explicit root.
        ("https://example.com", "https://example.com/"),
        # Tracking parameters are dropped; real ones survive and are sorted.
        ("https://example.com/p?utm_source=news&id=7", "https://example.com/p?id=7"),
        ("https://example.com/p?fbclid=xyz", "https://example.com/p"),
        ("https://example.com/p?b=2&a=1", "https://example.com/p?a=1&b=2"),
        # Trailing slashes are preserved: /about and /about/ may differ.
        ("https://example.com/about/", "https://example.com/about/"),
    ],
)
def test_normalize_url_edge_cases(raw, expected):
    assert normalize_url(raw) == expected


def test_normalize_url_resolves_relative_against_base():
    base = "https://example.com/docs/guide"
    assert normalize_url("/about", base=base) == "https://example.com/about"
    assert normalize_url("../team", base=base) == "https://example.com/team"
    assert normalize_url("faq.html", base=base) == "https://example.com/docs/faq.html"


@pytest.mark.parametrize(
    "raw",
    [
        "mailto:hi@example.com",
        "tel:+919876543210",
        "javascript:void(0)",
        "data:text/html,hi",
        "/relative-without-base",
        "",
        "   ",
    ],
)
def test_normalize_url_rejects_non_crawlable(raw):
    with pytest.raises(InvalidURLError):
        normalize_url(raw)


def test_normalize_url_keeps_query_value_of_tracking_lookalike():
    # "ref" is NOT in TRACKING_PARAMS — a site may route on it, and dropping it
    # would silently remove real pages from the crawl.
    assert normalize_url("https://example.com/p?ref=abc") == "https://example.com/p?ref=abc"


@pytest.mark.parametrize(
    ("url", "base", "expected"),
    [
        ("https://example.com/a", "example.com", True),
        ("https://www.example.com/a", "example.com", True),
        ("https://example.com/a", "www.example.com", True),
        ("http://example.com/a", "https://example.com/", True),
        ("https://blog.example.com/a", "example.com", False),
        ("https://example.com.evil.com/a", "example.com", False),
        ("https://notexample.com/a", "example.com", False),
        ("mailto:hi@example.com", "example.com", False),
        ("", "example.com", False),
    ],
)
def test_is_same_domain(url, base, expected):
    assert is_same_domain(url, base) is expected


def test_is_same_domain_can_opt_into_subdomains():
    assert is_same_domain("https://blog.example.com/a", "example.com", include_subdomains=True)
    # Still must be a genuine subdomain, not a suffix collision.
    assert not is_same_domain(
        "https://evilexample.com/a", "example.com", include_subdomains=True
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # www., scheme and trailing slash are all folded for dedup purposes.
        ("https://example.com/about", "https://www.example.com/about"),
        ("https://example.com/about", "http://example.com/about"),
        ("https://example.com/about", "https://example.com/about/"),
        ("https://example.com/about#team", "https://example.com/about"),
        ("https://example.com/about?utm_source=x", "https://example.com/about"),
    ],
)
def test_generate_dedup_key_folds_cosmetic_differences(a, b):
    assert generate_dedup_key(a) == generate_dedup_key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Path case is significant on most origins.
        ("https://example.com/About", "https://example.com/about"),
        # A real query parameter distinguishes two resources.
        ("https://example.com/p?id=1", "https://example.com/p?id=2"),
        ("https://example.com/a", "https://example.com/b"),
    ],
)
def test_generate_dedup_key_keeps_meaningful_differences(a, b):
    assert generate_dedup_key(a) != generate_dedup_key(b)


def test_generate_dedup_key_never_returned_as_fetchable():
    # The key is a comparison token, deliberately not a URL.
    assert generate_dedup_key("https://www.example.com/about/") == "example.com/about"


# ---------------------------------------------------------------------------
# robots.py
# ---------------------------------------------------------------------------

ROBOTS_FIXTURE = """
User-agent: *
Disallow: /private/
Disallow: /admin
Crawl-delay: 2

User-agent: seo-audit-agent
Disallow: /private/
Crawl-delay: 1

Sitemap: https://example.com/sitemap.xml
""".strip()


def _robots_transport(body: str = ROBOTS_FIXTURE, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/robots.txt"
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


def _checker(transport: httpx.MockTransport, domain: str = "https://example.com") -> RobotsChecker:
    checker = RobotsChecker(domain)
    with httpx.Client(transport=transport) as client:
        checker.load(client=client)
    return checker


def test_robots_disallow_rules_are_respected():
    checker = _checker(_robots_transport())
    assert checker.status == "ok"
    assert checker.can_fetch("https://example.com/") is True
    assert checker.can_fetch("https://example.com/about") is True
    assert checker.can_fetch("https://example.com/private/secret") is False


def test_robots_crawl_delay_is_read_per_user_agent():
    checker = _checker(_robots_transport())
    assert checker.crawl_delay("seo-audit-agent") == 1.0
    assert checker.crawl_delay("some-other-bot") == 2.0


def test_robots_sitemap_declarations_are_exposed():
    checker = _checker(_robots_transport())
    assert checker.sitemaps() == ["https://example.com/sitemap.xml"]


def test_robots_missing_allows_everything():
    checker = _checker(_robots_transport(body="", status=404))
    assert checker.status == "missing"
    assert checker.can_fetch("https://example.com/anything") is True


def test_robots_server_error_disallows_everything():
    # A broken origin means the rules are unknown; assuming consent is the
    # impolite failure mode, so we assume the opposite.
    checker = _checker(_robots_transport(body="", status=503))
    assert checker.status == "server_error"
    assert checker.can_fetch("https://example.com/") is False


def test_robots_network_failure_fails_open_but_records_why():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom", request=request)

    checker = _checker(httpx.MockTransport(handler))
    assert checker.status.startswith("fetch_failed")
    assert checker.can_fetch("https://example.com/") is True


# ---------------------------------------------------------------------------
# sitemap.py
# ---------------------------------------------------------------------------

SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/</loc></url>
  <url><loc>https://example.com/about</loc></url>
  <url><loc>https://example.com/contact?utm_source=news</loc></url>
  <url><loc>https://other-site.com/spam</loc></url>
</urlset>"""

SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
</sitemapindex>"""

SITEMAP_NO_NAMESPACE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url><loc>https://example.com/no-ns</loc></url>
</urlset>"""


def _sitemap_transport(routes: dict[str, tuple[int, str]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        status, body = routes.get(request.url.path, (404, ""))
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


def _fetch(routes: dict[str, tuple[int, str]]) -> list[str]:
    with httpx.Client(transport=_sitemap_transport(routes)) as client:
        return fetch_sitemap("https://example.com", client=client)


def test_sitemap_parses_urls_normalizes_and_drops_off_domain():
    urls = _fetch({"/sitemap.xml": (200, SITEMAP_XML)})
    assert urls == [
        "https://example.com/",
        "https://example.com/about",
        "https://example.com/contact",  # utm_source stripped
    ]
    assert not any("other-site.com" in u for u in urls)


def test_sitemap_index_is_followed():
    urls = _fetch(
        {
            "/sitemap.xml": (200, SITEMAP_INDEX_XML),
            "/sitemap-pages.xml": (200, SITEMAP_XML),
        }
    )
    assert "https://example.com/about" in urls


def test_sitemap_handles_missing_file():
    assert _fetch({}) == []


def test_sitemap_handles_malformed_xml_without_raising():
    # A soft-404 HTML page served at /sitemap.xml is common in the wild.
    assert _fetch({"/sitemap.xml": (200, "<html><body>Not found</body></html>")}) == []
    assert _fetch({"/sitemap.xml": (200, "<urlset><url><loc>trunc")}) == []


def test_sitemap_tolerates_missing_namespace():
    urls = _fetch({"/sitemap.xml": (200, SITEMAP_NO_NAMESPACE_XML)})
    assert urls == ["https://example.com/no-ns"]


def test_sitemap_does_not_loop_on_self_referential_index():
    self_ref = """<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.com/sitemap.xml</loc></sitemap>
    </sitemapindex>"""
    assert _fetch({"/sitemap.xml": (200, self_ref)}) == []


# ---------------------------------------------------------------------------
# crawler.py — fixture site
# ---------------------------------------------------------------------------


def _page(*hrefs: str, body: str = "Some real page content here. " * 20) -> str:
    links = "".join(f'<a href="{h}">link</a>' for h in hrefs)
    return f"<html><body><p>{body}</p>{links}</body></html>"


FIXTURE_SITE: dict[str, tuple[int, str, str]] = {
    "/robots.txt": (200, "text/plain", "User-agent: *\nDisallow: /private/\n"),
    "/": (200, "text/html", _page("/about", "/contact", "/private/secret", "https://ext.com/x")),
    "/about": (200, "text/html", _page("/team", "/")),
    "/contact": (200, "text/html", _page("/")),
    "/team": (200, "text/html", _page("/deep")),
    "/deep": (200, "text/html", _page()),
    "/private/secret": (200, "text/html", _page()),
    "/brochure.pdf": (200, "application/pdf", "%PDF-1.4 binary"),
    "/missing": (404, "text/html", _page(body="Not found")),
}


def _site_transport(
    site: dict[str, tuple[int, str, str]] | None = None,
    counter: dict[str, int] | None = None,
) -> httpx.MockTransport:
    site = site or FIXTURE_SITE

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if counter is not None:
            counter[path] = counter.get(path, 0) + 1
        if path not in site:
            return httpx.Response(404, text="not found")
        status, content_type, body = site[path]
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    return httpx.MockTransport(handler)


def _crawler(**kwargs) -> Crawler:
    kwargs.setdefault("polite_delay", 0.0)
    kwargs.setdefault("use_sitemap", False)
    return Crawler(**kwargs)


def _paths(results: list[FetchResult]) -> set[str]:
    return {httpx.URL(r.final_url).path for r in results}


def test_crawl_discovers_pages_by_following_links():
    results = _crawler().crawl(
        "https://example.com/", max_pages=50, max_depth=3, transport=_site_transport()
    )
    assert {"/", "/about", "/contact", "/team"} <= _paths(results)


def test_crawl_never_exceeds_max_pages():
    results = _crawler().crawl(
        "https://example.com/", max_pages=2, max_depth=5, transport=_site_transport()
    )
    assert len(results) == 2


def test_crawl_never_exceeds_max_depth():
    crawler = _crawler()
    results = crawler.crawl(
        "https://example.com/", max_pages=50, max_depth=1, transport=_site_transport()
    )
    # Depth 0 is the homepage, depth 1 its direct links. /team sits at depth 2.
    assert max(r.depth for r in results) <= 1
    assert "/team" not in _paths(results)


def test_crawl_respects_robots_disallow():
    crawler = _crawler()
    results = crawler.crawl(
        "https://example.com/", max_pages=50, max_depth=3, transport=_site_transport()
    )
    assert "/private/secret" not in _paths(results)
    assert crawler.last_report is not None
    assert any(
        reason == "robots_disallow" for reason in crawler.last_report.skipped.values()
    )


def test_crawl_stays_on_domain_and_records_why():
    crawler = _crawler()
    crawler.crawl(
        "https://example.com/", max_pages=50, max_depth=3, transport=_site_transport()
    )
    skipped = crawler.last_report.skipped
    assert any(
        "ext.com" in url and reason == "off_domain" for url, reason in skipped.items()
    )


def test_crawl_dedups_by_normalized_key_not_raw_url():
    site = dict(FIXTURE_SITE)
    # Four spellings of the same page; only one fetch should result.
    site["/"] = (
        200,
        "text/html",
        _page("/about", "/about/", "/about?utm_source=x", "/about#team"),
    )
    counter: dict[str, int] = {}
    _crawler().crawl(
        "https://example.com/",
        max_pages=50,
        max_depth=2,
        transport=_site_transport(site, counter),
    )
    assert counter.get("/about", 0) == 1


def test_crawl_skips_non_html_content():
    site = dict(FIXTURE_SITE)
    site["/"] = (200, "text/html", _page("/brochure.pdf", "/about"))
    crawler = _crawler()
    results = crawler.crawl(
        "https://example.com/", max_pages=50, max_depth=2, transport=_site_transport(site)
    )
    pdf = next(r for r in results if r.url.endswith("/brochure.pdf"))
    assert pdf.html == ""
    assert crawler.last_report.skipped.get(pdf.url) == "non_html"


def test_crawl_records_redirects_without_double_counting():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://example.com/new"})
        if request.url.path == "/new":
            return httpx.Response(
                200, text=_page(), headers={"content-type": "text/html"}
            )
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        return httpx.Response(404, text="")

    results = _crawler().crawl(
        "https://example.com/old",
        max_pages=5,
        max_depth=0,
        transport=httpx.MockTransport(handler),
    )
    assert len(results) == 1
    result = results[0]
    assert result.url == "https://example.com/old"
    assert result.final_url == "https://example.com/new"
    assert result.redirect_chain == ["https://example.com/old"]


def test_crawl_retries_transient_errors_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="")
        return httpx.Response(200, text=_page(), headers={"content-type": "text/html"})

    results = _crawler(max_attempts=3).crawl(
        "https://example.com/", max_pages=1, max_depth=0, transport=httpx.MockTransport(handler)
    )
    assert results[0].status_code == 200
    assert attempts["n"] == 3


def test_crawl_does_not_retry_a_404():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        attempts["n"] += 1
        return httpx.Response(404, text="gone", headers={"content-type": "text/html"})

    results = _crawler(max_attempts=3).crawl(
        "https://example.com/", max_pages=1, max_depth=0, transport=httpx.MockTransport(handler)
    )
    assert results[0].status_code == 404
    assert attempts["n"] == 1  # 4xx is a real answer, not a transient failure


def test_crawl_records_connection_failure_instead_of_crashing():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        raise httpx.ConnectError("refused", request=request)

    results = _crawler(max_attempts=2).crawl(
        "https://example.com/", max_pages=1, max_depth=0, transport=httpx.MockTransport(handler)
    )
    assert results[0].status_code == 0
    assert results[0].error == "ConnectError"
    assert results[0].ok is False


def test_crawl_seeds_frontier_from_sitemap():
    site = dict(FIXTURE_SITE)
    site["/"] = (200, "text/html", _page())  # homepage links nowhere
    site["/sitemap.xml"] = (200, "application/xml", SITEMAP_XML)
    crawler = _crawler(use_sitemap=True)
    results = crawler.crawl(
        "https://example.com/", max_pages=50, max_depth=2, transport=_site_transport(site)
    )
    assert "/about" in _paths(results)
    assert crawler.last_report.sitemap_urls_found > 0


def test_crawl_captures_x_robots_tag_header():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        return httpx.Response(
            200,
            text=_page(),
            headers={"content-type": "text/html", "x-robots-tag": "noindex"},
        )

    results = _crawler().crawl(
        "https://example.com/", max_pages=1, max_depth=0, transport=httpx.MockTransport(handler)
    )
    assert results[0].x_robots_tag == ["noindex"]


def test_crawl_captures_multiple_x_robots_tag_header_occurrences():
    # A response can legitimately carry the header more than once (a general
    # directive plus a per-bot one is a real pattern); dropping any of them
    # could hide a real noindex.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        headers = httpx.Headers(
            [
                ("content-type", "text/html"),
                ("x-robots-tag", "noindex"),
                ("x-robots-tag", "googlebot: nofollow"),
            ]
        )
        return httpx.Response(200, text=_page(), headers=headers)

    results = _crawler().crawl(
        "https://example.com/", max_pages=1, max_depth=0, transport=httpx.MockTransport(handler)
    )
    assert results[0].x_robots_tag == ["noindex", "googlebot: nofollow"]


def test_crawl_of_a_page_with_no_x_robots_tag_header_has_an_empty_list():
    results = _crawler().crawl(
        "https://example.com/", max_pages=1, max_depth=0, transport=_site_transport()
    )
    assert results[0].x_robots_tag == []


def test_crawl_notes_js_rendered_shell():
    shell = '<html><body><div id="root"></div><script src="/app.js"></script></body></html>'
    site = {
        "/robots.txt": (404, "text/plain", ""),
        "/": (200, "text/html", shell),
    }
    crawler = _crawler()
    crawler.crawl(
        "https://example.com/", max_pages=1, max_depth=0, transport=_site_transport(site)
    )
    assert any("client-rendered" in note for note in crawler.last_report.notes)


def test_crawl_report_records_robots_status_and_depth():
    crawler = _crawler()
    crawler.crawl(
        "https://example.com/", max_pages=50, max_depth=2, transport=_site_transport()
    )
    report = crawler.last_report
    assert report.start_url == "https://example.com/"
    assert report.robots_status == "ok"
    assert report.max_depth_reached <= 2


def test_crawl_rejects_a_non_http_start_url():
    with pytest.raises(InvalidURLError):
        _crawler().crawl("mailto:hi@example.com")


# ---------------------------------------------------------------------------
# crawler.py — helpers
# ---------------------------------------------------------------------------


def test_extract_links_resolves_and_filters():
    html = """
      <a href="/about">About</a>
      <a href="https://example.com/about#team">Dup</a>
      <a href="mailto:hi@example.com">Mail</a>
      <a href="#top">Anchor</a>
      <a href="https://other.com/x">External</a>
    """
    links = _extract_links(html, base="https://example.com/")
    assert links == ["https://example.com/about", "https://other.com/x"]


def test_looks_js_rendered_distinguishes_shell_from_real_page():
    shell = '<html><body><div id="root"></div><script src="/app.js"></script></body></html>'
    assert _looks_js_rendered(shell) is True
    assert _looks_js_rendered(_page()) is False


def test_looks_js_rendered_does_not_flag_a_genuinely_small_page():
    # example.com is a real, complete, server-rendered page of ~170 characters with
    # no scripts at all. Flagging it would put a false claim in the crawl report.
    tiny = (
        "<html><head><title>Example Domain</title></head><body>"
        "<h1>Example Domain</h1><p>This domain is for use in illustrative examples "
        "in documents.</p></body></html>"
    )
    assert _looks_js_rendered(tiny) is False
