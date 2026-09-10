"""Tests for the Phase 3 extraction layer (``app/extraction/*``).

Three kinds of assertion:

1. **Snapshot tests** against the hand-verified fixture pages in
   ``tests/fixtures/``. Expected values were read off the fixture markup by hand,
   so a change in extraction behaviour has to be justified against real HTML.
2. **Robustness tests** proving no input crashes the layer — malformed HTML,
   broken JSON-LD, empty bodies, missing attributes.
3. **The Q3 substring invariant**, which is the one test in this file that the
   anti-hallucination guarantee actually depends on: every block of retrieval
   content must appear verbatim in ``PageData.text``.
"""

from __future__ import annotations

import pathlib

import pytest

from app.extraction.content_extractor import BLOCK_SEPARATOR, extract_clean_content
from app.extraction.html_parser import extract_visible_text, normalize_whitespace, parse_html
from app.extraction.schema_extractor import extract_json_ld, extract_microdata
from app.extraction.seo_extractor import (
    build_page_data,
    content_hash,
    extract_clean_text,
    extract_seo_fields,
)
from app.models.page import PageData

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

CLEAN_URL = "https://ridgeline.example/"
MESSY_URL = "https://messy.example/products"
MICRODATA_URL = "https://harbourdental.example/contact"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def clean_html() -> str:
    return load("clean_page.html")


@pytest.fixture
def messy_html() -> str:
    return load("messy_page.html")


@pytest.fixture
def microdata_html() -> str:
    return load("microdata_page.html")


# ---------------------------------------------------------------------------
# html_parser
# ---------------------------------------------------------------------------


def test_parse_html_strips_scripts_styles_and_comments(clean_html):
    text = extract_visible_text(parse_html(clean_html))
    assert "this text must never appear in visible text" not in text
    assert "nor must this" not in text
    assert "must not survive into the extracted text" not in text
    assert "Small-batch coffee, roasted in Portland" in text


def test_extract_visible_text_is_whitespace_normalized(clean_html):
    text = extract_visible_text(parse_html(clean_html))
    assert "  " not in text
    assert "\n" not in text
    assert "\t" not in text
    assert text == text.strip()


def test_parse_html_survives_malformed_markup(messy_html):
    # Unclosed <p> and <li> tags throughout; must parse, not raise.
    text = extract_visible_text(parse_html(messy_html))
    assert "We sell widgets of every description" in text


@pytest.mark.parametrize("raw", ["", "   ", "<html>", "<<<>>>", "<p>unclosed"])
def test_parse_html_never_raises_on_degenerate_input(raw):
    assert isinstance(extract_visible_text(parse_html(raw)), str)


def test_normalize_whitespace_is_the_single_shared_rule():
    assert normalize_whitespace("  a \n\t b  ") == "a b"
    assert normalize_whitespace("") == ""
    assert normalize_whitespace(None) == ""


# ---------------------------------------------------------------------------
# seo_extractor — snapshot against the clean fixture
# ---------------------------------------------------------------------------


def test_seo_fields_snapshot_clean_page(clean_html):
    fields = extract_seo_fields(parse_html(clean_html), CLEAN_URL)

    assert fields["title"] == "Ridgeline Coffee Roasters — Small-batch coffee in Portland"
    assert fields["title_count"] == 1
    assert fields["meta_description"] == (
        "Ridgeline Coffee Roasters roasts small-batch single-origin coffee in Portland, Oregon."
    )
    assert fields["canonical"] == "https://ridgeline.example/"
    assert fields["canonical_count"] == 1
    assert fields["robots_meta"] == ["index", "follow"]

    assert fields["headings"]["h1"] == ["Small-batch coffee, roasted in Portland"]
    assert fields["headings"]["h2"] == ["Our roasting process", "Visit the roastery"]
    assert fields["headings"]["h3"] == []


def test_images_distinguish_missing_alt_from_decorative_alt(clean_html):
    images = extract_seo_fields(parse_html(clean_html), CLEAN_URL)["images"]

    assert [i.src for i in images] == [
        "https://ridgeline.example/img/roaster.jpg",
        "https://ridgeline.example/img/divider.png",
    ]
    assert images[0].alt == "A twelve-kilo drum roaster mid-cycle"
    # alt="" is correct markup for a decorative image, NOT a missing alt.
    assert images[1].alt == ""
    assert images[1].alt is not None


def test_links_resolve_and_flag_internal_versus_external(clean_html):
    links = extract_seo_fields(parse_html(clean_html), CLEAN_URL)["links"]
    by_href = {link.href: link for link in links}

    assert by_href["https://ridgeline.example/about"].is_internal is True
    assert by_href["https://ridgeline.example/about"].anchor_text == "About"
    assert by_href["https://social.example/ridgeline"].is_internal is False
    assert by_href["https://ridgeline.example/sourcing"].anchor_text == "sourcing practices"


def test_links_skip_non_crawlable_schemes(clean_html):
    links = extract_seo_fields(parse_html(clean_html), CLEAN_URL)["links"]
    hrefs = [link.href for link in links]
    assert not any(h.startswith(("mailto:", "tel:", "javascript:")) for h in hrefs)
    assert not any("#" in h for h in hrefs)


# ---------------------------------------------------------------------------
# seo_extractor — snapshot against the messy fixture
# ---------------------------------------------------------------------------


def test_seo_fields_snapshot_messy_page(messy_html):
    fields = extract_seo_fields(parse_html(messy_html), MESSY_URL)

    # Present-but-empty title: "" not None. Phase 4 needs to tell these apart.
    assert fields["title"] == ""
    assert fields["title"] is not None
    assert fields["title_count"] == 1

    # Absent meta description: None, not "".
    assert fields["meta_description"] is None
    assert fields["meta_description_count"] == 0

    # Two canonicals is its own finding; keeping only the first would hide it.
    assert fields["canonical_count"] == 2
    assert fields["canonical"] == "https://messy.example/products"

    # Directive casing and comma-splitting normalized.
    assert fields["robots_meta"] == ["noindex", "nofollow"]

    # Multiple H1s and a skipped level survive intact for Phase 4 to judge.
    assert fields["headings"]["h1"] == ["Widgets", "Also Widgets"]
    assert fields["headings"]["h2"] == []
    assert fields["headings"]["h3"] == ["Skipped straight past H2"]


def test_messy_page_image_alt_states(messy_html):
    images = extract_seo_fields(parse_html(messy_html), MESSY_URL)["images"]
    alts = [img.alt for img in images]

    assert alts[0] is None  # no alt attribute at all — a real finding
    assert alts[1] == ""  # decorative spacer — correct markup
    assert alts[2] == "An image with no source at all"

    # A src-less <img> is still reported; a broken image tag is worth flagging.
    assert images[2].src == ""


def test_messy_page_link_with_image_only_anchor_text(messy_html):
    links = extract_seo_fields(parse_html(messy_html), MESSY_URL)["links"]
    deals = next(link for link in links if link.href.endswith("/deals"))
    assert deals.anchor_text == ""
    assert deals.is_internal is True


# ---------------------------------------------------------------------------
# schema_extractor
# ---------------------------------------------------------------------------


def test_json_ld_parsed_from_clean_page(clean_html):
    blocks = extract_json_ld(clean_html)
    assert len(blocks) == 1
    assert blocks[0]["@type"] == "LocalBusiness"
    assert blocks[0]["name"] == "Ridgeline Coffee Roasters"
    assert blocks[0]["address"]["streetAddress"] == "412 SE Ankeny St"


def test_one_broken_json_ld_block_does_not_lose_the_good_one(messy_html):
    # The Product block has trailing commas and is invalid JSON; the Organization
    # block after it is fine and must still be returned.
    blocks = extract_json_ld(messy_html)
    assert len(blocks) == 1
    assert blocks[0]["@type"] == "Organization"
    assert blocks[0]["name"] == "Messy Widgets Ltd"


def test_json_ld_must_be_read_from_raw_html_not_a_stripped_tree(clean_html):
    # parse_html removes <script>, so a stripped tree legitimately yields nothing.
    # This documents the asymmetry rather than treating it as a bug.
    assert extract_json_ld(clean_html) != []
    assert extract_json_ld(parse_html(clean_html)) == []


def test_json_ld_unwraps_graph_and_arrays():
    graph = """<script type="application/ld+json">
    {"@context": "https://schema.org",
     "@graph": [{"@type": "Organization", "name": "A"}, {"@type": "WebSite", "name": "B"}]}
    </script>"""
    names = [b.get("name") for b in extract_json_ld(graph)]
    assert names == ["A", "B"]

    array = """<script type="application/ld+json">
    [{"@type": "Organization", "name": "A"}, {"@type": "WebSite", "name": "B"}]
    </script>"""
    assert [b.get("name") for b in extract_json_ld(array)] == ["A", "B"]


@pytest.mark.parametrize(
    "payload",
    ['{"a": 1,}', "not json at all", "", "   ", "[1, 2, 3]", "null"],
)
def test_json_ld_degenerate_payloads_never_raise(payload):
    html = f'<script type="application/ld+json">{payload}</script>'
    assert isinstance(extract_json_ld(html), list)


def test_microdata_snapshot(microdata_html):
    items = extract_microdata(parse_html(microdata_html))
    assert len(items) == 1

    item = items[0]
    assert item["@type"] == "LocalBusiness"
    assert item["name"] == "Harbour Dental Practice"
    assert item["telephone"] == "0117 496 0182"
    # <meta itemprop> carries its value in content=, not in its text.
    assert item["priceRange"] == "££"
    # <a itemprop> carries its value in href=.
    assert item["url"] == "https://harbourdental.example/"

    address = item["address"]
    assert address["@type"] == "PostalAddress"
    assert address["streetAddress"] == "18 Harbour Road"
    assert address["addressLocality"] == "Bristol"
    assert address["postalCode"] == "BS1 5TY"


def test_microdata_absent_returns_empty_list(clean_html):
    assert extract_microdata(parse_html(clean_html)) == []


# ---------------------------------------------------------------------------
# content_extractor — including the Q3 substring invariant
# ---------------------------------------------------------------------------


def test_clean_content_drops_nav_and_footer(clean_html):
    content = extract_clean_content(parse_html(clean_html))
    assert "All rights reserved worldwide" not in content
    assert "roasted single-origin coffee on the east side of Portland" in content


def test_clean_content_is_split_into_blocks(clean_html):
    blocks = extract_clean_content(parse_html(clean_html)).split(BLOCK_SEPARATOR)
    assert len(blocks) > 3
    assert "Small-batch coffee, roasted in Portland" in blocks


def test_every_clean_content_block_is_a_literal_substring_of_page_text(clean_html):
    # THE anti-hallucination invariant. Phase 6 chunks this content and Phase 6's
    # validator requires each excerpt to be a literal substring of PageData.text.
    # If a block were rewritten rather than merely selected, a genuinely correct
    # answer would fail the substring gate and the agent would return null.
    page = build_page_data(CLEAN_URL, CLEAN_URL, 200, clean_html)
    for block in extract_clean_text(clean_html).split(BLOCK_SEPARATOR):
        assert block in page.text, f"block is not verbatim in page.text: {block[:60]!r}"


def test_substring_invariant_also_holds_for_malformed_markup(messy_html):
    page = build_page_data(MESSY_URL, MESSY_URL, 200, messy_html)
    for block in extract_clean_text(messy_html).split(BLOCK_SEPARATOR):
        if block:
            assert block in page.text


def test_clean_content_does_not_mutate_the_callers_tree(clean_html):
    # The SEO extractor still needs the nav links that content extraction removes.
    soup = parse_html(clean_html)
    before = len(soup.find_all("a"))
    extract_clean_content(soup)
    assert len(soup.find_all("a")) == before


@pytest.mark.parametrize("raw", ["", "<html></html>", "<p>hi</p>"])
def test_clean_content_handles_pages_with_no_prose(raw):
    assert isinstance(extract_clean_content(parse_html(raw)), str)


def test_clean_content_survives_boilerplate_nested_inside_boilerplate():
    # Regression: find_all returns a snapshot, so decomposing the outer <nav>
    # destroys the inner .sidebar that is still queued for inspection. Touching a
    # destroyed tag's attrs raises AttributeError. Real sites nest chrome inside
    # chrome constantly; this crashed on live pages while every fixture passed.
    html = """
      <html><body>
        <nav class="site-nav" role="navigation">
          <div class="sidebar" id="site-footer-links">
            <div class="breadcrumb"><a href="/">Home</a></div>
          </div>
        </nav>
        <main><p>The real content of the page lives here and must survive.</p></main>
      </body></html>
    """
    content = extract_clean_content(parse_html(html))
    assert "The real content of the page lives here and must survive." in content
    assert "Home" not in content


# ---------------------------------------------------------------------------
# build_page_data — the Phase 2 -> Phase 4 seam
# ---------------------------------------------------------------------------


def test_build_page_data_returns_a_fully_populated_contract(clean_html):
    page = build_page_data(
        url="https://ridgeline.example",
        final_url=CLEAN_URL,
        status_code=200,
        html=clean_html,
    )

    assert isinstance(page, PageData)
    assert page.url == "https://ridgeline.example"
    assert page.final_url == CLEAN_URL
    assert page.status_code == 200
    assert page.title == "Ridgeline Coffee Roasters — Small-batch coffee in Portland"
    assert page.canonical == "https://ridgeline.example/"
    assert page.robots_meta == ["index", "follow"]
    assert page.headings["h1"] == ["Small-batch coffee, roasted in Portland"]
    assert len(page.images) == 2
    assert page.structured_data[0]["@type"] == "LocalBusiness"
    assert page.html == clean_html
    assert page.text
    assert len(page.content_hash) == 64


def test_build_page_data_resolves_links_against_final_url_not_requested_url():
    # After a redirect, relative hrefs resolve against where we landed. Using the
    # requested URL would produce links to a host the page never referenced.
    html = '<html><body><a href="/about">About</a></body></html>'
    page = build_page_data(
        url="https://old.example/start",
        final_url="https://new.example/landing",
        status_code=200,
        html=html,
    )
    assert page.links[0].href == "https://new.example/about"
    assert page.links[0].is_internal is True


def test_build_page_data_tolerates_an_empty_body():
    page = build_page_data("https://x.example/", "https://x.example/", 200, "")
    assert page.text == ""
    assert page.title is None
    assert page.images == []
    assert page.links == []
    assert page.structured_data == []


# ---------------------------------------------------------------------------
# X-Robots-Tag response header — Priority 4 fix
#
# A server can direct noindex purely at the HTTP layer, invisible in the page's
# own HTML. build_page_data merges it into the one existing, Phase-1-locked
# robots_meta field rather than adding a new one.
# ---------------------------------------------------------------------------


def test_parse_x_robots_tag_handles_plain_directives():
    from app.extraction.seo_extractor import parse_x_robots_tag

    assert parse_x_robots_tag(["noindex"]) == ["noindex"]
    assert parse_x_robots_tag(["noindex, nofollow"]) == ["noindex", "nofollow"]
    assert parse_x_robots_tag(["NOINDEX"]) == ["noindex"]


def test_parse_x_robots_tag_strips_a_bot_name_scope():
    from app.extraction.seo_extractor import parse_x_robots_tag

    # Per the header's own spec, an optional leading bot name scopes every
    # directive that follows it, comma-separated list included.
    assert parse_x_robots_tag(["googlebot: noindex"]) == ["noindex"]
    assert parse_x_robots_tag(["googlebot: noindex, nofollow"]) == ["noindex", "nofollow"]


def test_parse_x_robots_tag_merges_multiple_header_occurrences_without_duplicates():
    from app.extraction.seo_extractor import parse_x_robots_tag

    assert parse_x_robots_tag(["noindex", "googlebot: noindex, nofollow"]) == [
        "noindex",
        "nofollow",
    ]


def test_parse_x_robots_tag_of_nothing_is_empty():
    from app.extraction.seo_extractor import parse_x_robots_tag

    assert parse_x_robots_tag([]) == []
    assert parse_x_robots_tag(["", None]) == []  # tolerate a stray empty/None entry


def test_build_page_data_merges_header_noindex_into_robots_meta():
    page = build_page_data(
        "https://x.example/",
        "https://x.example/",
        200,
        "<html><head><title>A page with no HTML robots meta tag at all</title></head></html>",
        x_robots_tag=["noindex"],
    )
    assert "noindex" in page.robots_meta


def test_build_page_data_merges_header_and_html_directives_without_duplicating():
    page = build_page_data(
        "https://x.example/",
        "https://x.example/",
        200,
        '<html><head><meta name="robots" content="nofollow"></head></html>',
        x_robots_tag=["nofollow, noindex"],
    )
    assert page.robots_meta.count("nofollow") == 1
    assert "noindex" in page.robots_meta


def test_build_page_data_with_no_x_robots_tag_argument_is_unaffected():
    # Every call site that predates this parameter must behave identically.
    page = build_page_data(
        "https://x.example/",
        "https://x.example/",
        200,
        '<html><head><meta name="robots" content="noindex"></head></html>',
    )
    assert page.robots_meta == ["noindex"]


def test_build_page_data_never_raises_on_malformed_input(messy_html):
    page = build_page_data(MESSY_URL, MESSY_URL, 200, messy_html)
    assert page.title == ""
    assert page.robots_meta == ["noindex", "nofollow"]
    assert len(page.structured_data) == 1


def test_content_hash_detects_duplicate_prose_across_different_html():
    # Same visible text, different markup and different build ids: duplicate
    # content in the sense an SEO audit means. Hashing the HTML would miss it.
    a = '<html><body><p>Identical prose on both pages here.</p><!-- build 1 --></body></html>'
    b = '<html><body><div><p>Identical prose on both pages here.</p></div><!-- build 2 --></body></html>'
    page_a = build_page_data("https://x.example/a", "https://x.example/a", 200, a)
    page_b = build_page_data("https://x.example/b", "https://x.example/b", 200, b)
    assert page_a.content_hash == page_b.content_hash


def test_content_hash_differs_for_different_prose():
    assert content_hash("one") != content_hash("two")
    assert content_hash("") == content_hash("")
