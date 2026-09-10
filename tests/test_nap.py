"""Tests for the Phase 5 NAP extraction and consistency check.

The plan calls this the highest-subtlety piece, and names the two failures that
matter. Both have dedicated tests here:

* ``+91 98765 43210`` and ``919876543210`` **must** compare equal.
* Two genuinely different addresses **must not** be merged.

The second is the one worth guarding hardest. A missed inconsistency is a gap; a
falsely merged pair produces a confident ``consistent`` verdict on a site that has a
real problem, which is a claim the evidence does not support.
"""

from __future__ import annotations

import pathlib

import pytest

from app.agents.nap_agent import (
    MIN_PAGES_FOR_VERDICT,
    compare_field,
    describe_disagreement,
    filter_candidates_with_llm,
    run_nap_check,
)
from app.extraction.nap_extractor import extract_nap_candidates, identify_target_business
from app.extraction.normalize import (
    address_subsumes,
    differs_only_by_legal_suffix,
    normalize_address,
    normalize_name,
    normalize_phone,
    strip_legal_suffix,
)
from app.extraction.seo_extractor import build_page_data
from app.models.nap import NAPValue

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture_page(name: str, url: str):
    return build_page_data(url, url, 200, (FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def home():
    return fixture_page("clean_page.html", "https://ridgeline.example/")


@pytest.fixture
def contact():
    return fixture_page("nap_contact_page.html", "https://ridgeline.example/contact")


@pytest.fixture
def wholesale():
    return fixture_page("nap_inconsistent_page.html", "https://ridgeline.example/wholesale")


def value(page: str, raw: str, normalized: str, source: str = "visible_text") -> NAPValue:
    return NAPValue(page=page, raw_value=raw, normalized_value=normalized, source=source)


# ---------------------------------------------------------------------------
# Phone normalization — the plan's worked example
# ---------------------------------------------------------------------------


def test_the_plans_exact_phone_example_normalizes_to_one_value():
    assert normalize_phone("+91 98765 43210") == normalize_phone("919876543210")
    assert normalize_phone("+91 98765 43210") == "9876543210"


@pytest.mark.parametrize(
    "raw",
    [
        "+1 503-555-0147",
        "(503) 555-0147",
        "503.555.0147",
        "5035550147",
        "+1 (503) 555 0147",
        "tel: +1-503-555-0147",
    ],
)
def test_all_spellings_of_one_number_agree(raw):
    assert normalize_phone(raw) == "5035550147"


def test_country_code_is_never_guessed_at():
    # Comparing the last ten digits means we never have to decide whether a leading
    # "91" is India's dial code or part of the subscriber number. Guessing wrong is
    # what merges two genuinely different numbers.
    assert normalize_phone("+91 98765 43210") == normalize_phone("098765 43210")


def test_two_different_numbers_never_merge():
    assert normalize_phone("+1 503-555-0147") != normalize_phone("+1 503-555-0199")


@pytest.mark.parametrize(
    "raw",
    ["", "12345", "2026", "$1,299.00", "1" * 20, "no digits here"],
)
def test_implausible_phone_strings_are_rejected(raw):
    assert normalize_phone(raw) == ""


# ---------------------------------------------------------------------------
# Address normalization — the must-not-merge case
# ---------------------------------------------------------------------------


def test_abbreviations_expand_in_one_direction():
    assert normalize_address("412 SE Ankeny St") == "412 southeast ankeny street"
    assert normalize_address("412 Southeast Ankeny Street") == "412 southeast ankeny street"
    assert normalize_address("18 Harbour Rd.") == "18 harbour road"


def test_two_genuinely_different_addresses_are_never_merged():
    # The plan's explicit requirement. Same street, different suite: two tenants.
    a = normalize_address("100 Main St, Suite 200, Portland, OR")
    b = normalize_address("100 Main St, Suite 300, Portland, OR")
    assert a != b
    assert not address_subsumes(a, b)
    assert not address_subsumes(b, a)


def test_suite_numbers_are_normalized_but_never_removed():
    assert "suite" in normalize_address("100 Main St, Ste 200")
    assert "200" in normalize_address("100 Main St, Ste 200")


def test_locality_is_never_dropped():
    # Same street name in two towns is two different places.
    a = normalize_address("12 Church Lane, Bristol")
    b = normalize_address("12 Church Lane, Leeds")
    assert a != b
    assert not address_subsumes(a, b)


def test_saint_is_not_expanded_into_street():
    # "St" means Street after a house number and Saint at the start of a name.
    assert normalize_address("St Louis Avenue") == "st louis avenue"
    assert "street louis" not in normalize_address("St Louis Avenue")


def test_a_truncated_address_is_recognized_as_the_same_place():
    full = normalize_address("412 SE Ankeny St, Portland, OR 97214")
    street_only = normalize_address("412 SE Ankeny St")
    assert address_subsumes(full, street_only)


def test_subsumption_refuses_to_absorb_a_suite_number():
    # This is the boundary that keeps subsumption safe: the extra tokens include a
    # unit designator, so the shorter form is NOT treated as the same location.
    with_suite = normalize_address("412 SE Ankeny St, Suite 300, Portland, OR 97214")
    street_only = normalize_address("412 SE Ankeny St")
    assert not address_subsumes(with_suite, street_only)


def test_subsumption_requires_a_prefix_not_merely_a_substring():
    assert not address_subsumes(
        normalize_address("512 SE Ankeny St, Portland"), normalize_address("412 SE Ankeny St")
    )


# ---------------------------------------------------------------------------
# Name normalization — flag, do not merge
# ---------------------------------------------------------------------------


def test_names_compare_case_and_punctuation_insensitively():
    assert normalize_name("Ridgeline Coffee Roasters") == normalize_name("ridgeline coffee roasters")
    assert normalize_name("Acme, Inc.") == normalize_name("Acme Inc")


def test_possessive_apostrophe_is_removed_not_turned_into_a_word_break():
    # Regression: found live on a real e-commerce site. The general punctuation
    # rule turns "'" into a space, so "Zabar's" became "zabar s" -- two tokens,
    # with a spurious extra "s" word -- rather than being recognized as the same
    # name a page that simply drops the apostrophe would write as "Zabars".
    assert normalize_name("Zabar's") == "zabars"
    assert normalize_name("Zabar's") == normalize_name("Zabars")
    # The typographic "smart quote" (U+2019) real sites overwhelmingly use in
    # body copy must be treated identically to the ASCII straight quote.
    assert normalize_name("Zabar’s") == normalize_name("Zabar's")


def test_legal_suffix_difference_is_flagged_not_merged():
    # Acme Inc and Acme LLC are different legal entities. Merging them would hide a
    # real signal, so they stay distinct and are explained instead.
    inc = normalize_name("Acme Inc")
    llc = normalize_name("Acme LLC")
    assert inc != llc
    assert differs_only_by_legal_suffix(inc, llc)
    assert strip_legal_suffix(inc) == strip_legal_suffix(llc) == "acme"


def test_genuinely_different_names_are_not_a_suffix_difference():
    assert not differs_only_by_legal_suffix(
        normalize_name("Acme Inc"), normalize_name("Bertha's Bakery")
    )


# ---------------------------------------------------------------------------
# Extraction, and its source priority
# ---------------------------------------------------------------------------


def test_json_ld_business_details_are_extracted(home):
    found = extract_nap_candidates(home)

    assert found["name"][0].raw_value == "Ridgeline Coffee Roasters"
    assert found["name"][0].source == "json_ld"
    assert found["address"][0].source == "json_ld"
    assert "97214" in found["address"][0].raw_value
    assert normalize_phone("+1 503-555-0147") in {v.normalized_value for v in found["phone"]}


def test_tel_links_are_extracted_as_their_own_source(home):
    sources = {v.source for v in extract_nap_candidates(home)["phone"]}
    assert "tel_link" in sources


def test_microdata_business_details_are_extracted():
    page = fixture_page("microdata_page.html", "https://harbourdental.example/contact")
    found = extract_nap_candidates(page)

    assert found["name"][0].raw_value == "Harbour Dental Practice"
    assert found["name"][0].source == "microdata"
    assert "18 Harbour Road" in found["address"][0].raw_value
    assert found["phone"][0].normalized_value == normalize_phone("0117 496 0182")


def test_visible_text_is_used_only_when_no_precise_source_exists(home, contact):
    # The home page states its address in full in JSON-LD, so the fragment the
    # visible-text regex also matches is discarded — otherwise one page would appear
    # to disagree with itself.
    assert all(v.source != "visible_text" for v in extract_nap_candidates(home)["address"])

    # The contact page has no structured markup, so visible text is all there is.
    contact_address = extract_nap_candidates(contact)["address"]
    assert contact_address and contact_address[0].source == "visible_text"


def test_copyright_line_yields_a_name_from_visible_text(contact):
    names = extract_nap_candidates(contact)["name"]
    assert names
    # The capture stops at the sentence end, not running on into the boilerplate.
    assert names[0].raw_value == "Ridgeline Coffee Roasters"
    assert "All rights" not in names[0].raw_value


@pytest.mark.parametrize(
    "text",
    [
        "© 2026 All Rights Reserved",
        "© 2026 All Rights Reserved Worldwide.",
        "© 2026-2027 Some Rights Reserved",
        "Copyright 2026 All Rights Reserved",
    ],
)
def test_rights_reservation_boilerplate_is_never_captured_as_a_name(text):
    html = f"<html><body><footer><p>{text}</p></footer></body></html>"
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    assert extract_nap_candidates(page)["name"] == []


def test_a_real_name_containing_a_boilerplate_word_is_still_captured():
    # The denylist rejects a capture only when EVERY word in it is boilerplate;
    # a genuine name that happens to share a word must not be swept up with it.
    html = '<html><body><footer><p>© 2026 Worldwide Fabrics Inc</p></footer></body></html>'
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    names = extract_nap_candidates(page)["name"]
    assert any("Worldwide Fabrics" in v.raw_value for v in names)


def test_non_business_schema_types_are_ignored():
    html = """<html><head><script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Product","name":"Widget",
     "telephone":"+1 555-000-1111"}
    </script></head><body><p>A product page.</p></body></html>"""
    page = build_page_data("https://x.example/p", "https://x.example/p", 200, html)
    found = extract_nap_candidates(page)
    # A Product's phone number may belong to a manufacturer, not this business.
    assert found["name"] == []
    assert found["phone"] == []


def test_every_extracted_value_appears_literally_on_the_page(home, contact, wholesale):
    # The property Phase 7's validator relies on: nothing is inferred or completed.
    # Structured-data candidates are joined from several JSON fields (e.g. address
    # parts joined with ", "), so a json_ld/microdata raw_value is checked against
    # its source dict's own field values, not against the page's raw HTML/text.
    import json as _json

    for page in (home, contact, wholesale):
        haystack = page.text + page.html
        structured_text = _json.dumps(page.structured_data)
        for values in extract_nap_candidates(page).values():
            for candidate in values:
                if candidate.source in ("json_ld", "microdata"):
                    parts = candidate.raw_value.split(", ")
                    assert all(part in structured_text for part in parts), candidate
                else:
                    assert candidate.raw_value in haystack


def test_legal_document_pages_skip_visible_text_extraction_entirely():
    # Regression: found live on stripe.com's real Terms of Service pages.
    # Numbered legal clauses ("11 Severability. If any court...", cross-
    # references like "25.10.3-101") are adversarial to every visible-text shape
    # heuristic at once -- "court" is a legitimate street suffix, and clause
    # citation numbers pass the phone-shape gates. Rather than chasing every
    # possible legal-numbering collision, /legal/ pages skip visible-text NAP
    # extraction entirely: real contact info is never meaningfully stated in ToS
    # boilerplate anyway.
    html = (
        "<html><body><main>"
        "<p>11 Severability. If any court of competent jurisdiction holds that "
        "any provision is invalid, the remainder shall continue in effect.</p>"
        "<p>See Section 25.10.3-101 for the governing arbitration procedure.</p>"
        "</main></body></html>"
    )
    page = build_page_data(
        "https://x.example/legal/terms-of-service",
        "https://x.example/legal/terms-of-service",
        200,
        html,
    )
    found = extract_nap_candidates(page)
    assert found["address"] == []
    assert found["phone"] == []


@pytest.mark.parametrize(
    "url",
    [
        "https://x.example/legal/tos",
        "https://x.example/terms",
        "https://x.example/terms-of-service",
        "https://x.example/privacy",
        "https://x.example/privacy-policy",
        "https://x.example/cookie-policy",
        "https://x.example/gdpr",
        "https://x.example/eula",
    ],
)
def test_common_legal_page_url_conventions_are_recognized(url):
    from app.extraction.nap_extractor import _is_legal_document_url

    assert _is_legal_document_url(url) is True


def test_ordinary_pages_are_never_mistaken_for_legal_pages():
    from app.extraction.nap_extractor import _is_legal_document_url

    assert _is_legal_document_url("https://x.example/") is False
    assert _is_legal_document_url("https://x.example/about") is False
    assert _is_legal_document_url("https://x.example/contact") is False
    # A path that merely contains a legal-adjacent substring elsewhere must not
    # false-positive -- only real path segments count.
    assert _is_legal_document_url("https://x.example/legality-of-vpns") is False


def test_legal_page_skip_does_not_affect_structured_or_tel_link_sources():
    # A genuine tel: link or JSON-LD block on a legal page is still a deliberate,
    # structured statement and must still be captured -- only the loose
    # visible-text regex tier is skipped.
    html = (
        '<html><head><script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"LocalBusiness","name":"Acme Corp",'
        '"telephone":"+1 503-555-0147"}'
        "</script></head><body><main>"
        '<p>Call us: <a href="tel:+15035550147">+1 503-555-0147</a></p>'
        "<p>11 Severability. If any court finds otherwise...</p>"
        "</main></body></html>"
    )
    page = build_page_data(
        "https://x.example/legal/terms", "https://x.example/legal/terms", 200, html
    )
    found = extract_nap_candidates(page)
    assert {v.source for v in found["phone"]} == {"json_ld", "tel_link"}
    assert any(v.raw_value == "Acme Corp" for v in found["name"])


def test_copyright_year_near_a_street_type_word_is_not_captured_as_an_address():
    # Regression: found live on a real Shopify store's footer. A copyright year
    # ("2026") sitting a few words before a "Close" button label matched
    # ADDRESS_PATTERN, because "close" is both a legitimate street-type suffix and
    # an ordinary English word — the same collision class as the phone/date bug
    # above, this time for addresses.
    html = "<html><body><footer><p>Copyright 2026 Terms of service Close</p></footer></body></html>"
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    assert extract_nap_candidates(page)["address"] == []


def test_a_long_run_of_prose_ending_in_a_street_word_is_not_captured_as_an_address():
    # Regression: found live on the same page. A marketing tagline ending in
    # "way" ("Better things in a better way") with a stray list-count "32" in
    # front of it matched the old 5-filler-word gap; tightened to 3.
    html = (
        "<html><body><footer><p>32 Better things in a better way</p></footer></body></html>"
    )
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    assert extract_nap_candidates(page)["address"] == []


def test_a_real_address_with_a_non_year_house_number_is_still_captured():
    html = "<html><body><footer><p>8484 Wilshire Blvd, Los Angeles, CA</p></footer></body></html>"
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    addresses = extract_nap_candidates(page)["address"]
    assert any("8484 Wilshire Blvd" in v.raw_value for v in addresses)


def test_a_retail_quantity_label_is_never_captured_as_an_address():
    # Regression: found live on a real e-commerce grocery site. "ct" is both the
    # street-suffix abbreviation for "Court" and common retail shorthand for
    # "count" ("4 ct" = a 4-pack). Every product page repeated "4 ct" and it was
    # captured as address "4 Court" with confidence 1.0 -- not merely noisy
    # evidence but a confidently WRONG accepted value. Fixed by requiring a
    # plausible house number to have at least 2 digits.
    html = "<html><body><main><p>Package contains 4 ct of assorted bars.</p></main></body></html>"
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    assert extract_nap_candidates(page)["address"] == []


def test_two_digit_house_numbers_are_still_captured():
    # The fix must not overcorrect: a genuine, common 2-digit house number stays.
    html = "<html><body><footer><p>18 Harbour Road, Bristol</p></footer></body></html>"
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    addresses = extract_nap_candidates(page)["address"]
    assert any("18 Harbour Road" in v.raw_value for v in addresses)


def test_a_plus_prefixed_solid_digit_run_is_recognized_as_a_phone_number():
    # Regression: found live on a real site's WhatsApp contact number, repeated
    # identically on 18 of 20 pages as "+919390111761" -- zero internal spacing,
    # which collapses to a single digit group and was silently rejected by the
    # "at least 2 groups" rule everywhere except the one page that happened to
    # add a space after the country code. This is an extremely common real-world
    # format (WhatsApp/mobile numbers across South Asia and elsewhere), and the
    # leading "+" is itself a strong, unambiguous phone signal on its own.
    html = (
        "<html><body><footer><p>WhatsApp us: +919390111761 anytime.</p></footer>"
        "</body></html>"
    )
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    phones = extract_nap_candidates(page)["phone"]
    assert any(v.normalized_value == "9390111761" for v in phones)


def test_a_solid_digit_run_without_a_leading_plus_is_still_rejected():
    # The exception is scoped narrowly to a leading "+" specifically -- a solid
    # digit run inside other punctuation (parens, no prefix) is not granted the
    # same benefit of the doubt, since only "+" is an unambiguous phone marker.
    from app.extraction.nap_extractor import _looks_phone_shaped

    assert _looks_phone_shaped("(919390111761)") is False


def test_dates_and_year_ranges_are_not_captured_as_phone_numbers():
    # Found via a live crawl of python.org: "2026-09-17" and "2001-2026" both pass
    # the raw digit-count gate and both contain a hyphen, so a naive phone-shape
    # regex reports them as candidate phone numbers on a page that has none.
    html = (
        "<html><body><p>Copyright 2001-2026 Python Software Foundation.</p>"
        "<p>Released on 2026-09-17.</p>"
        "<p>Fibonacci: 1 1 2 3 5 8 13 21 34 55 89 144 233</p></body></html>"
    )
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    assert extract_nap_candidates(page)["phone"] == []


def test_two_four_digit_groups_are_never_captured_as_phone_numbers():
    # Regression: found live on python.org via app.main's full pipeline, not
    # a fixture. A copyright-year and a "last updated" date, concatenated by
    # whitespace normalization into "2026 2026-09-17", survived the earlier
    # date-shape and single-digit-group gates because it has 4 digit groups, not
    # 2. It produced a false "phone: consistent" verdict on a site with no phone
    # number at all -- a wrong top-level claim, not just noisy evidence.
    html = (
        "<html><body><p>Copyright 2026. Last updated 2026-09-17.</p>"
        "<p>Archive: 2001-2026, 1998-2005, 2010-2015.</p></body></html>"
    )
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    assert extract_nap_candidates(page)["phone"] == []


def test_copyright_capture_keeps_a_typographic_apostrophe_possessive():
    # Regression: found live on a real e-commerce site's product pages, which
    # write "© 2026 Zabar’s." with the typographic smart-quote apostrophe (’),
    # not the ASCII straight quote. Without ’ in the capture's character class,
    # the match stopped right before it, silently truncating "Zabar's" to "Zabar".
    html = "<html><body><footer><p>© 2026 Zabar’s. All Rights Reserved.</p></footer></body></html>"
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    names = extract_nap_candidates(page)["name"]
    assert names
    assert names[0].raw_value == "Zabar’s"
    assert names[0].raw_value != "Zabar"


def test_copyright_regex_ignores_the_word_copyright_used_as_a_common_noun():
    # Regression: found live on a real Shopify store's legal-boilerplate footer
    # during the Phase 10 sweep. Without requiring a year (or the bare "©" glyph),
    # "Digital Millennium Copyright Act" and "copyright infringement claims"
    # matched the word "copyright" and captured the capitalized phrase that
    # happened to follow ("Act", "Infringement Dispute Resolution") as if it were
    # the business name -- contaminating the Q2 name comparison with garbage that
    # then produced a false "inconsistent" verdict.
    html = (
        "<html><body><footer><p>Please notify us pursuant to the Digital "
        "Millennium Copyright Act (DMCA) if you believe a copyright "
        "infringement claim is warranted. Infringement Dispute Resolution "
        "Procedure details are available on request.</p></footer></body></html>"
    )
    page = build_page_data("https://x.example/", "https://x.example/", 200, html)
    assert extract_nap_candidates(page)["name"] == []


def test_extraction_of_an_empty_page_yields_nothing():
    page = build_page_data("https://x.example/", "https://x.example/", 200, "")
    assert extract_nap_candidates(page) == {"name": [], "address": [], "phone": []}


# ---------------------------------------------------------------------------
# Target business identification — Priority 1 fix
#
# The generalization risk: without knowing which business a site's audit is
# actually about, every business-typed schema block anywhere on the site —
# a payment processor, a shipping partner, a sister brand — pools into the
# same NAP comparison as the site's own business.
# ---------------------------------------------------------------------------


def _schema_page(url: str, blocks: list[dict]) -> Any:
    import json

    scripts = "".join(
        f'<script type="application/ld+json">{json.dumps(b)}</script>' for b in blocks
    )
    html = f"<html><head>{scripts}</head><body><p>content</p></body></html>"
    return build_page_data(url, url, 200, html)


def test_identify_target_business_prefers_the_homepage():
    home = _schema_page(
        "https://x.example/",
        [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters"}],
    )
    # A different, more frequent name appears on two other, deeper pages — the
    # homepage's own statement of identity must still win.
    about = _schema_page(
        "https://x.example/about", [{"@type": "Organization", "name": "Some Other Name"}]
    )
    contact = _schema_page(
        "https://x.example/contact", [{"@type": "Organization", "name": "Some Other Name"}]
    )
    assert identify_target_business([home, about, contact]) == "ridgeline coffee roasters"


def test_identify_target_business_falls_back_to_majority_vote_with_no_homepage_schema():
    home = _schema_page("https://x.example/", [])  # no schema at all on the homepage
    a = _schema_page(
        "https://x.example/a", [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters"}]
    )
    b = _schema_page(
        "https://x.example/b", [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters"}]
    )
    c = _schema_page("https://x.example/c", [{"@type": "LocalBusiness", "name": "Rare Name"}])
    assert identify_target_business([home, a, b, c]) == "ridgeline coffee roasters"


def test_identify_target_business_returns_none_with_no_schema_anywhere():
    page = build_page_data("https://x.example/", "https://x.example/", 200, "<html></html>")
    assert identify_target_business([page]) is None


def test_a_non_root_page_is_never_treated_as_the_homepage_even_if_shortest():
    # The crawl seed was /shop; root ("/") was never fetched at all. /shop being
    # the shortest available path must NOT grant it homepage priority — it is
    # just another non-root page, and majority vote must decide instead.
    shop = _schema_page(
        "https://x.example/shop", [{"@type": "Organization", "name": "Shop Page Business"}]
    )
    about1 = _schema_page(
        "https://x.example/shop/about-page",
        [{"@type": "Organization", "name": "Real Site Business"}],
    )
    about2 = _schema_page(
        "https://x.example/shop/contact-page",
        [{"@type": "Organization", "name": "Real Site Business"}],
    )
    assert identify_target_business([shop, about1, about2]) == "real site business"


def test_root_with_trailing_query_string_still_counts_as_the_homepage():
    home = _schema_page(
        "https://x.example/?utm_source=test",
        [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters"}],
    )
    other = _schema_page(
        "https://x.example/other", [{"@type": "LocalBusiness", "name": "Some Other Name"}]
    )
    assert identify_target_business([home, other]) == "ridgeline coffee roasters"


def test_root_path_with_no_trailing_slash_still_counts_as_the_homepage():
    home = _schema_page(
        "https://x.example", [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters"}]
    )
    other = _schema_page(
        "https://x.example/other", [{"@type": "LocalBusiness", "name": "Some Other Name"}]
    )
    assert identify_target_business([home, other]) == "ridgeline coffee roasters"


def test_third_party_business_schema_is_excluded_from_the_target_businesss_nap(home, contact):
    # The real risk scenario: a page embeds a genuinely different business's
    # schema (a payment processor, a shipping partner) alongside the site's own.
    # That third party's phone/address must never enter this site's comparison.
    third_party_page = _schema_page(
        "https://ridgeline.example/checkout",
        [
            {
                "@type": "LocalBusiness",
                "name": "Ridgeline Coffee Roasters",
                "telephone": "+1 503-555-0147",
            },
            {
                "@type": "Organization",
                "name": "Acme Payments Inc",
                "telephone": "+1 800-555-9999",
            },
        ],
    )
    pages = [home, contact, third_party_page]
    target = identify_target_business(pages)
    assert target == "ridgeline coffee roasters"

    found = extract_nap_candidates(third_party_page, target_business=target)
    phones = {v.raw_value for v in found["phone"]}
    names = {v.raw_value for v in found["name"]}
    assert "+1 503-555-0147" in phones
    assert "+1 800-555-9999" not in phones  # Acme Payments' number is excluded
    assert "Acme Payments Inc" not in names


def test_third_party_exclusion_prevents_a_false_inconsistent_nap_verdict(home, contact):
    third_party_page = _schema_page(
        "https://ridgeline.example/checkout",
        [
            {
                "@type": "LocalBusiness",
                "name": "Ridgeline Coffee Roasters",
                "telephone": "+1 503-555-0147",
            },
            {"@type": "Organization", "name": "Acme Payments Inc", "telephone": "+1 800-555-9999"},
        ],
    )
    results = run_nap_check([home, contact, third_party_page])
    phone = next(c for c in results if c.field == "phone")
    # Without entity filtering this would be a false "inconsistent" driven by a
    # payment processor's phone number that has nothing to do with the site.
    assert phone.verdict == "consistent"
    name = next(c for c in results if c.field == "name")
    assert name.verdict == "consistent"


def test_legal_suffix_variant_still_counts_as_the_target_business():
    # The Phase 5 legal-suffix-disagreement feature must survive entity
    # filtering: "Ridgeline Coffee Roasters LLC" is the SAME business as
    # "Ridgeline Coffee Roasters" with a legal suffix, not a different one, and
    # must not be silently dropped before the disagreement can be flagged.
    home = _schema_page(
        "https://x.example/", [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters"}]
    )
    other = _schema_page(
        "https://x.example/wholesale",
        [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters LLC", "telephone": "+1 503-555-0199"}],
    )
    target = identify_target_business([home, other])
    found = extract_nap_candidates(other, target_business=target)
    assert any(v.raw_value == "Ridgeline Coffee Roasters LLC" for v in found["name"])
    assert any(v.raw_value == "+1 503-555-0199" for v in found["phone"])


def test_target_business_filtering_never_touches_a_block_with_no_name():
    # An address-only PostalAddress fragment with no accompanying name cannot be
    # checked against the target, so it is kept — the existing conservative
    # default (accept what cannot be disproven), unaffected by this feature.
    page = _schema_page(
        "https://x.example/",
        [{"@type": "LocalBusiness", "address": "100 Main St, Springfield"}],
    )
    found = extract_nap_candidates(page, target_business="some other business")
    assert found["address"] != []


def test_identify_target_business_is_robust_to_an_unreadable_page():
    class Broken:
        final_url = "https://x.example/bad"

        @property
        def html(self):
            raise RuntimeError("boom")

        @property
        def structured_data(self):
            raise RuntimeError("boom")

    good = _schema_page(
        "https://x.example/", [{"@type": "LocalBusiness", "name": "Ridgeline Coffee Roasters"}]
    )
    assert identify_target_business([Broken(), good]) == "ridgeline coffee roasters"


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_consistent_site_reports_consistent_on_every_field(home, contact):
    results = {c.field: c for c in run_nap_check([home, contact])}

    assert [c.verdict for c in results.values()] == ["consistent"] * 3
    assert results["phone"].normalized_values == ["5035550147"]
    assert results["phone"].confidence == 1.0
    assert len(results["phone"].pages_compared) == 2


def test_a_truncated_address_does_not_produce_a_false_inconsistency(home, contact):
    # The home page's JSON-LD gives the full address; the contact page writes only
    # the street line. Same place, two levels of detail. Reporting this as an
    # inconsistency would be a confident claim the evidence does not support.
    address = next(c for c in run_nap_check([home, contact]) if c.field == "address")
    assert address.verdict == "consistent"


def test_genuine_mismatches_are_reported(home, contact, wholesale):
    results = {c.field: c for c in run_nap_check([home, contact, wholesale])}

    assert results["phone"].verdict == "inconsistent"
    assert set(results["phone"].normalized_values) == {"5035550147", "5035550199"}

    # Suite 300 is a different unit and must survive as its own value.
    assert results["address"].verdict == "inconsistent"
    assert any("suite 300" in v for v in results["address"].normalized_values)

    assert results["name"].verdict == "inconsistent"


def test_consistent_when_one_page_carries_consistent_values(home):
    results = {c.field: c for c in run_nap_check([home])}
    for comparison in results.values():
        assert comparison.verdict == "consistent"
        assert comparison.confidence == 1.0


def test_values_from_one_page_are_called_consistent():
    values = [
        value("https://x.example/", "555-000-1111", "5550001111", "json_ld"),
        value("https://x.example/", "(555) 000-1111", "5550001111", "tel_link"),
        value("https://x.example/", "555.000.1111", "5550001111"),
    ]
    comparison = compare_field("phone", values)
    assert comparison.verdict == "consistent"
    assert comparison.confidence == 1.0
    assert len(comparison.pages_compared) >= MIN_PAGES_FOR_VERDICT


def test_conflicting_values_from_one_page_are_called_inconsistent():
    values = [
        value("https://x.example/", "555-000-1111", "5550001111", "json_ld"),
        value("https://x.example/", "(555) 999-2222", "5559992222", "tel_link"),
    ]
    comparison = compare_field("phone", values)
    assert comparison.verdict == "inconsistent"
    assert len(comparison.evidence) == 2


def test_a_field_absent_everywhere_still_appears_in_the_report():
    page = build_page_data("https://x.example/", "https://x.example/", 200, "<html></html>")
    results = run_nap_check([page, page])
    assert [c.field for c in results] == ["name", "address", "phone"]
    assert all(c.verdict == "insufficient_data" for c in results)


def test_confidence_is_the_majority_share_and_is_deterministic():
    values = [
        value("https://x.example/a", "555-000-1111", "5550001111"),
        value("https://x.example/b", "555-000-1111", "5550001111"),
        value("https://x.example/c", "555-000-2222", "5550002222"),
    ]
    comparison = compare_field("phone", values)
    assert comparison.verdict == "inconsistent"
    assert comparison.confidence == pytest.approx(2 / 3, abs=1e-4)
    # Same input, same number, every time.
    assert compare_field("phone", values).confidence == comparison.confidence


def test_evidence_carries_every_value_that_was_compared():
    values = [
        value("https://x.example/a", "555-000-1111", "5550001111", "json_ld"),
        value("https://x.example/b", "555-000-2222", "5550002222", "tel_link"),
    ]
    comparison = compare_field("phone", values)
    assert comparison.evidence == values
    assert {e.source for e in comparison.evidence} == {"json_ld", "tel_link"}


def test_representative_raw_value_prefers_the_most_authoritative_source():
    values = [
        value("https://x.example/a", "footer spelling 555-000-1111", "5550001111"),
        value("https://x.example/b", "+1 555-000-1111", "5550001111", "json_ld"),
    ]
    comparison = compare_field("phone", values)
    assert comparison.values == ["+1 555-000-1111"]


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


def test_legal_suffix_disagreement_is_explained_as_such(home, contact, wholesale):
    name = next(c for c in run_nap_check([home, contact, wholesale]) if c.field == "name")
    explanation = describe_disagreement(name)
    assert "legal suffix" in explanation
    assert "merged" in explanation


def test_insufficient_data_explains_what_was_missing():
    empty_page = build_page_data("https://x.example", "https://x.example", 200, "<html><body>No nap data here</body></html>")
    comparison = next(c for c in run_nap_check([empty_page]) if c.field == "phone")
    assert comparison.verdict == "insufficient_data"
    assert "No phone value found" in describe_disagreement(comparison)


# ---------------------------------------------------------------------------
# The optional LLM filter — same structural guarantee as Phase 4
# ---------------------------------------------------------------------------


def _candidates() -> list[NAPValue]:
    return [
        value("https://x.example/a", "+1 555-000-1111", "5550001111", "json_ld"),
        value("https://x.example/b", "+1 555-999-8888", "5559998888"),
    ]


def test_llm_filter_can_reject_a_candidate():
    def decide(prompt: str) -> str:
        return "no" if "555-999-8888" in prompt else "yes"

    kept = filter_candidates_with_llm(_candidates(), decide)
    assert [v.raw_value for v in kept] == ["+1 555-000-1111"]


def test_llm_filter_can_only_remove_never_add_or_edit():
    # Whatever the model returns, the surviving values are the originals unchanged.
    hostile = '{"raw_value": "+1 999-999-9999", "normalized_value": "9999999999"}'
    original = _candidates()
    kept = filter_candidates_with_llm(original, lambda _: hostile)

    assert kept == original
    assert all(v in original for v in kept)
    assert "9999999999" not in {v.normalized_value for v in kept}


def test_ambiguous_llm_answers_keep_the_candidate():
    # The default favours reporting a real value over silently dropping one.
    for answer in ["maybe", "", "I am not sure", "yes, probably"]:
        assert len(filter_candidates_with_llm(_candidates(), lambda _: answer)) == 2


def test_llm_filter_failure_does_not_empty_the_evidence():
    def explode(_prompt: str) -> str:
        raise RuntimeError("rate limited")

    assert filter_candidates_with_llm(_candidates(), explode) == _candidates()


def test_no_llm_configured_returns_candidates_unchanged():
    original = _candidates()
    assert filter_candidates_with_llm(original, None) == original


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_run_nap_check_survives_a_page_that_cannot_be_parsed():
    class Broken:
        final_url = "https://x.example/bad"

        @property
        def html(self):
            raise RuntimeError("boom")

    results = run_nap_check([Broken()])
    assert [c.field for c in results] == ["name", "address", "phone"]


def test_run_nap_check_of_no_pages_returns_three_empty_verdicts():
    results = run_nap_check([])
    assert len(results) == 3
    assert all(c.verdict == "insufficient_data" and c.evidence == [] for c in results)


# ---------------------------------------------------------------------------
# LLM NAP candidate recovery -- found live on ironlocksandlevers.com: a real
# business name ("Iron Locks & Levers") with no JSON-LD/microdata/copyright line
# anywhere, and a real address ("Springville, UT") with no street name, house
# number, or generic street-suffix word at all. Neither is a shape the
# regex/schema tiers can recognize, so both came back insufficient_data with
# zero candidates despite being genuinely, correctly stated on the page.
# ---------------------------------------------------------------------------

import json as _json

from app.extraction.nap_extractor import extract_nap_candidates_via_llm

IRON_LOCKS_HTML = (
    "<html><body>"
    "<header><h1>Iron Locks &amp; Levers</h1></header>"
    "<main><p>Elegant Wrought Iron Door Hardware</p></main>"
    "<footer><p>Contact Us:[Email: sales@Ironlocksandlevers.com]   |   "
    "[Springville, UT]   |   [Phone: 801-919-5730]</p></footer>"
    "</body></html>"
)


def _iron_locks_page(url: str = "https://ironlocksandlevers.com/"):
    return build_page_data(url, url, 200, IRON_LOCKS_HTML)


def test_llm_recovers_a_real_name_and_address_the_regex_tiers_cannot_see():
    page = _iron_locks_page()

    def generate(prompt: str, **_kwargs) -> str:
        assert "Iron Locks" in prompt  # the real page text actually reached it
        return _json.dumps({"name": "Iron Locks & Levers", "address": "Springville, UT"})

    found = extract_nap_candidates_via_llm(page, ["name", "address"], generate)

    assert found["name"][0].raw_value == "Iron Locks & Levers"
    assert found["name"][0].source == "llm"
    assert found["address"][0].raw_value == "Springville, UT"
    assert found["address"][0].source == "llm"


def test_llm_claiming_text_not_actually_on_the_page_is_rejected():
    # The model composing/paraphrasing instead of pointing at real text must
    # never become evidence -- the same guarantee every other source holds.
    page = _iron_locks_page()

    def generate(prompt: str, **_kwargs) -> str:
        return _json.dumps({"name": "Iron Locks and Levers LLC", "address": "123 Fake St"})

    found = extract_nap_candidates_via_llm(page, ["name", "address"], generate)
    assert found == {}


def test_llm_saying_null_for_an_absent_field_is_respected():
    page = _iron_locks_page()

    def generate(prompt: str, **_kwargs) -> str:
        return _json.dumps({"name": "Iron Locks & Levers", "phone": None})

    found = extract_nap_candidates_via_llm(page, ["name", "phone"], generate)
    assert "name" in found
    assert "phone" not in found


def test_llm_nap_exception_degrades_to_no_candidates_never_a_guess():
    def explode(_prompt: str, **_kwargs) -> str:
        raise RuntimeError("rate limited")

    found = extract_nap_candidates_via_llm(_iron_locks_page(), ["name"], explode)
    assert found == {}


def test_llm_nap_unparseable_response_degrades_to_no_candidates():
    found = extract_nap_candidates_via_llm(
        _iron_locks_page(), ["name"], lambda *_a, **_k: "not json at all"
    )
    assert found == {}


def test_run_nap_check_recovers_name_and_address_via_llm_when_regex_finds_nothing():
    page = _iron_locks_page()

    def generate(prompt: str, **_kwargs) -> str:
        return _json.dumps({"name": "Iron Locks & Levers", "address": "Springville, UT"})

    results = run_nap_check([page], generate=generate)
    by_field = {c.field: c for c in results}

    assert any(v.source == "llm" for v in by_field["name"].evidence)
    assert any(v.source == "llm" for v in by_field["address"].evidence)
    # With MIN_PAGES_FOR_VERDICT = 1, single-page recovered value evaluates to consistent
    assert by_field["name"].verdict == "consistent"


def test_run_nap_check_never_calls_the_llm_for_a_field_the_regex_tiers_already_found():
    # Cost control: the LLM fallback exists only to recover a field with zero
    # candidates site-wide, never to double-check a field that already has one.
    page = fixture_page("clean_page.html", "https://ridgeline.example/")
    calls: list[list[str]] = []

    def spy(prompt: str, **_kwargs) -> str:
        calls.append([])
        return "{}"

    run_nap_check([page], generate=spy)
    # clean_page.html's own fixture already supplies enough real candidates for
    # name/address/phone that none of the three fields should be missing
    # site-wide -- if this assertion ever fails because the fixture changes, the
    # real thing to check is whether the LLM fallback fired for a field that
    # genuinely had zero candidates (expected) or one that already had some
    # (a cost-control regression).
    from app.extraction.nap_extractor import extract_nap_candidates

    already_found = extract_nap_candidates(page)
    missing_fields = [f for f in ("name", "address", "phone") if not already_found.get(f)]
    if not missing_fields:
        assert calls == []


def test_llm_sourced_evidence_survives_phase_7_validation():
    # The critical wiring check: nap_validator re-derives evidence with the
    # plain (non-LLM) extract_nap_candidates(), which can never reproduce an
    # "llm"-sourced value by construction. Without a deliberate exception for
    # that source, filter_valid_comparisons would silently discard every
    # comparison this whole feature ever recovers.
    from app.validation.nap_validator import filter_valid_comparisons, validate_nap

    page = _iron_locks_page()

    def generate(prompt: str, **_kwargs) -> str:
        return _json.dumps({"name": "Iron Locks & Levers", "address": "Springville, UT"})

    results = run_nap_check([page], generate=generate)
    assert validate_nap(results, [page])
    survivors = filter_valid_comparisons(results, [page])
    assert len(survivors) == 3


def test_llm_sourced_evidence_claiming_text_not_on_the_page_fails_validation():
    # Guards the other direction: validation must still reject a tampered/
    # fabricated "llm" value, not wave through anything with that source tag.
    # Built via compare_field() itself (rather than a hand-typed NAPComparison)
    # so the verdict/values/confidence are internally self-consistent and only
    # the literal-substring check (Check 1) is what can fail here.
    from app.validation.nap_validator import validate_nap

    page = _iron_locks_page()
    fabricated_evidence = [
        value(page.final_url, "Totally Made Up Business Inc", "totally made up business inc", source="llm")
    ]
    fabricated = compare_field("name", fabricated_evidence)

    assert validate_nap(fabricated, [page]) is False
