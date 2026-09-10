"""Normalization rules for NAP (Name, Address, Phone) comparison.

The plan's second named risk is over-eager normalization: a rule aggressive enough
to merge ``+91 98765 43210`` with ``919876543210`` is also aggressive enough to merge
two genuinely different branches, and *that* failure is invisible — it produces a
confident "consistent" verdict on a site with a real inconsistency.

So each rule below is conservative in one specific direction:

* **Phone** — compare the last ten digits rather than stripping a country code.
  Deciding that a leading ``91`` is India's dial code requires knowing the country;
  guessing wrong merges two different numbers. The last ten digits are the
  subscriber number in every numbering plan we care about, so comparing them needs
  no such guess.
* **Address** — expand abbreviations in one direction only (``St.`` → ``street``,
  never the reverse), and never touch suite/unit numbers or locality names, since
  those are exactly what distinguishes two branches on the same street.
* **Name** — case and punctuation are folded, but a legal suffix is **kept**.
  ``Acme Inc`` and ``Acme LLC`` are different legal entities and merging them would
  hide a real signal; they are reported as inconsistent for a human to judge.

Both the Phase 5 agent and the Phase 7 validator import from here, so the validator
recomputes with the same rules rather than a drifting second implementation.
"""

from __future__ import annotations

import re

#: Street-type abbreviations, expanded in one direction so ``St.``/``St``/``Street``
#: all land on ``street``. Expanding rather than contracting keeps the normalized
#: form readable in evidence.
STREET_SUFFIXES: dict[str, str] = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "pkwy": "parkway",
    "hwy": "highway",
    "cres": "crescent",
    "cl": "close",
    "gdns": "gardens",
}

#: Directional prefixes, expanded so ``SE Ankeny`` and ``Southeast Ankeny`` match.
DIRECTIONS: dict[str, str] = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

#: Unit designators, normalized in form but never removed — a suite number is the
#: difference between two tenants at one address.
UNIT_DESIGNATORS: dict[str, str] = {
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "flr": "floor",
    "rm": "room",
    "dept": "department",
    "unit": "unit",
}

#: Legal-entity suffixes. Detected and reported, never stripped for comparison.
LEGAL_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "llp",
        "lp",
        "ltd",
        "limited",
        "plc",
        "pvt",
        "private",
        "corp",
        "corporation",
        "co",
        "company",
        "gmbh",
        "bv",
        "nv",
        "ag",
        "sa",
        "srl",
        "pty",
        "pte",
    }
)

#: Digits compared when deciding whether two phone numbers are the same. Ten is the
#: subscriber-number length in the NANP, India, the UK and most of Europe.
PHONE_COMPARISON_DIGITS = 10

#: Fewer digits than this cannot be a real phone number (extensions, years, prices).
MIN_PHONE_DIGITS = 7

#: More digits than this is not a phone number either — usually a concatenated id.
MAX_PHONE_DIGITS = 15

_NON_DIGITS = re.compile(r"\D+")
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_phone(raw: str) -> str:
    """Return the comparison form of a phone number: its last ten digits.

    All non-digits are discarded, then the trailing
    :data:`PHONE_COMPARISON_DIGITS` are kept. This is what makes
    ``+91 98765 43210`` and ``919876543210`` compare equal without ever deciding
    that ``91`` is a country code — a decision that would merge two different
    numbers whenever the guess was wrong.

    The original string is preserved separately as ``NAPValue.raw_value``, so
    evidence always shows what the page actually said.

    Args:
        raw: A phone number as written on the page.

    Returns:
        Up to ten digits, or ``""`` when the input holds no plausible number.
    """
    digits = _NON_DIGITS.sub("", raw or "")

    if not (MIN_PHONE_DIGITS <= len(digits) <= MAX_PHONE_DIGITS):
        return ""

    return digits[-PHONE_COMPARISON_DIGITS:]


def normalize_address(raw: str) -> str:
    """Return the comparison form of a postal address.

    Lowercases, strips punctuation, collapses whitespace, and expands street-type,
    directional and unit abbreviations in a single direction.

    Two things are deliberately left alone because they distinguish genuinely
    different locations:

    * **Unit and suite numbers** — normalized in wording (``Ste`` → ``suite``) but
      never removed. ``Suite 200`` and ``Suite 300`` must not compare equal.
    * **Locality, region and postcode** — never dropped. Two branches often share a
      street name in different towns.

    ``st`` is expanded to ``street`` only when a digit appeared earlier in the
    address, so ``412 Ankeny St`` expands while ``St Louis Avenue`` does not — the
    same token means "Saint" at the start of a name and "Street" after a house
    number.

    Args:
        raw: An address as written on the page.

    Returns:
        The normalized address, or ``""`` for empty input.
    """
    if not raw:
        return ""

    text = _PUNCTUATION.sub(" ", raw.lower())
    tokens = text.split()

    out: list[str] = []
    seen_digit = False

    for token in tokens:
        if any(char.isdigit() for char in token):
            seen_digit = True
            out.append(token)
            continue

        if token in DIRECTIONS:
            out.append(DIRECTIONS[token])
        elif token in UNIT_DESIGNATORS:
            out.append(UNIT_DESIGNATORS[token])
        elif token in STREET_SUFFIXES and seen_digit:
            # Only after a house number: "st" is "Saint" in "St Louis".
            out.append(STREET_SUFFIXES[token])
        else:
            out.append(token)

    return " ".join(out)


#: A possessive apostrophe — both the ASCII straight quote and the typographic
#: "smart quote" real sites overwhelmingly use in body copy. Removed entirely
#: (not turned into a space, unlike other punctuation) so "Zabar's" and a page
#: that drops the apostrophe and writes "Zabars" normalize to the identical
#: "zabars" — a genuinely common real-world spelling variation for possessive
#: names ("Zabar's" / "Zabars", "Trader Joe's" / "Trader Joes"), not a real
#: difference in business identity. Found live: without this, "Zabar's" and
#: "Zabar’s" and "Zabars" would normalize to three different strings("zabar s"
#: twice via the general punctuation-to-space rule, plus "zabars"), which would
#: falsely report a business's own name as inconsistent with itself.
_APOSTROPHE = re.compile(r"['’]")


def normalize_name(raw: str) -> str:
    """Return the comparison form of a business name.

    Lowercases, strips punctuation and collapses whitespace. A possessive
    apostrophe is removed rather than turned into a word break — see
    :data:`_APOSTROPHE` — so "Zabar's" and "Zabars" compare equal. The legal
    suffix is **kept**: ``Acme Inc`` and ``Acme LLC`` are different legal
    entities, and silently merging them would hide a signal worth surfacing. Use
    :func:`differs_only_by_legal_suffix` to describe such a pair rather than
    normalizing the difference away.

    Args:
        raw: A business name as written on the page.

    Returns:
        The normalized name, or ``""`` for empty input.
    """
    if not raw:
        return ""
    without_apostrophes = _APOSTROPHE.sub("", raw.lower())
    return " ".join(_PUNCTUATION.sub(" ", without_apostrophes).split())


def strip_legal_suffix(normalized_name: str) -> str:
    """Return a normalized name with any trailing legal suffixes removed.

    Only for *describing* a difference, never for deciding one. Comparison always
    uses :func:`normalize_name`.

    Args:
        normalized_name: Output of :func:`normalize_name`.

    Returns:
        The name without trailing legal-entity tokens.
    """
    tokens = normalized_name.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def differs_only_by_legal_suffix(first: str, second: str) -> bool:
    """True when two normalized names match apart from their legal suffixes.

    Lets the agent say *why* two names disagree — ``Acme Inc`` vs ``Acme LLC`` is a
    different situation from ``Acme`` vs ``Bertha's Bakery`` — without merging them.

    Args:
        first: A normalized name.
        second: Another normalized name.

    Returns:
        ``True`` if the names differ but their suffix-stripped forms match.
    """
    if first == second:
        return False

    stripped_first = strip_legal_suffix(first)
    stripped_second = strip_legal_suffix(second)

    return bool(stripped_first) and stripped_first == stripped_second


def address_subsumes(longer: str, shorter: str) -> bool:
    """True when ``shorter`` is a truncation of ``longer`` rather than a different place.

    Solves a real and common case: one page states its address in full in JSON-LD
    ("412 SE Ankeny St, Portland, OR 97214") while another writes only the street
    line in visible text ("412 SE Ankeny St"). Those are the same location written
    at two levels of detail, and reporting them as inconsistent would be a confident
    claim the evidence does not support.

    The rule is deliberately narrow, because the opposite error is worse. Agreement
    requires **both**:

    1. ``shorter``'s tokens are a contiguous prefix of ``longer``'s — so a different
       house number, street or town can never match; and
    2. the extra tokens in ``longer`` contain no unit designator — so
       ``412 Ankeny St`` does **not** absorb ``412 Ankeny St Suite 300``. A suite
       number is exactly what distinguishes two tenants at one street address, and
       merging on it would hide a genuine inconsistency.

    ``Suite 200`` vs ``Suite 300`` fails test 1 outright, since neither is a prefix
    of the other.

    Args:
        longer: A normalized address.
        shorter: Another normalized address, expected to be the shorter one.

    Returns:
        Whether the two describe the same location at different detail levels.
    """
    if not longer or not shorter:
        return False

    long_tokens = longer.split()
    short_tokens = shorter.split()

    if len(short_tokens) >= len(long_tokens):
        return False
    if long_tokens[: len(short_tokens)] != short_tokens:
        return False

    extra = long_tokens[len(short_tokens) :]
    return not any(token in UNIT_DESIGNATORS.values() for token in extra)


def looks_like_phone(raw: str) -> bool:
    """True when a string holds a plausible phone number.

    A digit-count gate only. Deciding whether a plausible number is the *business's*
    number — as opposed to a figure quoted in a case study — is the agent's job,
    using where on the page it appeared.

    Args:
        raw: Candidate text.

    Returns:
        Whether the digit count falls in the phone range.
    """
    return bool(normalize_phone(raw))
