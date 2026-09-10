# Evidence-Grounded SEO Audit Agent

An agent that, given a single web URL, crawls the site and answers three
questions about it — **without owning or controlling the site** and **without
inventing anything it can't point to on the page**:

1. **On-page SEO audit** — deterministic checks (missing titles, broken
   canonicals, thin content, broken internal links, a `noindex` sent only via
   the `X-Robots-Tag` HTTP response header with nothing in the HTML to show for
   it, ...), each finding backed by a verbatim quote or a directly observed
   fact. Severity distinguishes real indexability defects from advisory
   best-practice recommendations — a missing meta description doesn't stop a
   page being indexed; a `noindex` directive does.
2. **NAP consistency** — does the business's Name, Address and Phone number
   match across every page it appears on?
3. **Grounded Q&A** — given a natural-language question, return the exact page
   and passage that answers it, or say plainly that the site doesn't answer it.

Every output is either traceable to real markup on a real crawled page, or it is
`null`. There is no step in this pipeline where a model is allowed to write a
`metric`, an `evidence` string, a NAP value, or a Q&A excerpt from scratch — see
[How correctness is enforced](#how-correctness-is-enforced) below.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional — see Configuration below
python -m app.main --url https://example.com
```

That's the whole setup. No paid API, key-gated service, or paid data connector is
used anywhere in this pipeline — see [LLM providers](#llm-providers-optional)
below for what "optional" means here.

## Usage

```bash
python -m app.main --url https://example.com
```

Add a question to also run Q3:

```bash
python -m app.main --url https://example.com \
  --question "What are your business hours?"
```

```
Crawled 6 page(s) of 6 fetched from https://example.com/
Q1 findings: 12 1 critical, 8 warning, 3 info
Q2 NAP: name=consistent, address=consistent, phone=consistent
Q3 answer: found on https://example.com/contact
Wrote: outputs/audit.json, outputs/nap_report.json, outputs/answer.json
```

All flags:

| Flag | Default | Purpose |
|---|---|---|
| `--url` | *(required)* | Target site. Any URL on the domain works as the seed. |
| `--question` | *(none)* | Natural-language question for Q3. Omit it and `answer.json` is simply not written — there's no honest empty shape for an answer with no question attached. |
| `--max-pages` | 100 | Crawl budget. |
| `--max-depth` | 3 | Link-distance budget from the seed URL. |
| `--timeout` | 10 | Per-request timeout, in seconds. |
| `--output-dir` | `outputs` | Where the three JSON files are written. |
| `--llm-provider` | `none` | `groq`, `gemini`, `nvidia`, or `none`. See below. |
| `--no-robots` | *(respects robots.txt)* | Ignore robots.txt. Not recommended. |
| `--no-sitemap` | *(uses the sitemap)* | Don't seed the crawl from `/sitemap.xml`. |

## Web UI

A minimal browser UI wraps the same pipeline — enter a URL (and optionally a
question), click **Run audit**, and see the findings, NAP verdicts, and grounded
answer rendered directly instead of reading JSON files.

```bash
uvicorn app.web:app --reload
```

Then open <http://127.0.0.1:8000/>. It calls `app.main.run_pipeline` — the exact
same function the CLI calls — so every guarantee in
[How correctness is enforced](#how-correctness-is-enforced) holds identically
here; `app/web.py` never touches a `Finding`, `NAPComparison`, or `QAAnswer`
directly, only the CLI's own serializers. It's a single static HTML file (no
build step, no external CDN — works fully offline) served by a small FastAPI
app; both are optional dependencies the CLI itself doesn't need.

## Configuration

`.env` (copy from `.env.example`):

```bash
LLM_PROVIDER=groq          # groq | gemini | nvidia | none
LLM_API_KEY=your_key_here  # only needed if LLM_PROVIDER isn't "none"
MAX_PAGES=100
MAX_DEPTH=3
TIMEOUT_SECONDS=10
```

`.env` values are defaults; any flag passed on the command line overrides them.

## Output shapes

Three files, matching the assignment brief exactly (a few internal fields the
pipeline needs for traceability — `check_id`, NAP `evidence`, Q&A `match_type` —
are intentionally *not* in these files; they exist only inside the pipeline).

**`audit.json`** — one entry per finding, severity distinguishing real
indexability defects from advisory recommendations (real pipeline output on a
small synthetic page, chosen to show both severities in one short example —
`missing_title` on a real crawl looks identical to this):

```json
[
  {
    "metric": "missing_title",
    "page": "https://example.com/products/",
    "severity": "critical",
    "evidence": "No <title> element is present in the document <head>.",
    "suggested_fix": "Add a unique <title> of roughly 30-60 characters describing this page."
  },
  {
    "metric": "image_missing_alt",
    "page": "https://example.com/products/",
    "severity": "warning",
    "evidence": "1 of 1 <img> elements have no alt attribute: https://example.com/hero.jpg",
    "suggested_fix": "Add descriptive alt text to each image, or alt=\"\" if it is purely decorative."
  },
  {
    "metric": "missing_canonical",
    "page": "https://example.com/products/",
    "severity": "info",
    "evidence": "No <link rel=\"canonical\"> element is present in the document <head>.",
    "suggested_fix": "Add a self-referencing canonical link to consolidate duplicate URLs."
  }
]
```

**`nap_report.json`** — one entry per field (`name`, `address`, `phone`). This
example is from a two-page fixture business site to show a `consistent` verdict;
a real single-page site with no NAP claim reports `insufficient_data` on every
field instead, honestly, rather than guessing:

```json
[
  {
    "field": "phone",
    "pages_compared": [
      "https://ridgeline.example/",
      "https://ridgeline.example/contact"
    ],
    "values": ["+1 503-555-0147"],
    "normalized_values": ["5035550147"],
    "confidence": 1.0,
    "verdict": "consistent"
  }
]
```

**`answer.json`** — `url`/`excerpt` are `null` when the site doesn't support an
answer:

```json
{
  "query": "Since when has Ridgeline Coffee Roasters been roasting coffee?",
  "url": "https://ridgeline.example/",
  "excerpt": "Ridgeline Coffee Roasters has roasted single-origin coffee on the east side of Portland since 2009."
}
```

```json
{
  "query": "What is your refund policy for damaged shipments?",
  "url": null,
  "excerpt": null
}
```

## How correctness is enforced

The brief is graded on whether findings and answers hold up under manual review.
Three things make that likely rather than hopeful:

- **Every check is a pure function over the page's own extracted data.** Nothing
  in Q1 or Q2 is inferred, summarized, or LLM-written. `evidence` is always a
  literal quote or a directly observed fact.
- **An LLM, if configured, is structurally restricted to rewriting phrasing.**
  In Q1 it may only replace `suggested_fix` — every finding is rebuilt from the
  original with everything else copied verbatim, so a model has no field through
  which to alter what was found. In Q3 the model may only *select* a passage and
  copy it; whatever it returns is independently re-checked (below) before it's
  ever trusted.
- **A separate validation pass re-derives every output from the crawl snapshot
  before it's allowed to reach a JSON file.** Q1 findings are reproduced by
  re-running the entire deterministic audit and requiring an identical result.
  Q2 evidence is re-extracted from the cited page and its normalization
  recomputed from scratch. Q3's gate is the strictest: an excerpt is accepted
  only if it is a **literal, whitespace-normalized substring** of the cited
  page's own text — no fuzzy matching, and a near-miss is treated as a failure,
  never "fixed" into shape. If any of this fails, the output is dropped before
  it's written, not after.

## LLM providers (optional)

Every check above runs, correctly, with `LLM_PROVIDER=none` — an LLM only
polishes `suggested_fix` wording and helps Q3 pick a more precise excerpt than
the deterministic top-match fallback. Nothing about correctness depends on one
being configured.

Three free-tier providers are supported, selected via `--llm-provider` or
`LLM_PROVIDER` in `.env` — swapping between them is a one-line config change,
not a code change:

- **Groq** (`GROQ_API_KEY`, or reuse `LLM_API_KEY`) — Groq's free-tier chat
  completions endpoint.
- **Gemini** (`GEMINI_API_KEY`, or reuse `LLM_API_KEY`) — Google's Gemini
  free-tier `generateContent` endpoint.
- **NVIDIA NIM** (`NVIDIA_API_KEY`, or reuse `LLM_API_KEY`) — NVIDIA's
  OpenAI-compatible free-tier catalog endpoint at build.nvidia.com. Its free
  tier is a separate quota entirely from Groq's or Gemini's, so it is a genuine
  fallback if one of the others runs out of free-tier headroom mid-testing.

All three are called directly over HTTP (no vendor SDK required — one less thing
to install). If a provider is selected but misconfigured or unreachable, the
pipeline logs a warning and continues fully deterministically; it never crashes
or silently degrades correctness. Every free tier here has its own real limits
(request-rate caps, a daily or total token/credit budget) — pick whichever one
still has headroom rather than assuming any is unlimited.

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

The full suite (models, crawler, extraction, all three agents, validators, CLI,
LLM providers, web UI — 415+ tests) runs offline: every network-facing module is tested
against fixture HTML or a mocked HTTP transport, never the live internet, so the
suite is fast and deterministic.

One additional script is *not* part of `pytest` because it makes real requests
to real third-party sites and is meant to be run by hand:

```bash
python tests/manual/cross_site_sweep.py
```

This runs the full pipeline against five real, unrelated sites — one each on
WordPress, Shopify, Webflow, plain static HTML, and a JS-heavy client-rendered
SPA — none of them used anywhere in development, per the project's own
generalization requirement. It prints every finding and Q&A result for manual
spot-checking and saves the full log as
`tests/manual/cross_site_sweep_results.json`.

## Known limitations

Stated plainly rather than hidden, per the plan's own guidance that this is
itself evidence-first thinking:

- **No JavaScript rendering.** The crawler fetches server-rendered HTML only. A
  client-rendered SPA (React/Vue app with an empty initial shell) will show up
  as a near-empty page — the pipeline detects this shape and attaches a note to
  the crawl report rather than silently reporting a false "thin content"
  finding as if the page really were that sparse, but it genuinely cannot see
  content that only exists after JavaScript runs. Verified live against
  [excalidraw.com](https://excalidraw.com) in the cross-site sweep.
- **Lexical, not semantic, retrieval.** Q3's pipeline is three separate stages:
  BM25 ranks its top 10–15 chunks by shared vocabulary alone (no relevance
  filter — that judgment belongs to the next stage, not to ranking); an LLM,
  when configured, judges which of those candidates *actually* answers the
  question rather than merely sharing its topic, then its selection is
  independently re-verified as a literal substring of the source page. Without
  an LLM, there is no semantic judge available, so the offline fallback
  compensates with its own much stricter coverage bar
  (`OFFLINE_ANSWER_MIN_COVERAGE` in `qa_agent.py`) before it will call anything
  "answered." Neither path understands meaning: asking "what's the cost" won't
  match a page that only ever says "price," and the offline path in particular
  will correctly return `null` on a fair number of questions a human would say
  the page answers, rather than risk presenting a merely-related passage as if
  it were the answer.
- **Phone/address normalization is deliberately conservative.** It compares the
  last 10 digits of a phone number (never guessing at a country code, which
  risks merging two genuinely different numbers) and treats a truncated address
  as the same location as its fuller form only when one is a strict prefix of
  the other with no differing unit/suite number. Some real inconsistencies —
  wording differences beyond what the normalizer expects — may not be flagged.
- **NAP entity identification is schema-based, not full-page semantic
  understanding.** `identify_target_business()` reads only JSON-LD/microdata
  business names to decide which entity a site's NAP data belongs to — the
  site's actual root page wins outright if it declares one (never merely the
  shortest-path page the crawl happened to reach, if root itself was never
  fetched), otherwise majority vote across every page. This is what stops a
  payment processor's or a sister brand's schema block from polluting the
  comparison. A business with no schema markup at all — name, phone and
  address stated only in plain visible text — has no target to filter against,
  so entity filtering silently does nothing there; the visible-text extraction
  patterns themselves are still as tightly gated as before.
- **One crawl is one snapshot.** The audit reflects the site at the moment it
  was crawled; it does not detect changes over time or run continuously.
- **The LLM providers have not been exercised against a real API key in this
  project's own development** (no key was available). They're implemented
  against each vendor's documented REST contract and tested offline against a
  mocked response of that exact shape — genuinely worth a real run with your
  own key before relying on the polish they add.

## Project structure

```text
seo_audit_agent/
├── app/
│   ├── crawler/        # URL normalization, robots.txt, sitemap, BFS crawler
│   ├── extraction/      # HTML parsing, SEO/schema/NAP extraction, normalization
│   ├── agents/          # Q1 (seo_agent), Q2 (nap_agent), Q3 (qa_agent)
│   ├── retrieval/       # BM25 chunk index + top-k retriever
│   ├── validation/       # Independent re-derivation checks for Q1/Q2/Q3 output
│   ├── models/           # Locked Pydantic data contracts (Phase 1)
│   ├── llm/              # Swappable Groq/Gemini/NVIDIA provider
│   ├── main.py            # CLI: wires the pipeline together, writes outputs/
│   ├── web.py             # Optional FastAPI wrapper over app.main.run_pipeline
│   └── static/index.html  # Single-file frontend served by web.py
├── tests/                # ~415 offline tests + fixture pages
│   ├── fixtures/
│   └── manual/            # cross_site_sweep.py — real-network, run by hand
├── outputs/                # audit.json / nap_report.json / answer.json land here
├── requirements.txt
└── .env.example
```
