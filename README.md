# Evidence-Grounded SEO Audit Agent

An AI-driven, evidence-grounded SEO auditing system designed to crawl websites, extract structured SEO & NAP (Name, Address, Phone) data, run deterministic checks, and perform retrieval-augmented question answering.

## Project Structure

```text
seo-audit-agent/
├── app/
│   ├── crawler/        # Web crawling, robots.txt parsing, sitemap parsing, and URL utilities
│   ├── extraction/     # HTML parsing, SEO meta tag, schema (JSON-LD/Microdata), NAP, and content extraction
│   ├── agents/         # SEO audit agent, NAP consistency agent, and QA agent
│   ├── retrieval/      # BM25/TF-IDF indexing and retrieval module
│   ├── validation/     # Audit, NAP, and QA output validators
│   ├── models/         # Pydantic data contract models
│   ├── llm/            # Swappable LLM provider interface (Groq, Gemini, etc.)
│   └── main.py         # CLI entry point
├── tests/              # Test suite and fixtures
│   ├── fixtures/
│   └── manual/
├── outputs/            # Generated audit reports and outputs
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup Instructions

1. **Clone the repository and create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to set your `LLM_PROVIDER` and `LLM_API_KEY`.

## Usage

Run the main entrypoint:
```bash
python -m app.main --url https://example.com
```
