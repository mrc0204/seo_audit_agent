"""Command-line interface entry point for Evidence-Grounded SEO Audit Agent."""

import argparse


def main() -> None:
    """CLI entrypoint parsing flags and executing SEO audit pipeline."""
    parser = argparse.ArgumentParser(
        description="Evidence-Grounded SEO Audit Agent CLI"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Target website URL to crawl and audit",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional question for evidence-grounded QA agent",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum number of pages to crawl",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum crawl depth",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory path to write JSON outputs",
    )
    args = parser.parse_args()
    raise NotImplementedError(f"CLI execution for URL {args.url} not implemented.")


if __name__ == "__main__":
    main()
