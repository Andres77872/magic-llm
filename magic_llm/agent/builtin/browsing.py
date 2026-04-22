"""Builtin browsing tools for web search and scrape.

These are plain Python callables with docstrings that magic-llm's
_schema_from_callable() can extract to generate OpenAI-compatible tool schemas.

Per spec.md:
- Tools return plain text output optimized for model consumption
- Provider selection resolved via get_browsing_adapter()
- Docstrings include parameter descriptions for schema generation
"""

from typing import Optional

from magic_llm.agent.builtin.adapters import get_browsing_adapter
from magic_llm.agent.builtin._config import get_default_provider


def web_search(query: str, max_results: int = 10, provider: Optional[str] = None) -> str:
    """Search the web for information.

    Use this tool when you need to find current information, news,
    or research topics on the web. Returns formatted search results
    with titles, URLs, and snippets in plain text format.

    Args:
        query: The search query. Be specific for better results.
        max_results: Number of results to return. Defaults to 10.
        provider: Search provider to use. Defaults to configured provider (serper).

    Returns:
        Plain text search results with URLs and snippets, formatted for model readability.
    """
    resolved_provider = provider or get_default_provider()
    adapter = get_browsing_adapter(resolved_provider)
    return adapter.search(query, max_results=max_results)


def web_scrape(url: str, provider: Optional[str] = None) -> str:
    """Extract and read content from a web page.

    Use this tool when you need to read the full content of a
    specific web page. Useful after web_search to get detailed
    information from promising results.

    Args:
        url: The URL of the page to scrape. Must be a valid HTTP or HTTPS URL.
        provider: Provider to use for scraping. Defaults to configured provider (serper).

    Returns:
        Plain text content from the page, trimmed to reasonable length for model consumption.
    """
    resolved_provider = provider or get_default_provider()
    adapter = get_browsing_adapter(resolved_provider)
    return adapter.scrape(url)


__all__ = ["web_search", "web_scrape"]