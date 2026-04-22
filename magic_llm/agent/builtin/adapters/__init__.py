"""Browsing adapter protocol and factory.

This module defines the BrowsingAdapter Protocol that all provider adapters
must implement, and provides a factory function to select adapters by provider
name.

Per design.md:
- All adapters normalize responses to plain text optimized for model consumption
- API keys are read from environment variables (lazy check at usage)
- Factory raises ValueError for unknown providers
"""

from typing import Protocol, runtime_checkable

from magic_llm.agent.builtin.adapters.serper import SerperAdapter
from magic_llm.agent.builtin.adapters.tavily import TavilyAdapter
from magic_llm.agent.builtin.adapters.exa import ExaAdapter


@runtime_checkable
class BrowsingAdapter(Protocol):
    """Protocol for browsing provider adapters.

    Adapters normalize provider-specific APIs to plain text output
    optimized for model consumption.

    All methods return plain text strings (NOT JSON strings) formatted
    for readability by LLMs.

    Methods:
        search: Execute web search and return formatted results.
        scrape: Extract content from a URL and return plain text.
    """

    def search(self, query: str, max_results: int = 10) -> str:
        """Execute search and return plain text results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 10).

        Returns:
            Plain text formatted results, e.g.:
            "Search results for 'quantum computing':
            1. https://example.com - 'Title' - Snippet text...
            2. https://example2.com - 'Title2' - Snippet..."
        """
        ...

    def scrape(self, url: str) -> str:
        """Extract content from URL and return plain text.

        Args:
            url: URL to scrape.

        Returns:
            Plain text content from the page, trimmed to reasonable length.
        """
        ...


# Adapter registry mapping provider names to adapter classes
_ADAPTER_REGISTRY: dict[str, type] = {
    "serper": SerperAdapter,
    "tavily": TavilyAdapter,
    "exa": ExaAdapter,
}


def get_browsing_adapter(provider: str = "serper") -> BrowsingAdapter:
    """Factory function to get adapter by provider name.

    Args:
        provider: One of "serper", "tavily", "exa". Defaults to "serper".

    Returns:
        BrowsingAdapter instance for the specified provider.

    Raises:
        ValueError: If provider is not supported.

    Example:
        >>> adapter = get_browsing_adapter("serper")
        >>> results = adapter.search("Python tutorials")
    """
    provider_lower = provider.lower()
    if provider_lower not in _ADAPTER_REGISTRY:
        valid_providers = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown browsing provider: '{provider}'. "
            f"Supported providers: {valid_providers}"
        )
    adapter_class = _ADAPTER_REGISTRY[provider_lower]
    return adapter_class()


def get_supported_providers() -> list[str]:
    """Return list of supported provider names.

    Returns:
        List of provider names that can be passed to get_browsing_adapter().
    """
    return sorted(_ADAPTER_REGISTRY.keys())


__all__ = [
    "BrowsingAdapter",
    "get_browsing_adapter",
    "get_supported_providers",
    "SerperAdapter",
    "TavilyAdapter",
    "ExaAdapter",
]