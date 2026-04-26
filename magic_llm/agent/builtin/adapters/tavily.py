"""Tavily API adapter for web search and scrape.

Tavily is an alternative browsing provider optimized for AI agents.
It provides search results with relevance scores and content extraction.

Per spec.md:
- Search endpoint: POST https://api.tavily.com/search
- Auth header: Authorization: Bearer tvly-<TAVILY_API_KEY>
- Response normalization: Extract results array with title, url, content, score

API key is read from TAVILY_API_KEY environment variable (lazy check at usage).
"""

import logging
import os
from typing import Any

from magic_llm.util.http import HttpClient, HttpError

logger = logging.getLogger(__name__)


class TavilyAdapter:
    """Tavily API adapter implementing BrowsingAdapter protocol.

    Provides web search via Tavily's AI-optimized search API and content
    extraction via their extract endpoint.

    Attributes:
        api_key: Tavily API key from TAVILY_API_KEY env var.
        _api_key_checked: Whether API key presence has been verified.
    """

    # Search endpoint
    SEARCH_URL = "https://api.tavily.com/search"

    # Extract/scrape endpoint
    EXTRACT_URL = "https://api.tavily.com/extract"

    def __init__(self) -> None:
        """Initialize adapter with lazy API key check.

        The API key is not verified at initialization time.
        It will be checked when search() or scrape() is called.
        """
        self._api_key: str | None = None
        self._api_key_checked: bool = False

    def _get_api_key(self) -> str:
        """Get API key from environment, raising error if missing.

        Returns:
            Tavily API key string (formatted as Bearer token).

        Raises:
            RuntimeError: If TAVILY_API_KEY environment variable is not set.
        """
        if not self._api_key_checked:
            self._api_key = os.environ.get("TAVILY_API_KEY")
            self._api_key_checked = True

        if not self._api_key:
            raise RuntimeError(
                "TAVILY_API_KEY environment variable not set. "
                "Set it to your Tavily API key from https://tavily.com"
            )
        return self._api_key

    def search(self, query: str, max_results: int = 10) -> str:
        """Execute search via Tavily API and return plain text results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 10).

        Returns:
            Plain text formatted results with relevance scores:
            "[Relevance: 0.95]
            Title: ...
            URL: ...
            Content: ..."

        Raises:
            RuntimeError: If TAVILY_API_KEY is not set.
        """
        api_key = self._get_api_key()

        try:
            with HttpClient() as client:
                response = client.post_json(
                    self.SEARCH_URL,
                    json={
                        "query": query,
                        "max_results": max_results,
                        "include_raw_content": False,
                        "include_images": False,
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            return self._normalize_search_response(response, query)
        except HttpError as e:
            logger.error(f"Tavily search failed: {e}")
            return f"Search failed: {str(e)}"
        except Exception as e:
            logger.error(f"Tavily search unexpected error: {e}")
            return f"Search failed: {str(e)}"

    def _normalize_search_response(self, response: dict[str, Any], query: str) -> str:
        """Normalize Tavily API response to plain text.

        Args:
            response: Raw JSON response from Tavily API.
            query: Original search query for header.

        Returns:
            Plain text formatted results with relevance scores.
        """
        lines = [f"Search results for '{query}':"]

        results = response.get("results", [])
        if not results:
            lines.append("No results found.")
            return "\n".join(lines)

        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            score = result.get("score")

            # Format with relevance score when available
            lines.append(f"\n{i}. {url}")
            if score is not None:
                lines.append(f"   [Relevance: {score:.2f}]")
            lines.append(f"   Title: {title}")
            if content:
                # Truncate content to keep output readable
                max_content_len = 300
                if len(content) > max_content_len:
                    content = content[:max_content_len] + "..."
                lines.append(f"   Content: {content}")

        return "\n".join(lines)

    def scrape(self, url: str) -> str:
        """Extract content from URL via Tavily extract endpoint.

        Args:
            url: URL to scrape.

        Returns:
            Plain text content from the page.

        Raises:
            RuntimeError: If TAVILY_API_KEY is not set.
        """
        api_key = self._get_api_key()

        try:
            with HttpClient() as client:
                response = client.post_json(
                    self.EXTRACT_URL,
                    json={"urls": [url]},
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            return self._normalize_scrape_response(response, url)
        except HttpError as e:
            logger.error(f"Tavily extract failed: {e}")
            return f"Failed to scrape URL: {str(e)}"
        except Exception as e:
            logger.error(f"Tavily extract unexpected error: {e}")
            return f"Failed to scrape URL: {str(e)}"

    def _normalize_scrape_response(self, response: dict[str, Any], url: str) -> str:
        """Normalize Tavily extract response to plain text.

        Args:
            response: Raw JSON response from Tavily extract API.
            url: Original URL for context.

        Returns:
            Plain text content from the page.
        """
        # Tavily extract returns {"results": [{"url": "...", "raw_content": "..."}]}
        results = response.get("results", [])

        for result in results:
            if result.get("url") == url:
                content = result.get("raw_content", "")
                if content:
                    # Truncate to reasonable length
                    max_chars = 8000
                    if len(content) > max_chars:
                        content = content[:max_chars] + "\n... (truncated)"
                    return f"Content from {url}:\n{content}"

        # Check for error in response
        failed = response.get("failed", [])
        for fail in failed:
            if fail.get("url") == url:
                error = fail.get("error", "Unknown error")
                return f"Failed to extract content from {url}: {error}"

        return f"Content from {url}:\n(No extractable text found)"