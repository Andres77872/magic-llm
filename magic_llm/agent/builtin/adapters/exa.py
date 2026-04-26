"""Exa API adapter for web search and scrape.

Exa is an alternative browsing provider with neural search capabilities.
It provides AI-powered search and content retrieval via their API.

Per spec.md:
- Search endpoint: POST https://api.exa.ai/search
- Auth header: x-api-key: <EXA_API_KEY>
- Response normalization: Extract results array with title, url, text/highlights

API key is read from EXA_API_KEY environment variable (lazy check at usage).
"""

import logging
import os
from typing import Any

from magic_llm.util.http import HttpClient, HttpError

logger = logging.getLogger(__name__)


class ExaAdapter:
    """Exa API adapter implementing BrowsingAdapter protocol.

    Provides neural web search via Exa's AI-powered search API and content
    retrieval via their contents endpoint.

    Attributes:
        api_key: Exa API key from EXA_API_KEY env var.
        _api_key_checked: Whether API key presence has been verified.
    """

    # Search endpoint
    SEARCH_URL = "https://api.exa.ai/search"

    # Contents endpoint for scraping
    CONTENTS_URL = "https://api.exa.ai/contents"

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
            Exa API key string.

        Raises:
            RuntimeError: If EXA_API_KEY environment variable is not set.
        """
        if not self._api_key_checked:
            self._api_key = os.environ.get("EXA_API_KEY")
            self._api_key_checked = True

        if not self._api_key:
            raise RuntimeError(
                "EXA_API_KEY environment variable not set. "
                "Set it to your Exa API key from https://exa.ai"
            )
        return self._api_key

    def search(self, query: str, max_results: int = 10) -> str:
        """Execute neural search via Exa API and return plain text results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 10).

        Returns:
            Plain text formatted results:
            "Search results for 'query':
            1. URL - Title - Text..."

        Raises:
            RuntimeError: If EXA_API_KEY is not set.
        """
        api_key = self._get_api_key()

        try:
            with HttpClient() as client:
                response = client.post_json(
                    self.SEARCH_URL,
                    json={
                        "query": query,
                        "numResults": max_results,
                        "useAutoprompt": True,
                        "contents": {
                            "text": {"maxCharacters": 1000},
                        },
                    },
                    headers={"x-api-key": api_key},
                )
            return self._normalize_search_response(response, query)
        except HttpError as e:
            logger.error(f"Exa search failed: {e}")
            return f"Search failed: {str(e)}"
        except Exception as e:
            logger.error(f"Exa search unexpected error: {e}")
            return f"Search failed: {str(e)}"

    def _normalize_search_response(self, response: dict[str, Any], query: str) -> str:
        """Normalize Exa API response to plain text.

        Args:
            response: Raw JSON response from Exa API.
            query: Original search query for header.

        Returns:
            Plain text formatted results.
        """
        lines = [f"Search results for '{query}':"]

        results = response.get("results", [])
        if not results:
            lines.append("No results found.")
            return "\n".join(lines)

        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            url = result.get("url", "")

            # Exa returns text in different locations depending on request
            text = result.get("text") or ""
            highlights = result.get("highlights", [])
            if not text and highlights:
                text = " ".join(highlights)

            lines.append(f"\n{i}. {url}")
            lines.append(f"   Title: {title}")
            if text:
                # Truncate text to keep output readable
                max_text_len = 300
                if len(text) > max_text_len:
                    text = text[:max_text_len] + "..."
                lines.append(f"   Text: {text}")

        return "\n".join(lines)

    def scrape(self, url: str) -> str:
        """Extract content from URL via Exa contents endpoint.

        Args:
            url: URL to scrape.

        Returns:
            Plain text content from the page.

        Raises:
            RuntimeError: If EXA_API_KEY is not set.
        """
        api_key = self._get_api_key()

        try:
            with HttpClient() as client:
                response = client.post_json(
                    self.CONTENTS_URL,
                    json={
                        "ids": [url],
                        "contents": {"text": {"maxCharacters": 8000}},
                    },
                    headers={"x-api-key": api_key},
                )
            return self._normalize_scrape_response(response, url)
        except HttpError as e:
            logger.error(f"Exa contents failed: {e}")
            return f"Failed to scrape URL: {str(e)}"
        except Exception as e:
            logger.error(f"Exa contents unexpected error: {e}")
            return f"Failed to scrape URL: {str(e)}"

    def _normalize_scrape_response(self, response: dict[str, Any], url: str) -> str:
        """Normalize Exa contents response to plain text.

        Args:
            response: Raw JSON response from Exa contents API.
            url: Original URL for context.

        Returns:
            Plain text content from the page.
        """
        # Exa contents returns {"results": [{"id": "url", "text": "..."}]}
        results = response.get("results", [])

        for result in results:
            if result.get("id") == url or result.get("url") == url:
                text = result.get("text", "")
                if text:
                    return f"Content from {url}:\n{text}"
                break

        return f"Content from {url}:\n(No extractable text found)"