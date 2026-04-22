"""Serper API adapter for web search and scrape.

Serper is the default browsing provider. It provides Google search results
via the Serper API (google.serper.dev) and page scraping via scrape.serper.dev.

Per spec.md:
- Search endpoint: POST https://google.serper.dev/search
- Auth header: X-API-KEY: <SERPER_API_KEY>
- Response normalization: Extract organic results as formatted text

API key is read from SERPER_API_KEY environment variable (lazy check at usage).
"""

import json
import logging
import os
from typing import Any

from magic_llm.util.http import HttpClient, HttpError

logger = logging.getLogger(__name__)


class SerperAdapter:
    """Serper API adapter implementing BrowsingAdapter protocol.

    Provides web search via Serper's Google search API and page scraping
    via Serper's scrape endpoint (with Jina fallback).

    Attributes:
        api_key: Serper API key from SERPER_API_KEY env var.
        _api_key_checked: Whether API key presence has been verified.
    """

    # Search endpoint for organic results
    SEARCH_URL = "https://google.serper.dev/search"

    # Scrape endpoint (Serper's dedicated scraper)
    SCRAPE_URL = "https://scrape.serper.dev"

    # Fallback scrape endpoint (Jina AI reader)
    JINA_SCRAPE_URL = "https://r.jina.ai/{url}"

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
            Serper API key string.

        Raises:
            RuntimeError: If SERPER_API_KEY environment variable is not set.
        """
        if not self._api_key_checked:
            self._api_key = os.environ.get("SERPER_API_KEY")
            self._api_key_checked = True

        if not self._api_key:
            raise RuntimeError(
                "SERPER_API_KEY environment variable not set. "
                "Set it to your Serper API key from https://serper.dev"
            )
        return self._api_key

    def search(self, query: str, max_results: int = 10) -> str:
        """Execute search via Serper API and return plain text results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 10).

        Returns:
            Plain text formatted results:
            "Search results for 'query':
            1. URL - Title - Snippet
            2. URL - Title - Snippet..."

        Raises:
            RuntimeError: If SERPER_API_KEY is not set.
        """
        api_key = self._get_api_key()

        try:
            with HttpClient() as client:
                response = client.post_json(
                    self.SEARCH_URL,
                    json={"q": query, "num": max_results},
                    headers={"X-API-KEY": api_key},
                )
            return self._normalize_search_response(response, query)
        except HttpError as e:
            logger.error(f"Serper search failed: {e}")
            return f"Search failed: {str(e)}"
        except Exception as e:
            logger.error(f"Serper search unexpected error: {e}")
            return f"Search failed: {str(e)}"

    def _normalize_search_response(self, response: dict[str, Any], query: str) -> str:
        """Normalize Serper API response to plain text.

        Args:
            response: Raw JSON response from Serper API.
            query: Original search query for header.

        Returns:
            Plain text formatted results.
        """
        lines = [f"Search results for '{query}':"]

        organic = response.get("organic", [])
        if not organic:
            lines.append("No results found.")
            return "\n".join(lines)

        for i, result in enumerate(organic, start=1):
            title = result.get("title", "No title")
            link = result.get("link", "")
            snippet = result.get("snippet", "")

            # Format each result as: "1. URL - Title - Snippet"
            lines.append(f"\n{i}. {link}")
            lines.append(f"   Title: {title}")
            if snippet:
                lines.append(f"   Snippet: {snippet}")

        return "\n".join(lines)

    def scrape(self, url: str) -> str:
        """Extract content from URL via Serper scrape or Jina fallback.

        Uses Serper's scrape endpoint first, falls back to Jina AI reader
        if Serper scrape fails.

        Args:
            url: URL to scrape.

        Returns:
            Plain text content from the page.

        Raises:
            RuntimeError: If SERPER_API_KEY is not set.
        """
        api_key = self._get_api_key()

        # Try Serper scrape first
        try:
            with HttpClient() as client:
                response = client.post_json(
                    self.SCRAPE_URL,
                    json={"url": url},
                    headers={"X-API-KEY": api_key},
                )
            return self._normalize_scrape_response(response, url)
        except HttpError as e:
            logger.warning(f"Serper scrape failed, trying Jina fallback: {e}")
        except Exception as e:
            logger.warning(f"Serper scrape unexpected error, trying Jina: {e}")

        # Fallback to Jina AI reader (no API key required)
        try:
            with HttpClient() as client:
                jina_url = self.JINA_SCRAPE_URL.format(url=url)
                content = client.request("GET", jina_url)
                return content.decode("utf-8", errors="replace")
        except HttpError as e:
            logger.error(f"Jina scrape failed: {e}")
            return f"Failed to scrape URL: {str(e)}"
        except Exception as e:
            logger.error(f"Scrape unexpected error: {e}")
            return f"Failed to scrape URL: {str(e)}"

    def _normalize_scrape_response(self, response: dict[str, Any], url: str) -> str:
        """Normalize Serper scrape response to plain text.

        Args:
            response: Raw JSON response from Serper scrape API.
            url: Original URL for context.

        Returns:
            Plain text content from the page.
        """
        # Serper scrape returns various formats depending on the endpoint
        # Most common: {"text": "..."} or {"content": "..."}
        text = response.get("text") or response.get("content") or response.get("markdown")

        if not text:
            # If no text field, try to extract from nested structure
            if "result" in response:
                result = response["result"]
                if isinstance(result, dict):
                    text = result.get("text") or result.get("content")

        if not text:
            return f"Content from {url}:\n(No extractable text found)"

        # Truncate to reasonable length (models have token limits)
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"

        return f"Content from {url}:\n{text}"