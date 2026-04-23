"""Unit tests for browsing provider adapters.

Tests cover:
- SerperAdapter normalization to plain text
- TavilyAdapter normalization with relevance scores
- ExaAdapter normalization
- Adapter factory (get_browsing_adapter)
- Error handling (API failures, missing API keys)
- HTTP mocking to avoid real API calls

Per tasks.md Phase 5.1:
- Test adapter normalization → plain text format
- Test factory returns correct adapter for each provider
- Test factory raises ValueError for unknown provider
"""

import json
import os
from unittest.mock import patch, MagicMock
from typing import Any

import pytest

from magic_llm.agent.builtin.adapters import (
    BrowsingAdapter,
    get_browsing_adapter,
    get_supported_providers,
)
from magic_llm.agent.builtin.adapters.serper import SerperAdapter
from magic_llm.agent.builtin.adapters.tavily import TavilyAdapter
from magic_llm.agent.builtin.adapters.exa import ExaAdapter
from magic_llm.util.http import HttpError


# ─── Mock HTTP Responses (fixtures) ────────────────────────────────────────

SERPER_SEARCH_RESPONSE = {
    "organic": [
        {"title": "Python Tutorial", "link": "https://example.com/python", "snippet": "Learn Python basics"},
        {"title": "Advanced Python", "link": "https://example.com/advanced", "snippet": "Deep dive into Python"},
        {"title": "Python Documentation", "link": "https://docs.python.org", "snippet": "Official docs"},
    ]
}

TAVILY_SEARCH_RESPONSE = {
    "results": [
        {"title": "AI Research", "url": "https://example.com/ai", "content": "Latest AI developments", "score": 0.95},
        {"title": "Machine Learning", "url": "https://example.com/ml", "content": "ML fundamentals", "score": 0.87},
    ]
}

EXA_SEARCH_RESPONSE = {
    "results": [
        {"title": "Quantum Computing", "url": "https://example.com/quantum", "text": "Introduction to quantum"},
        {"title": "Quantum Algorithms", "url": "https://example.com/algorithms", "highlights": ["Shor's algorithm", "Grover's algorithm"]},
    ]
}

SERPER_SCRAPE_RESPONSE = {
    "text": "This is the scraped content from the page. It contains useful information about the topic."
}

TAVILY_EXTRACT_RESPONSE = {
    "results": [
        {"url": "https://example.com/article", "raw_content": "Full article content here..."}
    ]
}

EXA_CONTENTS_RESPONSE = {
    "results": [
        {"id": "https://example.com/paper", "text": "Research paper abstract and content..."}
    ]
}


# ─── Helper: Mock HttpClient ──────────────────────────────────────────────

def mock_http_client_post_json(response_data: dict):
    """Create a mock HttpClient that returns response_data for post_json calls."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post_json.return_value = response_data
    return mock_client


def mock_http_client_request(content: str):
    """Create a mock HttpClient that returns content for request calls."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.request.return_value = content.encode('utf-8')
    return mock_client


# ─── Factory Tests ────────────────────────────────────────────────────────

class TestAdapterFactory:
    """Tests for get_browsing_adapter factory function."""

    def test_factory_returns_serper_adapter(self):
        """get_browsing_adapter('serper') returns SerperAdapter instance."""
        adapter = get_browsing_adapter("serper")
        assert isinstance(adapter, SerperAdapter)
        assert isinstance(adapter, BrowsingAdapter)

    def test_factory_returns_tavily_adapter(self):
        """get_browsing_adapter('tavily') returns TavilyAdapter instance."""
        adapter = get_browsing_adapter("tavily")
        assert isinstance(adapter, TavilyAdapter)
        assert isinstance(adapter, BrowsingAdapter)

    def test_factory_returns_exa_adapter(self):
        """get_browsing_adapter('exa') returns ExaAdapter instance."""
        adapter = get_browsing_adapter("exa")
        assert isinstance(adapter, ExaAdapter)
        assert isinstance(adapter, BrowsingAdapter)

    def test_factory_default_is_serper(self):
        """get_browsing_adapter() with no arg returns SerperAdapter."""
        adapter = get_browsing_adapter()
        assert isinstance(adapter, SerperAdapter)

    def test_factory_case_insensitive(self):
        """Provider names are case-insensitive."""
        adapter_upper = get_browsing_adapter("SERPER")
        adapter_mixed = get_browsing_adapter("TaViLy")
        assert isinstance(adapter_upper, SerperAdapter)
        assert isinstance(adapter_mixed, TavilyAdapter)

    def test_factory_unknown_provider_raises_valueerror(self):
        """get_browsing_adapter('unknown') raises ValueError with valid providers listed."""
        with pytest.raises(ValueError) as exc_info:
            get_browsing_adapter("unknown_provider")
        
        error_msg = str(exc_info.value)
        assert "Unknown browsing provider" in error_msg
        assert "serper" in error_msg
        assert "tavily" in error_msg
        assert "exa" in error_msg

    def test_get_supported_providers_returns_sorted_list(self):
        """get_supported_providers returns list of valid provider names."""
        providers = get_supported_providers()
        assert providers == ["exa", "serper", "tavily"]  # Sorted alphabetically
        assert len(providers) == 3


# ─── SerperAdapter Tests ──────────────────────────────────────────────────

class TestSerperAdapterSearch:
    """Tests for SerperAdapter.search() normalization."""

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_search_returns_plain_text_format(self, mock_http_class):
        """search() returns plain text (not JSON string) with formatted results."""
        mock_http_class.return_value = mock_http_client_post_json(SERPER_SEARCH_RESPONSE)
        
        adapter = SerperAdapter()
        result = adapter.search("Python tutorials", max_results=3)
        
        # Result is plain text, not JSON
        assert isinstance(result, str)
        assert not result.startswith("{")  # Not JSON string
        
        # Contains expected format elements
        assert "Search results for 'Python tutorials'" in result
        assert "https://example.com/python" in result
        assert "Python Tutorial" in result
        assert "Learn Python basics" in result

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_search_format_structure(self, mock_http_class):
        """search() format: "1. URL\n   Title: ...\n   Snippet: ..." """
        mock_http_class.return_value = mock_http_client_post_json(SERPER_SEARCH_RESPONSE)
        
        adapter = SerperAdapter()
        result = adapter.search("test query")
        
        # Check structure: numbered results with URL, title, snippet
        lines = result.split("\n")
        assert lines[0].startswith("Search results for")
        
        # First result should have these elements
        assert "1." in result
        assert "Title:" in result
        assert "Snippet:" in result

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_search_empty_results_returns_no_results_message(self, mock_http_class):
        """search() with empty organic results returns 'No results found.'"""
        mock_http_class.return_value = mock_http_client_post_json({"organic": []})
        
        adapter = SerperAdapter()
        result = adapter.search("nonexistent query")
        
        assert "Search results for 'nonexistent query'" in result
        assert "No results found" in result

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_search_api_error_returns_error_text(self, mock_http_class):
        """search() API failure returns error text (not raises exception)."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post_json.side_effect = HttpError("API rate limited", status_code=429)
        mock_http_class.return_value = mock_client
        
        adapter = SerperAdapter()
        result = adapter.search("test")
        
        # Error is returned as text (tool continues, model sees error)
        assert "Search failed" in result
        assert "API rate limited" in result

    def test_search_missing_api_key_raises_runtimeerror(self):
        """search() without SERPER_API_KEY raises RuntimeError with clear message."""
        # Ensure no API key
        with patch.dict(os.environ, {}, clear=True):
            adapter = SerperAdapter()
            with pytest.raises(RuntimeError) as exc_info:
                adapter.search("test query")
            
            error_msg = str(exc_info.value)
            assert "SERPER_API_KEY" in error_msg
            assert "not set" in error_msg
            assert "serper.dev" in error_msg


class TestSerperAdapterScrape:
    """Tests for SerperAdapter.scrape() normalization."""

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_scrape_returns_plain_text_content(self, mock_http_class):
        """scrape() returns plain text content from page."""
        mock_http_class.return_value = mock_http_client_post_json(SERPER_SCRAPE_RESPONSE)
        
        adapter = SerperAdapter()
        result = adapter.scrape("https://example.com/article")
        
        assert isinstance(result, str)
        assert "Content from https://example.com/article" in result
        assert "scraped content from the page" in result

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_scrape_truncates_long_content(self, mock_http_class):
        """scrape() truncates content > 8000 chars."""
        long_content = "A" * 10000  # Very long content
        mock_http_class.return_value = mock_http_client_post_json({"text": long_content})
        
        adapter = SerperAdapter()
        result = adapter.scrape("https://example.com/long")
        
        # Should be truncated
        assert len(result) < 9000  # Truncated + header + "... (truncated)"
        assert "... (truncated)" in result

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_scrape_fallback_to_jina_on_serper_failure(self, mock_http_class):
        """scrape() falls back to Jina AI reader when Serper scrape fails."""
        # First call (Serper) fails, second call (Jina) succeeds
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post_json.side_effect = HttpError("Serper failed", status_code=500)
        mock_client.request.return_value = b"Jina extracted content"
        mock_http_class.return_value = mock_client
        
        adapter = SerperAdapter()
        result = adapter.scrape("https://example.com/fallback-test")
        
        # Should have Jina content
        assert "Jina extracted content" in result


# ─── TavilyAdapter Tests ──────────────────────────────────────────────────

class TestTavilyAdapterSearch:
    """Tests for TavilyAdapter.search() normalization with relevance scores."""

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.tavily.HttpClient")
    def test_search_includes_relevance_scores(self, mock_http_class):
        """search() includes [Relevance: 0.XX] in output format."""
        mock_http_class.return_value = mock_http_client_post_json(TAVILY_SEARCH_RESPONSE)
        
        adapter = TavilyAdapter()
        result = adapter.search("AI research")
        
        assert isinstance(result, str)
        assert "Search results for 'AI research'" in result
        assert "[Relevance:" in result
        assert "0.95" in result  # Score from fixture

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.tavily.HttpClient")
    def test_search_format_with_scores(self, mock_http_class):
        """search() format: numbered URL, [Relevance: X.XX], Title, Content."""
        mock_http_class.return_value = mock_http_client_post_json(TAVILY_SEARCH_RESPONSE)
        
        adapter = TavilyAdapter()
        result = adapter.search("test")
        
        assert "Title:" in result
        assert "Content:" in result
        assert "https://example.com/ai" in result

    def test_search_missing_api_key_raises_runtimeerror(self):
        """search() without TAVILY_API_KEY raises RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            adapter = TavilyAdapter()
            with pytest.raises(RuntimeError) as exc_info:
                adapter.search("test")
            
            assert "TAVILY_API_KEY" in str(exc_info.value)
            assert "tavily.com" in str(exc_info.value)


class TestTavilyAdapterScrape:
    """Tests for TavilyAdapter.scrape() via extract endpoint."""

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.tavily.HttpClient")
    def test_scrape_returns_raw_content(self, mock_http_class):
        """scrape() returns raw_content from extract endpoint."""
        mock_http_class.return_value = mock_http_client_post_json(TAVILY_EXTRACT_RESPONSE)
        
        adapter = TavilyAdapter()
        result = adapter.scrape("https://example.com/article")
        
        assert "Full article content here" in result


# ─── ExaAdapter Tests ─────────────────────────────────────────────────────

class TestExaAdapterSearch:
    """Tests for ExaAdapter.search() normalization."""

    @patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.exa.HttpClient")
    def test_search_returns_plain_text_format(self, mock_http_class):
        """search() returns plain text with title and text/highlights."""
        mock_http_class.return_value = mock_http_client_post_json(EXA_SEARCH_RESPONSE)
        
        adapter = ExaAdapter()
        result = adapter.search("quantum computing")
        
        assert isinstance(result, str)
        assert "Search results for 'quantum computing'" in result
        assert "Quantum Computing" in result
        assert "https://example.com/quantum" in result

    @patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.exa.HttpClient")
    def test_search_uses_highlights_when_text_missing(self, mock_http_class):
        """search() uses highlights array when text field is missing."""
        # Result with only highlights, no text
        response = {
            "results": [
                {"title": "Test", "url": "https://example.com", "highlights": ["Highlight 1", "Highlight 2"]}
            ]
        }
        mock_http_class.return_value = mock_http_client_post_json(response)
        
        adapter = ExaAdapter()
        result = adapter.search("test")
        
        assert "Highlight 1 Highlight 2" in result

    def test_search_missing_api_key_raises_runtimeerror(self):
        """search() without EXA_API_KEY raises RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            adapter = ExaAdapter()
            with pytest.raises(RuntimeError) as exc_info:
                adapter.search("test")
            
            assert "EXA_API_KEY" in str(exc_info.value)
            assert "exa.ai" in str(exc_info.value)


class TestExaAdapterScrape:
    """Tests for ExaAdapter.scrape() via contents endpoint."""

    @patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"})
    @patch("magic_llm.agent.builtin.adapters.exa.HttpClient")
    def test_scrape_returns_text_from_contents(self, mock_http_class):
        """scrape() returns text from contents endpoint."""
        mock_http_class.return_value = mock_http_client_post_json(EXA_CONTENTS_RESPONSE)
        
        adapter = ExaAdapter()
        result = adapter.scrape("https://example.com/paper")
        
        assert "Research paper abstract" in result


# ─── Protocol Compliance Tests ─────────────────────────────────────────────

class TestBrowsingAdapterProtocol:
    """Tests for BrowsingAdapter Protocol compliance."""

    def test_serper_implements_protocol(self):
        """SerperAdapter implements BrowsingAdapter Protocol."""
        adapter = SerperAdapter()
        assert hasattr(adapter, "search")
        assert hasattr(adapter, "scrape")
        assert callable(adapter.search)
        assert callable(adapter.scrape)

    def test_tavily_implements_protocol(self):
        """TavilyAdapter implements BrowsingAdapter Protocol."""
        adapter = TavilyAdapter()
        assert hasattr(adapter, "search")
        assert hasattr(adapter, "scrape")

    def test_exa_implements_protocol(self):
        """ExaAdapter implements BrowsingAdapter Protocol."""
        adapter = ExaAdapter()
        assert hasattr(adapter, "search")
        assert hasattr(adapter, "scrape")

    @patch.dict(os.environ, {"SERPER_API_KEY": "test"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    def test_protocol_runtime_checkable(self, mock_http_class):
        """BrowsingAdapter is runtime_checkable."""
        mock_http_class.return_value = mock_http_client_post_json(SERPER_SEARCH_RESPONSE)
        
        adapter = SerperAdapter()
        # Protocol should be checkable at runtime
        assert isinstance(adapter, BrowsingAdapter)


# ─── Output Format Consistency Tests ───────────────────────────────────────

class TestOutputFormatConsistency:
    """Tests that all adapters produce consistent plain text format."""

    @patch.dict(os.environ, {"SERPER_API_KEY": "test", "TAVILY_API_KEY": "test", "EXA_API_KEY": "test"})
    @patch("magic_llm.agent.builtin.adapters.serper.HttpClient")
    @patch("magic_llm.agent.builtin.adapters.tavily.HttpClient")
    @patch("magic_llm.agent.builtin.adapters.exa.HttpClient")
    def test_all_adapters_return_str_not_json(self, mock_exa, mock_tavily, mock_serper):
        """All adapters return str type (not JSON string or dict)."""
        mock_serper.return_value = mock_http_client_post_json(SERPER_SEARCH_RESPONSE)
        mock_tavily.return_value = mock_http_client_post_json(TAVILY_SEARCH_RESPONSE)
        mock_exa.return_value = mock_http_client_post_json(EXA_SEARCH_RESPONSE)
        
        for provider in ["serper", "tavily", "exa"]:
            adapter = get_browsing_adapter(provider)
            result = adapter.search("test query")
            
            assert isinstance(result, str)
            # Should NOT be a JSON string (parsed would be dict)
            with pytest.raises(json.JSONDecodeError):
                json.loads(result)  # Plain text, not valid JSON