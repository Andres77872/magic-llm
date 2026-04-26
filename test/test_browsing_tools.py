"""Unit tests for builtin browsing tools (web_search, web_scrape).

Tests cover:
- Schema generation from callables via _schema_from_callable
- web_search callable signature and parameter extraction
- web_scrape callable signature and parameter extraction
- Tool provider resolution logic
- get_browsing_tools() factory function
- get_browsing_tool_functions() dict builder

Per tasks.md Phase 5.2:
- Test web_search() callable signature extraction
- Test _schema_from_callable(web_search) → OpenAI tool schema
- Test web_scrape() schema generation
- Test provider resolution logic
"""

import os
from unittest.mock import patch, MagicMock
from typing import Any

import pytest

from magic_llm.agent.builtin import (
    web_search,
    web_scrape,
    get_browsing_tools,
    get_browsing_tool_functions,
    get_browsing_adapter,
    get_default_provider,
    get_request_provider,
)
from magic_llm.agent.builtin._config import get_default_provider, get_request_provider
from magic_llm.agent.builtin.adapters import get_browsing_adapter


# ─── Schema Generation Tests ──────────────────────────────────────────────

class TestWebSearchSchema:
    """Tests for web_search callable and schema generation."""

    def test_web_search_is_callable(self):
        """web_search is a callable function."""
        assert callable(web_search)
        assert web_search.__name__ == "web_search"

    def test_web_search_has_docstring(self):
        """web_search has docstring for schema generation."""
        doc = web_search.__doc__
        assert doc is not None
        assert "Search the web" in doc
        assert "query" in doc

    def test_web_search_signature_extraction(self):
        """_schema_from_callable extracts web_search parameters correctly."""
        from magic_llm.util.tools_mapping import _schema_from_callable
        
        name, description, parameters = _schema_from_callable(web_search)
        
        assert name == "web_search"
        assert "search" in description.lower()
        
        # Parameters schema
        assert parameters["type"] == "object"
        assert "query" in parameters["properties"]
        assert parameters["properties"]["query"]["type"] == "string"
        assert "query" in parameters["required"]  # query is required
        
        # Optional parameters
        assert "max_results" in parameters["properties"]
        assert "provider" in parameters["properties"]

    def test_web_search_default_max_results(self):
        """web_search signature shows max_results has default=10 in description."""
        from magic_llm.util.tools_mapping import _schema_from_callable
        
        _, _, parameters = _schema_from_callable(web_search)
        
        max_results_param = parameters["properties"]["max_results"]
        assert max_results_param["type"] == "integer"
        # Note: _schema_from_callable puts default value info in description, not 'default' field
        assert "default" in max_results_param["description"].lower() or "10" in max_results_param["description"]


class TestWebScrapeSchema:
    """Tests for web_scrape callable and schema generation."""

    def test_web_scrape_is_callable(self):
        """web_scrape is a callable function."""
        assert callable(web_scrape)
        assert web_scrape.__name__ == "web_scrape"

    def test_web_scrape_has_docstring(self):
        """web_scrape has docstring for schema generation."""
        doc = web_scrape.__doc__
        assert doc is not None
        assert "Extract and read" in doc or "scrape" in doc.lower()
        assert "url" in doc

    def test_web_scrape_signature_extraction(self):
        """_schema_from_callable extracts web_scrape parameters correctly."""
        from magic_llm.util.tools_mapping import _schema_from_callable
        
        name, description, parameters = _schema_from_callable(web_scrape)
        
        assert name == "web_scrape"
        assert "scrape" in description.lower() or "extract" in description.lower()
        
        # Parameters schema
        assert parameters["type"] == "object"
        assert "url" in parameters["properties"]
        assert parameters["properties"]["url"]["type"] == "string"
        assert "url" in parameters["required"]  # url is required
        
        # Optional provider parameter
        assert "provider" in parameters["properties"]


class TestGetBrowsingTools:
    """Tests for get_browsing_tools() factory function."""

    def test_get_browsing_tools_returns_list_of_tuples(self):
        """get_browsing_tools() returns list of (schema, callable) tuples."""
        tools = get_browsing_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 2  # web_search + web_scrape
        
        for item in tools:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_get_browsing_tools_schema_format(self):
        """Schemas are OpenAI-compatible tool definitions."""
        tools = get_browsing_tools()
        
        for schema, callable_fn in tools:
            # OpenAI format: {"type": "function", "function": {...}}
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
            
            # Parameters is a valid JSON schema
            params = schema["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_get_browsing_tools_includes_callables(self):
        """get_browsing_tools(include_callables=True) includes callable functions."""
        tools = get_browsing_tools(include_callables=True)
        
        tool_names = {schema["function"]["name"] for schema, _ in tools}
        assert "web_search" in tool_names
        assert "web_scrape" in tool_names
        
        # Callables match tool names
        callables_dict = {schema["function"]["name"]: fn for schema, fn in tools}
        assert callables_dict["web_search"] == web_search
        assert callables_dict["web_scrape"] == web_scrape

    def test_get_browsing_tools_without_callables(self):
        """get_browsing_tools(include_callables=False) returns None for callables."""
        tools = get_browsing_tools(include_callables=False)
        
        for schema, callable_fn in tools:
            assert callable_fn is None

    def test_get_browsing_tools_provider_passed_to_callables(self):
        """Provider parameter is passed but tools use resolved provider."""
        # The provider arg affects which adapter is used internally
        tools = get_browsing_tools(provider="tavily")
        
        # Should still return valid schemas and callables
        assert len(tools) == 2
        assert all(isinstance(t[0], dict) for t in tools)


class TestGetBrowsingToolFunctions:
    """Tests for get_browsing_tool_functions() dict builder."""

    def test_get_browsing_tool_functions_returns_dict(self):
        """get_browsing_tool_functions() returns dict of name -> callable."""
        tool_functions = get_browsing_tool_functions()
        
        assert isinstance(tool_functions, dict)
        assert "web_search" in tool_functions
        assert "web_scrape" in tool_functions

    def test_tool_functions_callables_match(self):
        """Dict callables are the actual web_search/web_scrape functions."""
        tool_functions = get_browsing_tool_functions()
        
        assert tool_functions["web_search"] == web_search
        assert tool_functions["web_scrape"] == web_scrape


# ─── Provider Resolution Tests ────────────────────────────────────────────

class TestGetDefaultProvider:
    """Tests for get_default_provider() configuration helper."""

    def test_default_provider_is_serper(self):
        """get_default_provider() returns 'serper' when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            provider = get_default_provider()
            assert provider == "serper"

    def test_default_provider_from_env(self):
        """get_default_provider() reads BROWSING_DEFAULT_PROVIDER env var."""
        with patch.dict(os.environ, {"BROWSING_DEFAULT_PROVIDER": "tavily"}):
            provider = get_default_provider()
            assert provider == "tavily"

    def test_default_provider_env_normalized_lowercase(self):
        """Env var value is normalized to lowercase."""
        with patch.dict(os.environ, {"BROWSING_DEFAULT_PROVIDER": "EXA"}):
            provider = get_default_provider()
            assert provider == "exa"

    def test_default_provider_invalid_env_fallback(self):
        """Invalid env var value falls back to 'serper' silently."""
        with patch.dict(os.environ, {"BROWSING_DEFAULT_PROVIDER": "invalid"}):
            provider = get_default_provider()
            assert provider == "serper"


class TestGetRequestProvider:
    """Tests for get_request_provider() resolution logic."""

    def test_request_provider_from_request_arg(self):
        """get_request_provider() returns request_provider if provided."""
        provider = get_request_provider(request_provider="tavily")
        assert provider == "tavily"

    def test_request_provider_normalized_lowercase(self):
        """Request provider is normalized to lowercase."""
        provider = get_request_provider(request_provider="EXA")
        assert provider == "exa"

    def test_request_provider_fallback_to_default(self):
        """get_request_provider() falls back to default when request arg None."""
        with patch.dict(os.environ, {}, clear=True):
            provider = get_request_provider(request_provider=None)
            assert provider == "serper"

    def test_request_provider_override_default(self):
        """get_request_provider() uses default_provider arg if provided."""
        provider = get_request_provider(request_provider=None, default_provider="exa")
        assert provider == "exa"

    def test_request_provider_priority(self):
        """request_provider > default_provider > env default."""
        with patch.dict(os.environ, {"BROWSING_DEFAULT_PROVIDER": "exa"}):
            # request_provider wins
            provider = get_request_provider(request_provider="tavily", default_provider="serper")
            assert provider == "tavily"
            
            # default_provider wins when no request_provider
            provider = get_request_provider(request_provider=None, default_provider="serper")
            assert provider == "serper"
            
            # env default when both None
            provider = get_request_provider(request_provider=None, default_provider=None)
            assert provider == "exa"


# ─── Tool Execution Tests (Mocked Adapter) ─────────────────────────────────

class TestToolExecutionWithMockedAdapter:
    """Tests for web_search/web_scrape calling adapters."""

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    @patch("magic_llm.agent.builtin.browsing.get_browsing_adapter")
    def test_web_search_calls_adapter_search(self, mock_get_adapter):
        """web_search() calls adapter.search() with correct args."""
        mock_adapter = MagicMock()
        mock_adapter.search.return_value = "Mocked search results"
        mock_get_adapter.return_value = mock_adapter
        
        result = web_search("Python tutorials", max_results=5)
        
        mock_get_adapter.assert_called_once_with("serper")
        mock_adapter.search.assert_called_once_with("Python tutorials", max_results=5)
        assert result == "Mocked search results"

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    @patch("magic_llm.agent.builtin.browsing.get_browsing_adapter")
    def test_web_search_with_provider_arg(self, mock_get_adapter):
        """web_search(provider='tavily') uses specified provider."""
        mock_adapter = MagicMock()
        mock_adapter.search.return_value = "Tavily results"
        mock_get_adapter.return_value = mock_adapter
        
        result = web_search("test query", provider="tavily")
        
        mock_get_adapter.assert_called_once_with("tavily")

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    @patch("magic_llm.agent.builtin.browsing.get_browsing_adapter")
    def test_web_scrape_calls_adapter_scrape(self, mock_get_adapter):
        """web_scrape() calls adapter.scrape() with correct args."""
        mock_adapter = MagicMock()
        mock_adapter.scrape.return_value = "Mocked page content"
        mock_get_adapter.return_value = mock_adapter
        
        result = web_scrape("https://example.com/article")
        
        mock_get_adapter.assert_called_once_with("serper")
        mock_adapter.scrape.assert_called_once_with("https://example.com/article")
        assert result == "Mocked page content"


# ─── Module Export Tests ──────────────────────────────────────────────────

class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_builtin_module_exports(self):
        """magic_llm.agent.builtin exports correct functions."""
        from magic_llm.agent.builtin import __all__
        
        expected_exports = [
            "BrowsingAdapter",
            "get_browsing_adapter",
            "get_supported_providers",
            "web_search",
            "web_scrape",
            "get_default_provider",
            "get_request_provider",
            "get_browsing_tools",
            "get_browsing_tool_functions",
        ]
        
        for export in expected_exports:
            assert export in __all__

    def test_browsing_module_exports(self):
        """magic_llm.agent.builtin.browsing exports web_search and web_scrape."""
        from magic_llm.agent.builtin.browsing import __all__
        
        assert "web_search" in __all__
        assert "web_scrape" in __all__

    def test_config_module_exports(self):
        """magic_llm.agent.builtin._config exports config helpers."""
        from magic_llm.agent.builtin._config import __all__
        
        assert "get_default_provider" in __all__
        assert "get_request_provider" in __all__


# ─── Integration: Schema + Callable Match ─────────────────────────────────

class TestSchemaCallableMatch:
    """Tests that schema names match callable names."""

    def test_schema_names_match_callables(self):
        """Schema names match the callable __name__ attribute."""
        tools = get_browsing_tools()
        
        for schema, callable_fn in tools:
            schema_name = schema["function"]["name"]
            assert schema_name == callable_fn.__name__