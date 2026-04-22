"""Builtin tools module for magic_llm agent package.

This module provides reusable builtin tools that can be injected into
agent loops. Currently supports web_search and web_scrape for browsing
capabilities.

Per design.md:
- Tools are plain Python callables with docstrings for schema generation
- Provider selection is resolved via get_browsing_adapter()
- Tools return plain text optimized for model consumption
- get_browsing_tools() returns (schema, callable) tuples for injection

This module follows the lazy import pattern of magic_llm.agent —
importing magic_llm.agent alone does NOT import builtin modules.
"""

from typing import List, Tuple, Dict, Any

from magic_llm.agent.builtin.adapters import (
    BrowsingAdapter,
    get_browsing_adapter,
    get_supported_providers,
)
from magic_llm.agent.builtin.browsing import web_search, web_scrape
from magic_llm.agent.builtin._config import get_default_provider, get_request_provider


def get_browsing_tools(
    provider: str = None,
    include_callables: bool = True
) -> List[Tuple[Dict[str, Any], Any]]:
    """Get browsing tools as (schema, callable) tuples for injection.

    Args:
        provider: Provider to use for tools. If None, uses default provider.
        include_callables: If True, returns (schema, callable) tuples.
                          If False, returns only schema dicts.

    Returns:
        List of tool tuples: [(schema_dict, callable), ...]
        Each schema_dict is OpenAI-compatible tool definition.

    Example:
        >>> tools = get_browsing_tools()
        >>> schemas = [t[0] for t in tools]  # OpenAI tool schemas
        >>> callables = {t[0]["function"]["name"]: t[1] for t in tools}  # tool_functions dict
    """
    # Import here to avoid circular import at module level
    from magic_llm.util.tools_mapping import _schema_from_callable

    resolved_provider = provider or get_default_provider()

    # Generate schemas from callables using magic-llm's schema generator
    search_name, search_desc, search_params = _schema_from_callable(web_search)
    scrape_name, scrape_desc, scrape_params = _schema_from_callable(web_scrape)

    # Build OpenAI-compatible tool schema format
    search_schema = {
        "type": "function",
        "function": {
            "name": search_name,
            "description": search_desc,
            "parameters": search_params,
        }
    }

    scrape_schema = {
        "type": "function",
        "function": {
            "name": scrape_name,
            "description": scrape_desc,
            "parameters": scrape_params,
        }
    }

    if include_callables:
        return [
            (search_schema, web_search),
            (scrape_schema, web_scrape),
        ]
    else:
        return [
            (search_schema, None),
            (scrape_schema, None),
        ]


def get_browsing_tool_functions(provider: str = None) -> Dict[str, Any]:
    """Get tool_functions dict for browsing tools.

    Args:
        provider: Provider to use. If None, uses default provider.

    Returns:
        Dict mapping tool names to callables:
        {"web_search": web_search_fn, "web_scrape": web_scrape_fn}

    This is a convenience function for building tool_functions dict
    to pass to AsyncAgentLoop or ToolExecutor.
    """
    return {
        "web_search": web_search,
        "web_scrape": web_scrape,
    }


__all__ = [
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