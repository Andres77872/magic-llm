"""Compatibility wrapper for engine/core-owned tooling.

Canonical tooling lives in :mod:`magic_llm.engine.tooling`. This module remains
only for backwards-compatible imports.
"""

from magic_llm.engine.tooling import (  # noqa: F401
    OpenAITool,
    ToolChoice,
    _extract_param_docs_from_docstring,
    _inline_local_refs,
    _is_pydantic_model,
    _json_schema_for_annotation,
    _schema_from_callable,
    _schema_from_pydantic,
    coerce_tool_choice_to_string,
    map_to_anthropic,
    map_to_gemini,
    map_to_openai,
    normalize_openai_tool_choice,
    normalize_openai_tools,
)

__all__ = [
    "OpenAITool",
    "ToolChoice",
    "_extract_param_docs_from_docstring",
    "_inline_local_refs",
    "_is_pydantic_model",
    "_json_schema_for_annotation",
    "_schema_from_callable",
    "_schema_from_pydantic",
    "coerce_tool_choice_to_string",
    "map_to_anthropic",
    "map_to_gemini",
    "map_to_openai",
    "normalize_openai_tool_choice",
    "normalize_openai_tools",
]
