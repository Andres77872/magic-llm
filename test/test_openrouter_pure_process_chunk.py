"""Tests for OpenRouter process_chunk purity."""
import ast
import inspect
import json
import textwrap

import pytest

from magic_llm.engine.openai_adapters.openai_openrouter import ProviderOpenRouter
from magic_llm.model.ModelChatStream import ChatCompletionModel


class TestOpenRouterPureProcessChunk:
    """Test that process_chunk is a pure transformation with no side effects."""

    def test_returns_chat_completion_model_for_valid_chunk(self):
        """process_chunk returns ChatCompletionModel for valid SSE chunk."""
        provider = ProviderOpenRouter(api_key="test")
        chunk_data = {
            "id": "gen-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "openai/gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }],
        }
        result = provider.process_chunk(f"data: {json.dumps(chunk_data)}")
        assert isinstance(result, ChatCompletionModel)

    def test_returns_none_for_done(self):
        """process_chunk returns None for data: [DONE]."""
        provider = ProviderOpenRouter(api_key="test")
        result = provider.process_chunk("data: [DONE]")
        assert result is None

    def test_returns_none_for_done_with_sse_trailing_newlines(self):
        """Regression: process_chunk returns None for 'data: [DONE]\\n\\n' — the exact SSE shape.

        SSE chunks always end with \\n\\n. The previous .endswith('[DONE]') check
        failed because 'data: [DONE]\\n\\n' ends with '\\n\\n', not '[DONE]'.
        This caused json.loads('[DONE]\\n\\n') → JSONDecodeError.
        """
        provider = ProviderOpenRouter(api_key="test")
        result = provider.process_chunk("data: [DONE]\n\n")
        assert result is None

    def test_empty_choices_skipped(self):
        """OpenRouter chunks with empty choices are skipped (usage-only chunk)."""
        provider = ProviderOpenRouter(api_key="test")
        chunk_data = {
            "id": "gen-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "openai/gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = provider.process_chunk(f"data: {json.dumps(chunk_data)}")
        assert result is None

    def test_no_urllib_import_in_module(self):
        """openai_openrouter module does not import urllib."""
        import magic_llm.engine.openai_adapters.openai_openrouter as mod
        source = inspect.getsource(mod)
        tree = ast.parse(textwrap.dedent(source))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        assert not any("urllib" in imp for imp in imports)

    def test_no_time_import_in_module(self):
        """openai_openrouter module does not import time."""
        import magic_llm.engine.openai_adapters.openai_openrouter as mod
        source = inspect.getsource(mod)
        tree = ast.parse(textwrap.dedent(source))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        assert "time" not in imports

    def test_no_sleep_call_in_process_chunk(self):
        """process_chunk has no blocking/network calls; it is a pure transform."""
        provider = ProviderOpenRouter(api_key="test")
        source = inspect.getsource(provider.process_chunk)
        tree = ast.parse(textwrap.dedent(source))
        forbidden_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"sleep", "request"}:
                forbidden_calls.append(func.id)
            elif isinstance(func, ast.Attribute) and func.attr in {
                "sleep",
                "request",
                "post_json",
                "post_raw_binary",
                "post_multipart",
                "post_stream",
                "stream_request",
            }:
                forbidden_calls.append(func.attr)

        assert forbidden_calls == []
