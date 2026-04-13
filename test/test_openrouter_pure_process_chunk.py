"""Tests for OpenRouter process_chunk purity."""
import ast
import inspect
import json

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

    def test_no_urllib_import_in_module(self):
        """openai_openrouter module does not import urllib."""
        import magic_llm.engine.openai_adapters.openai_openrouter as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
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
        tree = ast.parse(source)
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
        """process_chunk method does not call time.sleep."""
        provider = ProviderOpenRouter(api_key="test")
        source = inspect.getsource(provider.process_chunk)
        assert "sleep" not in source
        assert "urllib" not in source
        assert "request" not in source.lower() or "http" not in source.lower()
