"""Tests for OpenAI adapter process_chunk methods.

Classical TDD — feed known chunk dicts/strings, assert extracted fields.
No network calls, no HTTP mocks — pure input→output verification.

Covers:
- ProviderGroq: x_groq.usage extraction
- ProviderDeepseek: prompt_cache_hit_tokens handling
- OpenAiBaseProvider: default process_chunk behavior
"""

import pytest

from magic_llm.engine.openai_adapters.openai_groq import ProviderGroq
from magic_llm.engine.openai_adapters.openai_deepseek import ProviderDeepseek
from magic_llm.engine.openai_adapters.openai_base import OpenAiBaseProvider
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel, PromptTokensDetailsModel


# ═══════════════════════════════════════════════════════════════════════════
# Slice 24 — ProviderGroq process_chunk tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGroqProcessChunk:
    """ProviderGroq.process_chunk — extracts x_groq.usage from chunks."""

    def setup_method(self):
        self.provider = ProviderGroq(api_key="test", model="llama-3.1-8b")

    def test_extracts_x_groq_usage(self):
        chunk = (
            'data: {"id":"chat-1","object":"chat.completion.chunk","model":"llama",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],'
            '"x_groq":{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert isinstance(result, ChatCompletionModel)
        assert result.choices[0].delta.content == "Hello"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_empty_choices_skipped(self):
        """Groq chunks with no choices are skipped (usage-only chunk)."""
        chunk = (
            'data: {"id":"chat-1","model":"llama","choices":[],'
            '"x_groq":{"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert result is None

    def test_returns_none_for_done(self):
        result = self.provider.process_chunk("data: [DONE]\n\n")
        assert result is None

    def test_returns_none_for_non_data_line(self):
        result = self.provider.process_chunk(": keep-alive\n\n")
        assert result is None

    def test_usage_defaults_when_missing(self):
        """When x_groq is absent, usage should be empty dict."""
        chunk = (
            'data: {"id":"chat-1","model":"llama",'
            '"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert result is not None
        # Usage should be set to empty dict (converted to UsageModel by ChatCompletionModel)
        assert result.usage is not None


# ═══════════════════════════════════════════════════════════════════════════
# Slice 24 — ProviderDeepseek process_chunk tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDeepseekProcessChunk:
    """ProviderDeepseek.process_chunk — handles prompt_cache_hit_tokens."""

    def setup_method(self):
        self.provider = ProviderDeepseek(api_key="test", model="deepseek-chat")

    def test_extracts_usage_with_cache_details(self):
        chunk = (
            'data: {"id":"chat-1","object":"chat.completion.chunk","model":"deepseek",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],'
            '"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,'
            '"prompt_cache_hit_tokens":50}}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert isinstance(result, ChatCompletionModel)
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 120
        assert result.usage.prompt_tokens_details is not None
        assert result.usage.prompt_tokens_details.cached_tokens == 50

    def test_handles_missing_cache_tokens(self):
        """When prompt_cache_hit_tokens is absent, still parses basic usage."""
        chunk = (
            'data: {"id":"chat-1","model":"deepseek",'
            '"choices":[{"index":0,"delta":{"content":"hi"}}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert result is not None
        assert result.usage.prompt_tokens == 10
        # cached_tokens should be missing from the details or be None
        if result.usage.prompt_tokens_details:
            assert result.usage.prompt_tokens_details.cached_tokens is None

    def test_empty_choices_skipped(self):
        chunk = (
            'data: {"id":"chat-1","model":"deepseek","choices":[],'
            '"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert result is None

    def test_returns_none_for_done(self):
        result = self.provider.process_chunk("data: [DONE]\n\n")
        assert result is None

    def test_no_usage_field(self):
        """Chunk without usage field still processes content."""
        chunk = (
            'data: {"id":"chat-1","model":"deepseek",'
            '"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert result is not None
        assert result.choices[0].delta.content == "hi"


# ═══════════════════════════════════════════════════════════════════════════
# Slice 24 — OpenAiBaseProvider process_chunk tests (default behavior)
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenAiBaseProcessChunk:
    """OpenAiBaseProvider.process_chunk — default SSE chunk handling."""

    def setup_method(self):
        self.provider = OpenAiBaseProvider(base_url="https://api.test.com", api_key="test", model="test-model")

    def test_parses_standard_chunk(self):
        chunk = (
            'data: {"id":"chat-1","object":"chat.completion.chunk","model":"test",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
        )

        result = self.provider.process_chunk(chunk)

        assert isinstance(result, ChatCompletionModel)
        assert result.choices[0].delta.content == "Hello"
        assert result.usage.prompt_tokens == 10

    def test_raises_when_no_choices(self):
        chunk = 'data: {"id":"chat-1","model":"test"}\n\n'

        with pytest.raises(Exception) as exc_info:
            self.provider.process_chunk(chunk)

        assert "no choices" in str(exc_info.value)

    def test_returns_none_for_done(self):
        result = self.provider.process_chunk("data: [DONE]\n\n")
        assert result is None

    def test_returns_none_for_ping(self):
        result = self.provider.process_chunk(": ping\n\n")
        assert result is None

    def test_fallback_for_non_json_chunk(self):
        """Non-data lines that aren't [DONE] or ping get wrapped as content."""
        result = self.provider.process_chunk("some random text")

        assert result is not None
        assert result.choices[0].delta.content == "some random text"
        assert result.model == "dummy"

    def test_empty_choices_skipped(self):
        """Base provider skips usage-only chunks with empty choices."""
        chunk = (
            'data: {"id":"chat-1","model":"test","choices":[],'
            '"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n'
        )
        result = self.provider.process_chunk(chunk)
        assert result is None

    def test_empty_chunk_returns_none(self):
        result = self.provider.process_chunk("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        result = self.provider.process_chunk("   \n\n")
        assert result is None
