import pytest

from magic_llm.engine.engine_anthropic import EngineAnthropic


def _make_engine():
    return EngineAnthropic(api_key="test-key", model="claude-3-haiku-20240307")


class TestProcessGenerate:
    """process_generate() handling of Anthropic responses."""

    def test_basic_usage_without_cache_tokens(self):
        """Response without cache tokens → usage counts match input/output."""
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150
        assert result.usage.prompt_tokens_details.cached_tokens == 0

    def test_usage_with_cache_read_tokens(self):
        """Response with cache_read_input_tokens → added to prompt and total."""
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        assert result.usage.prompt_tokens == 180  # 100 + 80
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 230  # 180 + 50
        assert result.usage.prompt_tokens_details.cached_tokens == 80

    def test_usage_with_cache_creation_tokens(self):
        """Response with cache_creation_input_tokens → added to prompt and total."""
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 25,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        assert result.usage.prompt_tokens == 125  # 100 + 25
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 175  # 125 + 50
        assert result.usage.prompt_tokens_details.cached_tokens == 0  # only cache_read counts

    def test_usage_with_both_cache_tokens(self):
        """Response with both cache tokens → all added correctly."""
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 25,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        assert result.usage.prompt_tokens == 205  # 100 + 80 + 25
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 255  # 205 + 50
        assert result.usage.prompt_tokens_details.cached_tokens == 80

    def test_usage_matches_streaming_semantics(self):
        """Non-streaming usage must match streaming prepare_chunk semantics."""
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Cached response"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 500,
                "output_tokens": 200,
                "cache_read_input_tokens": 450,
                "cache_creation_input_tokens": 0,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        # Must match what prepare_chunk computes in streaming:
        # prompt_tokens = input + cache_read + cache_creation
        # total_tokens = prompt_tokens + output
        # cached_tokens = cache_read
        assert result.usage.prompt_tokens == 950  # 500 + 450 + 0
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 1150  # 950 + 200
        assert result.usage.prompt_tokens_details.cached_tokens == 450