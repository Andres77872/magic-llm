import pytest
"""Tests for OpenRouter stream-chunk transformation."""
import json

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

    def test_usage_only_frame_is_kept(self):
        """A usage frame with no choice keeps its usage and replays no delta."""
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
        assert result.usage.prompt_tokens == 5 and result.usage.completion_tokens == 3
        assert result.choices[0].delta.content == ""

    def test_frame_without_choice_or_usage_is_skipped(self):
        provider = ProviderOpenRouter(api_key="test")
        assert provider.process_chunk('data: {"id": "gen-1", "choices": []}') is None

    def test_keep_alive_comments_carry_nothing(self):
        provider = ProviderOpenRouter(api_key="test")
        assert provider.process_chunk(": OPENROUTER PROCESSING") is None
        assert provider.process_chunk("event: ping") is None

    @pytest.mark.parametrize("frame,status,error_type", [
        ({"error": {"code": 429, "message": "slow down"}}, 429, None),
        ({"error": {"code": 400, "message": "too long", "metadata": {"error_type": "context_length_exceeded"}}},
         400, "context_length_exceeded"),
        ({"choices": [{"index": 0, "delta": {}, "finish_reason": "error",
                       "error": {"message": "upstream died", "metadata": {"error_type": "provider_unavailable"}}}]},
         502, "provider_unavailable"),
    ])
    def test_errors_inside_the_stream_fail_the_call_with_their_type(self, frame, status, error_type):
        from magic_llm.util.http import HttpError
        provider = ProviderOpenRouter(api_key="test")
        with pytest.raises(HttpError) as raised:
            provider.process_chunk("data: " + json.dumps({"id": "gen-1", **frame}))
        assert raised.value.status_code == status
        assert raised.value.error_type == error_type
        assert "slow down" not in str(raised.value) and "upstream died" not in str(raised.value)

    def test_context_overflow_mid_stream_is_classified(self):
        from magic_llm.util.http import HttpError
        provider = ProviderOpenRouter(api_key="test")
        with pytest.raises(HttpError) as raised:
            provider.process_chunk('data: {"error": {"code": 400, "metadata": {"error_type": "context_length_exceeded"}}}')
        assert raised.value.code == "PROVIDER_CONTEXT_EXCEEDED"

    def test_byok_cost_adds_the_upstream_charge(self):
        provider = ProviderOpenRouter(api_key="test")
        frame = {"id": "g", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0001,
                 "is_byok": True, "cost_details": {"upstream_inference_cost": 0.01}}}
        result = provider.process_chunk("data: " + json.dumps(frame))
        assert result.usage.provider_extra["cost"] == pytest.approx(0.0101)

    def test_a_response_with_no_choice_is_a_typed_failure(self):
        from magic_llm.util.http import HttpError
        provider = ProviderOpenRouter(api_key="test")
        with pytest.raises(HttpError) as raised:
            provider.transform_response({"id": "g", "choices": []})
        assert raised.value.error_type == "no_content"
