"""OpenAI reports streaming usage in a final chunk with no choices."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.engine.openai_adapters.openai_base import ProviderOpenAI
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel


def sse(choices, **fields):
    return "data: " + json.dumps({
        "id": "chatcmpl-usage", "model": "gpt-4o-mini", "choices": choices, **fields,
    })


def usage_frame():
    return sse([], service_tier="default", usage={
        "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25,
        "prompt_tokens_details": {"cached_tokens": 8},
        "completion_tokens_details": {"reasoning_tokens": 2},
    })


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "length"])
def test_usage_only_frame_keeps_usage_and_finish_reason_without_replaying_delta(finish_reason):
    provider = ProviderOpenAI(api_key="test")
    previous = ChatCompletionModel(id="chatcmpl-usage", model="gpt-4o-mini", choices=[{
        "index": 0, "finish_reason": finish_reason, "delta": {
            "content": "already delivered", "tool_calls": [{
                "id": "call-1", "function": {"name": "lookup", "arguments": "{}"},
            }],
        },
    }])

    result = provider.process_chunk(usage_frame(), last_chunk=previous)

    assert result is not None
    assert result.usage.total_tokens == 25
    assert result.usage.prompt_tokens_details.cached_tokens == 8
    assert result.usage.completion_tokens_details.reasoning_tokens == 2
    assert result.usage.provider_request_id == "chatcmpl-usage"
    assert result.usage.service_tier == "default"
    assert result.choices[0].finish_reason == finish_reason
    assert result.choices[0].delta.content == ""
    assert not result.choices[0].delta.tool_calls
    assert previous.choices[0].delta.content == "already delivered"


def test_usage_only_frame_without_previous_chunk_has_safe_empty_choice():
    result = ProviderOpenAI(api_key="test").process_chunk(usage_frame())
    assert result is not None
    assert result.choices[0].delta.content == ""
    assert result.choices[0].finish_reason is None
    assert result.usage.total_tokens == 25


@pytest.mark.parametrize("fields", [{}, {"usage": None}, {"usage": {}}])
def test_empty_choices_without_usage_still_skipped(fields):
    assert ProviderOpenAI(api_key="test").process_chunk(sse([], **fields)) is None


def provider_frames():
    yield sse([{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}])
    yield sse([{"index": 0, "delta": {}, "finish_reason": "stop"}])
    yield usage_frame()
    yield "data: [DONE]"


def assert_engine_usage(chunks, callback):
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
    assert chunks[-1].usage.total_tokens == 25
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.provider_extra["attempts"][0]["status"] == "completed"
    callback.assert_called_once()
    assert callback.call_args.args[1] == "Hello"
    assert callback.call_args.args[2].total_tokens == 25


def test_sync_stream_preserves_usage_in_final_event_and_callback():
    callback = MagicMock()
    engine = EngineOpenAI(api_key="test", model="gpt-4o-mini", callback=callback)
    with patch("magic_llm.engine.engine_openai.HttpClient") as client_class:
        client = client_class.return_value.__enter__.return_value
        client.stream_request.return_value = provider_frames()
        chunks = list(engine.stream_generate(ModelChat(system="test")))
    assert_engine_usage(chunks, callback)


@pytest.mark.asyncio
async def test_async_stream_preserves_usage_in_final_event_and_callback():
    callback = AsyncMock()
    engine = EngineOpenAI(api_key="test", model="gpt-4o-mini", callback=callback)

    async def stream(*args, **kwargs):
        for frame in provider_frames():
            yield frame.encode()

    with patch("magic_llm.engine.engine_openai.AsyncHttpClient") as client_class:
        client = AsyncMock()
        client.post_stream = stream
        client.__aenter__.return_value = client
        client_class.return_value = client
        chunks = [chunk async for chunk in engine.async_stream_generate(ModelChat(system="test"))]
    assert_engine_usage(chunks, callback)
