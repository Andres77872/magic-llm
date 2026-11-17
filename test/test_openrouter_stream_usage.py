"""OpenRouter: usage, cache and cost come from the stream, and nothing is polled."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.engine.openai_adapters.openai_openrouter import ProviderOpenRouter
from magic_llm.model import ModelChat

FRAMES = [
    ": OPENROUTER PROCESSING",
    "data: " + json.dumps({"id": "gen-1", "model": "openai/gpt-4", "choices": [
        {"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]}),
    "data: " + json.dumps({"id": "gen-1", "model": "openai/gpt-4", "choices": [
        {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002,
                  "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 6}}}),
    "data: [DONE]",
]


def _engine():
    engine = EngineOpenAI(api_key="test", model="openai/gpt-4", base_url="https://openrouter.ai/api/v1")
    assert isinstance(engine.base, ProviderOpenRouter)
    return engine


@pytest.mark.asyncio
async def test_async_stream_reads_usage_from_the_stream_and_makes_no_second_request():
    engine = _engine()

    async def stream(*args, **kwargs):
        for frame in FRAMES:
            yield frame.encode()

    with patch("magic_llm.engine.engine_openai.AsyncHttpClient") as client_class:
        client = AsyncMock()
        client.post_stream = stream
        client.request = AsyncMock(side_effect=AssertionError("no polling request"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client_class.return_value = client
        chunks = [chunk async for chunk in engine.async_stream_generate(ModelChat(system="test"))]

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hi"
    usage = chunks[-1].usage
    assert (usage.prompt_tokens, usage.completion_tokens) == (10, 5)
    assert usage.prompt_tokens_details.cached_tokens == 4
    assert usage.cached_tokens_write == 6
    assert usage.provider_extra["cost"] == 0.002
    assert usage.provider_request_id == "gen-1"
    client.request.assert_not_called()
