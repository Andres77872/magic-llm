"""Caller-visible behavior of the text streaming providers in both transports."""
import json
from unittest.mock import patch

import pytest

from magic_llm import MagicLLM
from magic_llm.model import ModelChat


@pytest.mark.parametrize("provider", ["cohere", "cloudflare"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_public_stream_preserves_text_order_and_reported_usage(provider, asynchronous):
    if provider == "cohere":
        kwargs = {}
        frames = [json.dumps(e) for e in [
            {"event_type": "stream-start", "generation_id": "r"},
            {"event_type": "text-generation", "text": "one"},
            {"event_type": "text-generation", "text": " two"},
            {"event_type": "stream-end", "response": {"meta": {"tokens": {"input_tokens": 2, "output_tokens": 3}}}},
        ]]
    else:
        kwargs = {"account_id": "account"}
        frames = ['data: {"response":"one"}', '',
                  'data: {"response":" two","usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}',
                  'data: [DONE]']
    engine = MagicLLM(engine=provider, model="test", private_key="dummy", retries=1, **kwargs).llm
    chat = ModelChat()
    chat.add_user_message("hello")
    target = f"magic_llm.engine.engine_{provider}"
    try:
        if asynchronous:
            async def stream(*args, **kwargs):
                for frame in frames:
                    yield frame.encode()
            with patch(target + ".AsyncHttpClient") as http:
                http.return_value.__aenter__.return_value.post_stream = stream
                chunks = [item async for item in engine.async_stream_generate(chat, timeout=7)]
        else:
            with patch(target + ".HttpClient") as http:
                send = http.return_value.__enter__.return_value.stream_request
                send.return_value = frames
                chunks = list(engine.stream_generate(chat, timeout=7))
                assert send.call_args.kwargs["timeout"] == 7
                assert "timeout" not in json.loads(send.call_args.kwargs["data"])
    finally:
        engine.executor.shutdown(wait=False)
    assert "".join(item.choices[0].delta.content or "" for item in chunks) == "one two"
    assert chunks[-1].usage.total_tokens == 5
