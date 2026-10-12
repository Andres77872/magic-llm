"""Provider bodies exclude local HTTP timeout controls."""
import json

import pytest

from magic_llm.engine.engine_anthropic import EngineAnthropic
from magic_llm.engine.engine_cohere import EngineCohere
from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.model import ModelChat


@pytest.mark.parametrize('engine', [EngineOpenAI, EngineAnthropic, EngineGoogle, EngineCohere])
def test_transport_timeout_is_not_serialized_into_a_model_request(engine):
    provider = engine(api_key='FAKE_TEST_KEY', model='test-model')
    chat = ModelChat()
    chat.add_user_message('Hello.')
    options = {'timeout': 10.0, 'temperature': 0.6}
    try:
        body, _ = provider.transform_request(chat, **options)
        payload = json.loads(body)
        assert 'timeout' not in payload and 'timeout' not in payload.get('generationConfig', {})
        assert payload.get('generationConfig', payload)['temperature'] == 0.6
        assert options == {'timeout': 10.0, 'temperature': 0.6}
    finally:
        provider.executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_async_google_transform_keeps_timeout_out_of_generation_config():
    provider = EngineGoogle(api_key='FAKE_TEST_KEY', model='test-model')
    try:
        chat = ModelChat()
        chat.add_user_message('Hello.')
        body, _ = await provider.async_transform_request(chat, timeout=10)
        assert 'timeout' not in json.loads(body)['generationConfig']
    finally:
        provider.executor.shutdown(wait=False)
