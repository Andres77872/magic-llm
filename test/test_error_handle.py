import pytest

from magic_llm import MagicLLM
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat

from conftest import get_provider_key

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional


def _get_chat_builder():
    chat = ModelChat()
    chat.add_user_message('hi')
    return chat


def _openai_key(provider_keys):
    return get_provider_key(provider_keys, 'openai', 'openai')


def _get_fallback_client(provider_keys):
    client = MagicLLM(
        model='gpt-4o',
        **_openai_key(provider_keys),
    )
    return client


def test_sync_error_1(provider_keys):
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o1',
        **_openai_key(provider_keys),
    )
    with pytest.raises(ChatException):
        content = ''
        for i in client.llm.stream_generate(chat):
            content += i.choices[0].delta.content or ''


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_1(provider_keys):
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o1',
        **_openai_key(provider_keys),
    )
    with pytest.raises(ChatException):
        content = ''
        async for i in client.llm.async_stream_generate(chat):
            content += i.choices[0].delta.content or ''


def test_sync_openai_base_stream_generate_2(provider_keys):
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o-model-fail',
        **_openai_key(provider_keys),
    )
    with pytest.raises(ChatException):
        content = client.llm.generate(chat)


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_2(provider_keys):
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o-model-fail',
        **_openai_key(provider_keys),
    )
    with pytest.raises(ChatException):
        content = await client.llm.async_generate(chat)
