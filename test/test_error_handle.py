import json
import os

import pytest

from magic_llm import MagicLLM
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional


def _get_chat_builder():
    chat = ModelChat()
    chat.add_user_message('hi')
    return chat


_KEYS_FILE = os.getenv("MAGIC_LLM_KEYS")
if not _KEYS_FILE or not os.path.exists(_KEYS_FILE):
    pytest.skip(
        "MAGIC_LLM_KEYS env var must point to a valid keys file for integration tests.",
        allow_module_level=True,
    )
OPENAI_KEY = json.load(open(_KEYS_FILE))['openai']


def _get_fallback_client():
    client = MagicLLM(
        model='gpt-4o',
        **OPENAI_KEY
    )
    return client


def test_sync_error_1():
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o1',
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = ''
        for i in client.llm.stream_generate(chat):
            content += i.choices[0].delta.content or ''


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_1():
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o1',
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = ''
        async for i in client.llm.async_stream_generate(chat):
            content += i.choices[0].delta.content or ''


def test_sync_openai_base_stream_generate_2():
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o-model-fail',
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = client.llm.generate(chat)


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_2():
    chat = _get_chat_builder()
    client = MagicLLM(
        model='gpt-4o-model-fail',
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = await client.llm.async_generate(chat)
