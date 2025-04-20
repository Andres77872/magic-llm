import pytest
import json

from magic_llm import MagicLLM
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat


def _get_chat_builder():
    chat = ModelChat()
    chat.add_user_message('hi')
    return chat


def _get_callback(msg: ModelChat,
                  content: str,
                  usage,
                  model: str,
                  meta):
    print('\n\n' + '=' * 50)
    print('==MODEL==', model)
    print('==USAGE==', usage)
    print('==GENERATED==', content)
    print('==INFO==', meta)
    print('=' * 50)


OPENAI_KEY = json.load(open('/home/andres/Documents/keys.json'))['novita.ai']

OPENAI_KEY["callback"] = _get_callback
MODEL = 'mistralai/mistral-nemo'
MODEL_FAIL = 'FAIL/mistralai/mistral-nemo'


def _get_fallback_client():
    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    return client


def test_sync_openai_base_stream_generate_1():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = ''
    for i in client.llm.stream_generate(chat):
        content += i.choices[0].delta.content or ''
    assert content != ''


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_1():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = ''
    async for i in client.llm.async_stream_generate(chat):
        content += i.choices[0].delta.content or ''
    assert content != ''


def test_sync_openai_base_stream_generate_2():
    chat = _get_chat_builder()
    client = MagicLLM(
        model=MODEL_FAIL,
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = ''
        for i in client.llm.stream_generate(chat):
            content += i.choices[0].delta.content or ''
        assert content == ''


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_2():
    chat = _get_chat_builder()
    client = MagicLLM(
        model=MODEL_FAIL,
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = ''
        async for i in client.llm.async_stream_generate(chat):
            content += i.choices[0].delta.content or ''
        assert content.startswith('Stream generation attempt 3')


def test_sync_openai_base_stream_generate_3():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **OPENAI_KEY
    )
    content = ''
    for i in client.llm.stream_generate(chat):
        content += i.choices[0].delta.content or ''
    assert content != ''


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_3():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **OPENAI_KEY
    )
    content = ''
    async for i in client.llm.async_stream_generate(chat):
        content += i.choices[0].delta.content or ''
    assert content != ''


def test_sync_openai_base_stream_generate_4():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = ''
    usage = None
    for i in client.llm.stream_generate(chat):
        content += i.choices[0].delta.content or ''
        usage = i.usage
    assert usage is not None
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.tps > 0
    assert usage.ttf > 0
    assert usage.ttft > 0
    assert content != ''


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_4():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = ''
    usage = None
    async for i in client.llm.async_stream_generate(chat):
        content += i.choices[0].delta.content or ''
        usage = i.usage
    assert usage is not None
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.tps > 0
    assert usage.ttf > 0
    assert usage.ttft > 0
    assert content != ''


def test_sync_openai_base_stream_generate_5():
    chat = _get_chat_builder()
    cr = 0

    def _callback_test(msg: ModelChat,
                       content: str,
                       usage,
                       model: str,
                       meta):
        nonlocal cr
        cr += 1
        print('\n\n' + '=' * 50)
        print('==MODEL==', model)
        print('==USAGE==', usage)
        print('==GENERATED==', content)
        print('==INFO==', meta)
        print('=' * 50)

    k = OPENAI_KEY.copy()
    k['callback'] = _callback_test
    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **k
    )
    content = ''
    for i in client.llm.stream_generate(chat):
        content += i.choices[0].delta.content or ''
    assert cr == 4
    assert content != ''


@pytest.mark.asyncio
async def test_async_openai_base_stream_generate_5():
    chat = _get_chat_builder()
    cr = 0

    def _callback_test(msg: ModelChat,
                       content: str,
                       usage,
                       model: str,
                       meta):
        nonlocal cr
        cr += 1
        print('\n\n' + '=' * 50)
        print('==MODEL==', model)
        print('==USAGE==', usage)
        print('==GENERATED==', content)
        print('==INFO==', meta)
        print('=' * 50)

    k = OPENAI_KEY.copy()
    k['callback'] = _callback_test
    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **k
    )
    content = ''
    async for i in client.llm.async_stream_generate(chat):
        content += i.choices[0].delta.content or ''
    assert cr == 4
    assert content != ''


def test_sync_openai_base_non_stream_generate_1():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = client.llm.generate(chat)
    assert content.content != ''


@pytest.mark.asyncio
async def test_async_openai_base_non_stream_generate_1():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = await client.llm.async_generate(chat)
    assert content.content != ''


def test_sync_openai_base_non_stream_generate_2():
    chat = _get_chat_builder()
    client = MagicLLM(
        model=MODEL_FAIL,
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = client.llm.generate(chat)
        assert content is None


@pytest.mark.asyncio
async def test_async_openai_base_non_stream_generate_2():
    chat = _get_chat_builder()
    client = MagicLLM(
        model=MODEL_FAIL,
        **OPENAI_KEY
    )
    with pytest.raises(ChatException):
        content = await client.llm.async_generate(chat)
        assert content is None


def test_sync_openai_base_non_stream_generate_3():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **OPENAI_KEY
    )
    content = client.llm.generate(chat)
    assert content.content != ''


@pytest.mark.asyncio
async def test_async_openai_base_non_stream_generate_3():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **OPENAI_KEY
    )
    content = await client.llm.async_generate(chat)
    assert content.content != ''


def test_sync_openai_base_non_stream_generate_4():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = client.llm.generate(chat)
    usage = content.usage
    assert usage is not None
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.tps > 0
    assert usage.ttf > 0
    assert usage.ttft == 0
    assert content.content != ''


@pytest.mark.asyncio
async def test_async_openai_base_non_stream_generate_4():
    chat = _get_chat_builder()

    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    content = await client.llm.async_generate(chat)
    usage = content.usage
    assert usage is not None
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.tps > 0
    assert usage.ttf > 0
    assert usage.ttft == 0
    assert content.content != ''


def test_sync_openai_base_non_stream_generate_5():
    chat = _get_chat_builder()
    cr = 0

    def _callback_test(msg: ModelChat,
                       content: str,
                       usage,
                       model: str,
                       meta):
        nonlocal cr
        cr += 1
        print('\n\n' + '=' * 50)
        print('==MODEL==', model)
        print('==USAGE==', usage)
        print('==GENERATED==', content)
        print('==INFO==', meta)
        print('=' * 50)

    k = OPENAI_KEY.copy()
    k['callback'] = _callback_test
    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **k
    )
    content = client.llm.generate(chat)
    assert cr == 3
    assert content.content != ''


@pytest.mark.asyncio
async def test_async_openai_base_non_stream_generate_5():
    chat = _get_chat_builder()
    cr = 0

    def _callback_test(msg: ModelChat,
                       content: str,
                       usage,
                       model: str,
                       meta):
        nonlocal cr
        cr += 1
        print('\n\n' + '=' * 50)
        print('==MODEL==', model)
        print('==USAGE==', usage)
        print('==GENERATED==', content)
        print('==INFO==', meta)
        print('=' * 50)

    k = OPENAI_KEY.copy()
    k['callback'] = _callback_test
    client = MagicLLM(
        model=MODEL_FAIL,
        fallback=_get_fallback_client(),
        **k
    )
    content = await client.llm.async_generate(chat)
    assert cr == 3
    assert content.content != ''
