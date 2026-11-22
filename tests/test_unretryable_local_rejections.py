"""Local rejections that no retry can change: one attempt, fallback kept.

A vision rejection or ``NotImplementedError`` is raised before any request and
repeats on every retry. The default chat wrappers stop retrying, still hand the
request to a configured fallback, and otherwise keep their usual final error.
"""
from unittest.mock import MagicMock, patch

import pytest

from magic_llm import MagicLLM
from magic_llm.engine.base_chat import BaseChat
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message, UsageModel
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=='


class _Chat(BaseChat):
    def generate(self, chat, **kwargs):
        raise AssertionError('not wrapped')

    async def async_generate(self, chat, **kwargs):
        raise AssertionError('not wrapped')

    def stream_generate(self, chat, **kwargs):
        raise AssertionError('not wrapped')

    async def async_stream_generate(self, chat, **kwargs):
        raise AssertionError('not wrapped')


def _chat():
    chat = ModelChat()
    chat.add_user_message('hello')
    return chat


def _response(content='fallback'):
    return ModelChatResponse(
        id='r1', object='chat.completion', created=1, model='fb',
        choices=[Choice(index=0, message=Message(role='assistant', content=content), finish_reason='stop')],
        usage=UsageModel(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _chunk(content='fallback'):
    return ChatCompletionModel(id='c1', model='fb', usage=UsageModel(),
                               choices=[ChoiceModel(index=0, delta=DeltaModel(content=content),
                                                    finish_reason='stop')])


def _vision_error():
    return ChatException('Model cannot process images', error_code='VISION_NOT_SUPPORTED')


REJECTIONS = [pytest.param(_vision_error, id='vision'), pytest.param(NotImplementedError, id='not-implemented')]


class _Calls:
    def __init__(self, error):
        self.count = 0
        self.error = error

    def generate(self, _self, chat, **kwargs):
        self.count += 1
        raise self.error()

    async def async_generate(self, _self, chat, **kwargs):
        self.count += 1
        raise self.error()

    def stream(self, _self, chat, **kwargs):
        self.count += 1
        raise self.error()
        yield  # pragma: no cover

    async def async_stream(self, _self, chat, **kwargs):
        self.count += 1
        raise self.error()
        yield  # pragma: no cover


async def _run(mode, calls, instance):
    if mode == 'sync':
        return BaseChat.sync_intercept_generate(calls.generate)(instance, _chat())
    if mode == 'async':
        return await BaseChat.async_intercept_generate(calls.async_generate)(instance, _chat())
    if mode == 'sync_stream':
        return list(BaseChat.sync_intercept_stream_generate(calls.stream)(instance, _chat()))
    return [c async for c in BaseChat.async_intercept_stream_generate(calls.async_stream)(instance, _chat())]


MODES = ['sync', 'async', 'sync_stream', 'async_stream']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('error', REJECTIONS)
async def test_rejection_is_attempted_once_and_keeps_the_final_error(mode, error):
    calls = _Calls(error)
    statuses = []
    instance = _Chat(retries=3, callback=lambda chat, content, usage, model, meta: statuses.append(meta.status))
    with patch('magic_llm.engine.base_chat.time.sleep') as sleep, \
            patch('magic_llm.engine.base_chat.asyncio.sleep') as async_sleep:
        with pytest.raises(ChatException) as caught:
            await _run(mode, calls, instance)
    assert calls.count == 1
    assert not sleep.called and not async_sleep.called
    assert caught.value.error_code == 'STREAM_GENERATION_ERROR'
    assert 'after 1 attempts' in caught.value.message
    assert isinstance(caught.value.__context__, (ChatException, NotImplementedError))
    assert len([s for s in statuses if s and s.startswith('ERROR')]) == 1
    instance.executor.shutdown(wait=False)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('error', REJECTIONS)
async def test_rejection_still_uses_the_fallback_once(mode, error):
    calls = _Calls(error)
    fallback = MagicMock()
    fallback.llm.generate.return_value = _response()

    async def async_generate(chat, **kwargs):
        return _response()

    def stream_generate(chat, **kwargs):
        yield _chunk()

    async def async_stream_generate(chat, **kwargs):
        yield _chunk()

    fallback.llm.async_generate = async_generate
    fallback.llm.stream_generate = stream_generate
    fallback.llm.async_stream_generate = async_stream_generate
    instance = _Chat(retries=3, fallback=fallback)
    with patch('magic_llm.engine.base_chat.time.sleep'), patch('magic_llm.engine.base_chat.asyncio.sleep'):
        result = await _run(mode, calls, instance)
    assert calls.count == 1
    if mode.endswith('stream'):
        assert [c.choices[0].delta.content for c in result] == ['fallback']
    else:
        assert result.content == 'fallback'
    instance.executor.shutdown(wait=False)


@pytest.mark.parametrize('mode', MODES)
async def test_transient_errors_keep_retrying(mode):
    calls = _Calls(lambda: ConnectionError('transient'))
    instance = _Chat(retries=3)
    with patch('magic_llm.engine.base_chat.time.sleep'), patch('magic_llm.engine.base_chat.asyncio.sleep'):
        with pytest.raises(ChatException) as caught:
            await _run(mode, calls, instance)
    assert calls.count == 3
    assert 'after 3 attempts' in caught.value.message
    instance.executor.shutdown(wait=False)


async def test_real_adapter_vision_rejection_is_not_retried():
    client = MagicLLM(engine='openai', model='m', private_key='fake-test-only',
                      base_url='https://api.groq.com/openai/v1', retries=3)
    chat = ModelChat()
    chat.add_user_message('Describe', image=f'data:image/png;base64,{PNG}')
    with patch('magic_llm.engine.base_chat.time.sleep') as sleep:
        with pytest.raises(ChatException) as caught:
            client.llm.generate(chat)
    assert not sleep.called
    assert caught.value.error_code == 'STREAM_GENERATION_ERROR'
    assert caught.value.__context__.error_code == 'VISION_NOT_SUPPORTED'


async def test_azure_chat_is_not_retried():
    client = MagicLLM(engine='azure', speech_key='fake-test-only', speech_region='eastus', retries=3)
    with patch('magic_llm.engine.base_chat.asyncio.sleep') as sleep:
        with pytest.raises(ChatException) as caught:
            await client.llm.async_generate(_chat())
    assert not sleep.called
    assert caught.value.error_code == 'STREAM_GENERATION_ERROR'
    assert isinstance(caught.value.__context__, NotImplementedError)


async def test_private_transport_does_not_list_models(monkeypatch):
    def no_http(*args, **kwargs):
        raise AssertionError('Private transport keys must not reach the listing client')

    monkeypatch.setattr('magic_llm.engine.discovery.base_discovery.HttpClient', no_http)
    monkeypatch.setattr('magic_llm.engine.discovery.base_discovery.AsyncHttpClient', no_http)
    client = MagicLLM(engine='openai', model='vendor/model', private_key='fake-test-only',
                      base_url='https://llm.vendor.example/v1', private_transport=True)
    with pytest.raises(NotImplementedError, match='private_transport'):
        client.list_models()
    with pytest.raises(NotImplementedError, match='private_transport'):
        await client.async_list_models()
