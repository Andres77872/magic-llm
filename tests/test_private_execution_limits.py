import json
from dataclasses import replace

import pytest

from magic_llm import MagicLLM
from magic_llm.exception.ChatException import ChatException, RequestValidationError
from magic_llm.model import ModelChat
from magic_llm.util.private_limits import PrivateRequestLimits

CANARY = 'FAKE_PRIVATE_PROVIDER_ERROR_CANARY'


pytestmark = pytest.mark.usefixtures('offline_token_estimator')


@pytest.fixture
def limits():
    return PrivateRequestLimits(model='approved-model', max_input_tokens=1024,
        max_output_tokens=64, max_messages=10, timeout_seconds=12, temperature=.4)


class Provider:
    def __init__(self, calls, *, fail=False):
        self.calls, self.fail = calls, fail

    async def __aenter__(self): return self
    async def __aexit__(self, *args): return None

    def receive(self, kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ValueError(CANARY)

    async def post_json(self, **kwargs):
        self.receive(kwargs)
        return {'id': 'safe-response', 'object': 'chat.completion', 'created': 0, 'model': 'approved-model',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'safe'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}

    async def post_stream(self, url, **kwargs):
        self.receive({'url': url, **kwargs})
        yield b'data: {"id":"safe-response","model":"approved-model","choices":[{"index":0,"delta":{"content":"safe"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}'


def client(limits, *, retries=1, **defaults):
    return MagicLLM(engine='openai', model=limits.model, private_key='FAKE_KEY',
        base_url='https://provider.example/v1', private_transport=True, private_limits=limits, retries=retries, **defaults)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('graph_cap,expected', [(500, 64), (24, 24)])
async def test_final_request_uses_profile_settings_and_caps(monkeypatch, limits, stream, graph_cap, expected):
    facade, calls = client(limits), []
    monkeypatch.setattr('magic_llm.engine.engine_openai.PrivateAsyncHttpClient', lambda: Provider(calls))
    chat = ModelChat(); chat.add_user_message('hello')
    arguments = {'max_tokens': graph_cap, 'temperature': 1.8, 'top_p': .1,
                 'reasoning': {'effort': 'high'}, 'timeout': 100}
    if stream:
        result = [item async for item in facade.llm.async_stream_generate(chat, **arguments)]
        assert result[0].choices[0].delta.content == 'safe'
    else:
        assert (await facade.llm.async_generate(chat, **arguments)).content == 'safe'
    body = json.loads(calls[0]['data'])
    assert body['max_tokens'] == expected and body['temperature'] == .4
    assert 'top_p' not in body and 'reasoning' not in body
    assert 'timeout' not in body and 'private_transport' not in body and 'private_limits' not in body
    assert body['model'] == 'approved-model' and body['store'] is False
    assert calls[0]['timeout'] == 12
    assert chat.max_input_tokens is None and not chat.complete_context_required
    facade.llm.executor.shutdown(wait=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('model', 'other'), ('extra_body', {'model': 'other'}),
    ('max_tokens', 0), ('max_tokens', True), ('timeout', float('inf')), ('n', 2),
    ('previous_response_id', 'provider-stored-context')])
async def test_overrides_fail_before_provider_dispatch(monkeypatch, limits, field, value):
    facade, calls = client(limits), []
    monkeypatch.setattr('magic_llm.engine.engine_openai.PrivateAsyncHttpClient', lambda: Provider(calls))
    chat = ModelChat(); chat.add_user_message('hello')
    with pytest.raises(RequestValidationError):
        await facade.llm.async_generate(chat, **{field: value})
    assert calls == []
    facade.llm.executor.shutdown(wait=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['message_count', 'tokens', 'tool_schema'])
async def test_mapped_context_includes_tools_and_message_limits(monkeypatch, limits, kind):
    limits = replace(limits, max_messages=1 if kind == 'message_count' else 10,
                     max_input_tokens=20 if kind != 'message_count' else 1024)
    facade, calls = client(limits), []
    monkeypatch.setattr('magic_llm.engine.engine_openai.PrivateAsyncHttpClient', lambda: Provider(calls))
    chat = ModelChat(); chat.add_user_message('hello')
    kwargs = {}
    if kind == 'message_count': chat.add_user_message('again')
    elif kind == 'tokens': chat.messages[0]['content'] = 'word ' * 100
    else:
        kwargs['tools'] = [{'type': 'function', 'function': {'name': 'valid',
            'description': 'word ' * 100, 'parameters': {'type': 'object', 'properties': {}}}}]
    with pytest.raises(RequestValidationError): await facade.llm.async_generate(chat, **kwargs)
    assert calls == []
    facade.llm.executor.shutdown(wait=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_private_errors_are_safe_in_logs_callbacks_and_output(monkeypatch, limits, caplog, stream):
    facade, calls, observed = client(limits), [], []
    monkeypatch.setattr('magic_llm.engine.engine_openai.PrivateAsyncHttpClient', lambda: Provider(calls, fail=True))
    facade.llm.callback = lambda *args: observed.append(args[-1].model_dump_json())
    chat = ModelChat(); chat.add_user_message('hello')
    emitted = []
    with pytest.raises(ChatException) as error:
        if stream:
            async for item in facade.llm.async_stream_generate(chat): emitted.append(item.model_dump_json())
        else: await facade.llm.async_generate(chat)
    assert CANARY not in str(error.value) + caplog.text + ''.join(emitted) + ''.join(observed)
    assert len(calls) == 1
    facade.llm.executor.shutdown(wait=False)


@pytest.mark.parametrize('url', ['https://api.together.xyz/custom/v1',
    'https://provider.example/api.groq.com/v1', 'https://openrouter.ai/api/v2'])
def test_private_destination_is_never_replaced_by_adapter_discovery(limits, url):
    facade = MagicLLM(engine='openai', model=limits.model, private_key='FAKE_KEY', base_url=url,
                      private_transport=True, private_limits=limits, retries=1)
    assert facade.llm.base.base_url == url
    assert facade.llm.chat_url == url + '/chat/completions'
    facade.llm.executor.shutdown(wait=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_attempt_control_sees_effective_limits(monkeypatch, limits, stream):
    facade, calls, attempts = client(limits, top_p=.7, reasoning={'effort': 'high'}), [], []
    monkeypatch.setattr('magic_llm.engine.engine_openai.PrivateAsyncHttpClient', lambda: Provider(calls))
    class Control:
        async def before_attempt(self, attempt): attempts.append(attempt)
        async def after_attempt(self, attempt, outcome): pass
    chat = ModelChat(); chat.add_user_message('hello')
    options = dict(max_tokens=999, temperature=1.8, timeout=90, provider_attempt_control=Control())
    if stream:
        assert [c async for c in facade.llm.async_stream_generate(chat, **options)]
    else:
        await facade.llm.async_generate(chat, **options)
    assert not {'top_p', 'reasoning'} & attempts[0].generation_options.keys()
    assert not {'top_p', 'reasoning'} & json.loads(calls[0]['data']).keys()
    assert attempts[0].generation_options['max_tokens'] == 64
    assert attempts[0].generation_options['temperature'] == .4
    assert attempts[0].generation_options['timeout'] == 12
    facade.llm.executor.shutdown(wait=False)


@pytest.mark.parametrize('controlled', [False, True])
async def test_revoked_profile_prevents_second_dispatch(monkeypatch, limits, controlled):
    calls, admissions = [], []
    limits = replace(limits, authorize_request=lambda: not calls)
    facade = client(limits, retries=2)
    facade.llm.retry_config.delay = 0
    monkeypatch.setattr('magic_llm.engine.engine_openai.PrivateAsyncHttpClient', lambda: Provider(calls, fail=True))

    class Control:
        async def before_attempt(self, attempt): admissions.append(attempt)
        async def after_attempt(self, attempt, outcome): pass

    try:
        with pytest.raises(RequestValidationError) as exc:
            await facade.llm.async_generate(ModelChat(system='hello'),
                provider_attempt_control=Control() if controlled else None)
        assert exc.value.code == 'EXECUTION_REQUEST_UNAVAILABLE'
        assert len(calls) == 1 and len(admissions) == int(controlled)
    finally:
        facade.llm.executor.shutdown(wait=False)
