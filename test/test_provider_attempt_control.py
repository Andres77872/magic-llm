"""Offline A33 gates: use real decorators/loops with in-memory providers/tools."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from magic_llm import MagicLLM
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.types import AgentBudget
from magic_llm.engine.attempt_control import (
    ProviderAttemptControlError, ProviderStreamInterrupted,
)
from magic_llm.engine.base_chat import BaseChat, RetryConfig
from magic_llm.exception.ChatException import RequestValidationError
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('usage', [None, UsageModel()])
async def test_completed_missing_usage_keeps_reservation_uncertain(stream, usage):
    async def successful_stream(index):
        yield chunk('done', finish='stop', usage=usage)

    async def successful_generate(index):
        return response().model_copy(update={'usage': usage})

    provider = FakeProvider(stream=successful_stream, generate=successful_generate)
    control = Control()
    if stream:
        await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control))
    else:
        await provider.async_generate(ModelChat(), provider_attempt_control=control)
    outcome = control.outcomes[0][1]
    assert outcome.status == 'completed'
    assert outcome.usage_uncertain


@pytest.mark.parametrize('stream', [False, True])
async def test_known_completed_usage_is_reconcilable(stream):
    usage = UsageModel(prompt_tokens=2, completion_tokens=1, total_tokens=3)

    async def successful_stream(index):
        yield chunk('done', finish='stop', usage=usage)

    async def successful_generate(index):
        return response().model_copy(update={'usage': usage})

    provider = FakeProvider(stream=successful_stream, generate=successful_generate)
    control = Control()
    if stream:
        await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control))
    else:
        await provider.async_generate(ModelChat(), provider_attempt_control=control)
    assert not control.outcomes[0][1].usage_uncertain


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('mutation', ['none', 'override', 'model_changed', 'endpoint_changed', 'invalid_endpoint'])
async def test_google_admission_matches_actual_bound_endpoint(monkeypatch, stream, mutation):
    from magic_llm.engine import engine_google

    provider = engine_google.EngineGoogle(api_key='fake-test-only', model='authorized-model', retries=1)
    dispatched = []
    prepared = []
    payload = {'candidates': [{'content': {'parts': [{'text': 'done'}]}, 'finishReason': 'STOP'}],
               'usageMetadata': {'promptTokenCount': 2, 'candidatesTokenCount': 1, 'totalTokenCount': 3}}

    async def prepare(chat, **kwargs):
        prepared.append(kwargs)
        return b'{}', {}, {}

    class MemoryHttp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post_json(self, url, **kwargs):
            dispatched.append(url)
            return payload

        async def post_stream(self, url, **kwargs):
            dispatched.append(url)
            yield 'data: ' + json.dumps(payload)

    monkeypatch.setattr(provider, 'prepare_data', prepare)
    monkeypatch.setattr(engine_google, 'AsyncHttpClient', MemoryHttp)
    options = {}
    endpoint = 'url_stream' if stream else 'url'
    if mutation == 'override':
        options['model'] = 'different-model'
    elif mutation == 'model_changed':
        provider.model = 'different-model'
    elif mutation == 'endpoint_changed':
        setattr(provider, endpoint, getattr(provider, endpoint).replace('authorized-model', 'different-model'))
    elif mutation == 'invalid_endpoint':
        setattr(provider, endpoint, 'https://example.invalid/unrecognized')
    control = Control()

    async def invoke():
        if stream:
            await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control, **options))
        else:
            await provider.async_generate(ModelChat(), provider_attempt_control=control, **options)

    if mutation == 'none':
        await invoke()
        assert control.admissions[0].model == 'authorized-model'
        assert dispatched == [getattr(provider, endpoint)]
    else:
        with pytest.raises(ProviderAttemptControlError) as caught:
            await invoke()
        assert caught.value.error_code == 'PROVIDER_ATTEMPT_MODEL_MISMATCH'
        assert not control.admissions
        assert not prepared
        assert not dispatched


def chunk(content='', *, calls=None, finish=None, **extra):
    return ChatCompletionModel(id='fake', model='fake', choices=[{
        'index': 0, 'delta': {'content': content, 'tool_calls': calls},
        'finish_reason': finish,
    }], **extra)


def call(index, identifier, message):
    import json
    return {'index': index, 'id': identifier, 'type': 'function',
            'function': {'name': 'send', 'arguments': json.dumps({'message': message})}}


def response(content='done'):
    return ModelChatResponse(id='fake', model='fake', object='chat.completion',
        created=0, choices=[{'index': 0, 'message': {'role': 'assistant', 'content': content},
                             'finish_reason': 'stop'}],
        usage=UsageModel(prompt_tokens=4, completion_tokens=2, total_tokens=6))


class Control:
    def __init__(self, deny=None, after_error=None):
        self.admissions = []
        self.outcomes = []
        self.deny = deny
        self.after_error = after_error

    async def before_attempt(self, attempt):
        self.admissions.append(attempt)
        if self.deny:
            self.deny(attempt)

    async def after_attempt(self, attempt, outcome):
        self.outcomes.append((attempt, outcome))
        if self.after_error:
            raise self.after_error


class FakeProvider:
    engine = 'openai'

    def __init__(self, stream=None, generate=None, *, model='primary', retries=2, fallback=None):
        self.model = model
        self.retry_config = RetryConfig(retries, delay=0)
        self.fallback = SimpleNamespace(llm=fallback) if fallback else None
        self.kwargs = {}
        self.stream_handler = stream
        self.generate_handler = generate
        self.calls = []
        self.callbacks = []

    async def _execute_callback(self, *args):
        self.callbacks.append(args)

    def _create_chat_meta_model(self, *args):
        return None

    def _update_metrics(self, *args):
        pass

    def _handle_fallback(self, **kwargs):
        return self.fallback.llm.async_stream_generate if self.fallback else None

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat, **kwargs):
        self.calls.append(kwargs)
        async for item in self.stream_handler(len(self.calls)):
            yield item

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat, **kwargs):
        self.calls.append(kwargs)
        return await self.generate_handler(len(self.calls))

    @BaseChat.sync_intercept_generate
    def generate(self, chat, **kwargs):
        self.calls.append(kwargs)
        return response()

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat, **kwargs):
        self.calls.append(kwargs)
        yield chunk('done', finish='stop')


async def collect(source):
    return [item async for item in source]


async def retry_tool_stream(index):
    if index == 1:
        yield chunk(calls=[call(0, 'failed-a', 'first'), call(1, 'failed-b', 'abandoned')])
        raise ConnectionError('lost stream')
    if index == 2:
        yield chunk(calls=[call(0, 'retry-a', 'retry')], finish='tool_calls')
    else:
        yield chunk('done', finish='stop')


@pytest.mark.parametrize('controlled', [True, False])
async def test_full_native_loop_never_executes_abandoned_call_when_controlled(controlled):
    executed = []

    async def send(message: str):
        executed.append(message)
        return 'accepted'

    provider = FakeProvider(stream=retry_tool_stream)
    control = Control() if controlled else None
    loop = AsyncAgentLoop(SimpleNamespace(llm=provider), tools=[send],
        builtin_todo_tools=False, budget=AgentBudget(max_iterations=4),
        provider_attempt_control=control)
    if controlled:
        with pytest.raises(ProviderStreamInterrupted):
            await collect(loop.stream('test'))
        assert executed == []
        assert len(provider.calls) == 1
        assert len(control.outcomes) == 1
        assert control.outcomes[0][1].status == 'failed'
        assert control.outcomes[0][1].semantic_output
    else:
        await collect(loop.stream('test'))
        # Explicit legacy control: this patch does not change disabled behavior.
        assert executed == ['abandoned']
        assert len(provider.calls) == 3


@pytest.mark.parametrize('semantic', [
    {'delta': {'content': 'partial'}},
    {'delta': {'reasoning_content': 'thinking'}},
    {'delta': {'reasoning': 'thinking'}},
    {'delta': {'reasoning_details': [{'text': 'thinking'}]}},
    {'delta': {'refusal': 'refused'}},
    {'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'send'}}]}},
    {'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{'}}]}},
    {'responses_output': [{'type': 'reasoning', 'id': 'r'}]},
    {'gemini_parts': [{'thoughtSignature': 'opaque'}]},
])
async def test_semantic_fragments_block_retry_and_fallback_for_schema_only_stream(semantic):
    async def failing(index):
        item = chunk()
        if 'delta' in semantic:
            item = ChatCompletionModel(id='fake', model='fake', choices=[{'delta': semantic['delta']}])
        else:
            for key, value in semantic.items():
                setattr(item, key, value)
        yield item
        raise ConnectionError('lost')

    fallback = FakeProvider(stream=failing, model='fallback')
    provider = FakeProvider(stream=failing, fallback=fallback)
    with pytest.raises(ProviderStreamInterrupted):
        await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=Control()))
    assert len(provider.calls) == 1
    assert fallback.calls == []


async def test_pre_delta_retry_is_admitted_with_distinct_id_and_stable_request():
    async def stream(index):
        if index == 1:
            raise ConnectionError('before output')
        yield chunk('done', finish='stop')

    provider = FakeProvider(stream=stream)
    control = Control()
    result = await collect(provider.async_stream_generate(ModelChat(),
        provider_attempt_control=control, max_tokens=20))
    assert result[-1].choices[0].delta.content == 'done'
    first, second = control.admissions
    assert first.provider_attempt_id != second.provider_attempt_id
    assert first.request_operation_id == second.request_operation_id
    assert second.parent_attempt_id == first.provider_attempt_id
    assert [item.retry_index for item in control.admissions] == [1, 2]
    assert [out.status for _, out in control.outcomes] == ['failed', 'completed']
    assert control.outcomes[0][1].usage_uncertain
    assert provider.calls == [{'max_tokens': 20}, {'max_tokens': 20}]


@pytest.mark.parametrize('code', ['CANCELLED', 'STALE_FENCE', 'DEADLINE', 'BUDGET_EXHAUSTED'])
@pytest.mark.parametrize('stream_mode', [False, True])
async def test_every_retry_requires_admission(code, stream_mode):
    async def fail(index):
        raise ConnectionError('failed before response')

    async def fail_stream(index):
        raise ConnectionError('failed before output')
        yield

    def deny_second(attempt):
        if attempt.attempt_index == 2:
            raise ProviderAttemptControlError('denied', error_code=code)

    control = Control(deny=deny_second)
    provider = FakeProvider(stream=fail_stream, generate=fail, retries=3)
    with pytest.raises(ProviderAttemptControlError) as caught:
        if stream_mode:
            await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control))
        else:
            await provider.async_generate(ModelChat(), provider_attempt_control=control)
    assert caught.value.error_code == code
    assert len(provider.calls) == 1
    assert len(control.admissions) == 2
    assert len(control.outcomes) == 1


@pytest.mark.parametrize('stream_mode', [False, True])
async def test_fallback_has_own_model_admission_and_same_operation(stream_mode):
    async def fail(index):
        raise ConnectionError('fail')

    async def fail_stream(index):
        raise ConnectionError('fail')
        yield

    def deny_fallback(attempt):
        if attempt.model == 'forbidden':
            raise PermissionError('model not authorized')

    fallback = FakeProvider(model='forbidden')
    provider = FakeProvider(stream=fail_stream, generate=fail, fallback=fallback, retries=1)
    control = Control(deny=deny_fallback)
    with pytest.raises(PermissionError):
        if stream_mode:
            await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control))
        else:
            await provider.async_generate(ModelChat(), provider_attempt_control=control)
    assert len(provider.calls) == 1
    assert fallback.calls == []
    first, second = control.admissions
    assert second.is_fallback and second.model == 'forbidden'
    assert second.request_operation_id == first.request_operation_id
    assert second.parent_attempt_id == first.provider_attempt_id


@pytest.mark.parametrize('stream_mode', [False, True])
async def test_successful_fallback_is_accounted_once(stream_mode):
    async def fail(index):
        raise ConnectionError('fail')

    async def fail_stream(index):
        raise ConnectionError('fail')
        yield

    async def succeed(index):
        return response()

    async def succeed_stream(index):
        yield chunk('done', finish='stop', usage=UsageModel(prompt_tokens=4, completion_tokens=2))

    fallback = FakeProvider(stream=succeed_stream, generate=succeed, model='allowed')
    provider = FakeProvider(stream=fail_stream, generate=fail, retries=1, fallback=fallback)
    control = Control()
    if stream_mode:
        await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control))
    else:
        await provider.async_generate(ModelChat(), provider_attempt_control=control)
    assert len(fallback.calls) == 1
    assert [out.status for _, out in control.outcomes] == ['failed', 'completed']
    assert control.outcomes[-1][1].usage.prompt_tokens == 4


@pytest.mark.parametrize('stream_mode', [False, True])
async def test_request_validation_denial_is_terminal(stream_mode):
    async def fail(index):
        raise RequestValidationError(ValueError('validation'))

    async def fail_stream(index):
        raise RequestValidationError(ValueError('validation'))
        yield

    fallback = FakeProvider()
    provider = FakeProvider(stream=fail_stream, generate=fail, fallback=fallback)
    control = Control()
    with pytest.raises(RequestValidationError):
        if stream_mode:
            await collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control))
        else:
            await provider.async_generate(ModelChat(), provider_attempt_control=control)
    assert len(provider.calls) == 1
    assert fallback.calls == []
    assert not control.outcomes[0][1].usage_uncertain


async def test_after_attempt_failure_never_retries_completed_request():
    async def succeed(index):
        return response()

    provider = FakeProvider(generate=succeed)
    with pytest.raises(RuntimeError, match='storage unavailable'):
        await provider.async_generate(ModelChat(), provider_attempt_control=Control(
            after_error=RuntimeError('storage unavailable')))
    assert len(provider.calls) == 1


async def test_cancellation_during_backoff_does_not_dispatch_next_attempt(monkeypatch):
    in_backoff = asyncio.Event()

    async def sleep(delay):
        in_backoff.set()
        await asyncio.Event().wait()

    async def fail(index):
        raise ConnectionError('before response')

    monkeypatch.setattr('magic_llm.engine.attempt_control.asyncio.sleep', sleep)
    provider = FakeProvider(generate=fail)
    control = Control()
    task = asyncio.create_task(provider.async_generate(ModelChat(), provider_attempt_control=control))
    await in_backoff.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(provider.calls) == len(control.admissions) == len(control.outcomes) == 1


@pytest.mark.parametrize('stream_mode', [False, True])
async def test_cancellation_of_active_attempt_reconciles_once(stream_mode):
    active = asyncio.Event()

    async def wait(index):
        active.set()
        await asyncio.Event().wait()

    async def wait_stream(index):
        await wait(index)
        yield

    provider = FakeProvider(stream=wait_stream, generate=wait)
    control = Control()
    call = (collect(provider.async_stream_generate(ModelChat(), provider_attempt_control=control))
            if stream_mode else provider.async_generate(ModelChat(), provider_attempt_control=control))
    task = asyncio.create_task(call)
    await active.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(provider.calls) == 1
    assert len(control.outcomes) == 1
    assert control.outcomes[0][1].status == 'cancelled'
    assert control.outcomes[0][1].usage_uncertain


async def test_consumer_close_reconciles_without_retry():
    async def stream(index):
        yield chunk('partial')
        await asyncio.Event().wait()

    provider = FakeProvider(stream=stream)
    control = Control()
    source = provider.async_stream_generate(ModelChat(), provider_attempt_control=control)
    await anext(source)
    await source.aclose()
    assert len(provider.calls) == 1
    assert [out.status for _, out in control.outcomes] == ['cancelled']


def test_sync_control_rejected_before_dispatch():
    provider = FakeProvider()
    for invoke in (lambda: provider.generate(ModelChat(), provider_attempt_control=Control()),
                   lambda: list(provider.stream_generate(ModelChat(), provider_attempt_control=Control())),
                   lambda: AgentLoop(SimpleNamespace(llm=provider), provider_attempt_control=Control())):
        with pytest.raises(ProviderAttemptControlError):
            invoke()
    assert provider.calls == []


async def test_unsupported_fallback_rejected_before_primary_dispatch():
    provider = FakeProvider(fallback=SimpleNamespace(async_generate=lambda *a, **kw: None))
    with pytest.raises(ProviderAttemptControlError):
        await provider.async_generate(ModelChat(), provider_attempt_control=Control())
    assert provider.calls == []


async def test_fallback_cycle_rejected_before_dispatch():
    provider = FakeProvider()
    provider.fallback = SimpleNamespace(llm=provider)
    with pytest.raises(ProviderAttemptControlError):
        await provider.async_generate(ModelChat(), provider_attempt_control=Control())
    assert provider.calls == []


async def test_private_request_snapshot_isolated_and_credentials_omitted():
    async def succeed(index):
        return response()

    provider = FakeProvider(generate=succeed)
    provider.kwargs = {'max_tokens': 9, 'api_key': 'hidden', 'headers': {'Authorization': 'hidden'}}
    chat = ModelChat()
    chat.add_user_message('private input')

    def inspect(attempt):
        assert attempt.generation_options == {'max_tokens': 9}
        assert 'private input' not in repr(attempt)
        attempt.messages[0]['content'] = 'mutated copy'
        attempt.generation_options['max_tokens'] = 999

    await provider.async_generate(chat, provider_attempt_control=Control(deny=inspect))
    assert chat.messages[0]['content'] == 'private input'
    assert provider.kwargs['max_tokens'] == 9
    assert provider.calls == [{}]


@pytest.mark.parametrize('stream_mode', [False, True])
async def test_magic_wrapper_explicit_control_plumbing(stream_mode):
    async def succeed(index):
        return response()

    async def succeed_stream(index):
        yield chunk('done', finish='stop')

    client = MagicLLM.__new__(MagicLLM)
    client.llm = FakeProvider(generate=succeed, stream=succeed_stream)
    client._task_executor = None
    control = Control()
    if stream_mode:
        await collect(client.run_agent_stream_async('hello', provider_attempt_control=control,
                                                  builtin_todo_tools=False))
    else:
        await client.run_agent_async('hello', provider_attempt_control=control, builtin_todo_tools=False)
    assert len(control.admissions) == 1
    assert 'provider_attempt_control' not in client.llm.calls[0]


async def test_native_loop_rejects_unsupported_engine_without_leaking_control():
    calls = []

    async def unsupported(chat, **kwargs):
        calls.append(kwargs)
        return response()

    loop = AsyncAgentLoop(SimpleNamespace(llm=SimpleNamespace(async_generate=unsupported)),
        provider_attempt_control=Control(), builtin_todo_tools=False)
    with pytest.raises(ProviderAttemptControlError):
        await loop.run('hello')
    assert calls == []
