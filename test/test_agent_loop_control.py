"""Canonical control/continuation gates with fake providers and in-memory state."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from magic_llm import MagicLLM
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.control import AgentControlError, AgentLoopCheckpoint, InboxMessage
from magic_llm.agent.types import AgentBudget, AgentBudgetExceeded
from magic_llm.engine.base_chat import BaseChat, RetryConfig
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


def answer(text='done', calls=None, **metadata):
    return ModelChatResponse(id='fake', object='chat.completion', created=0, model='fake',
        choices=[{'index': 0, 'message': {'role': 'assistant', 'content': text, 'tool_calls': calls},
                  'finish_reason': 'tool_calls' if calls else 'stop'}],
        usage=UsageModel(prompt_tokens=3, completion_tokens=2, total_tokens=5), **metadata)


def tool(identifier, name, **arguments):
    return {'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


class Provider:
    engine = 'openai'
    model = 'fake'

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.kwargs = {}
        self.retry_config = RetryConfig(1, 0)
        self.fallback = None

    async def _execute_callback(self, *args):
        pass

    def take(self, chat, kwargs):
        self.requests.append((copy.deepcopy(chat.messages), kwargs))
        return self.responses.pop(0)

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat, **kwargs):
        return self.take(chat, kwargs)

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat, **kwargs):
        response = self.take(chat, kwargs)
        calls = [dict(call.model_dump(), index=index) for index, call in enumerate(response.tool_calls or [])]
        yield ChatCompletionModel(id='fake', model='fake', usage=response.usage,
            responses_output=response.responses_output, gemini_parts=response.gemini_parts,
            choices=[{'index': 0, 'delta': {'content': response.content, 'tool_calls': calls or None},
                      'finish_reason': response.finish_reason}])


class Control:
    def __init__(self):
        self.pending = []
        self.saved = []
        self.candidates = []
        self.attempts = []
        self.outcomes = []

    async def before_turn(self, checkpoint):
        return list(self.pending)

    async def checkpoint(self, checkpoint, boundary):
        self.saved.append((boundary, checkpoint.detached()))
        self.pending = [item for item in self.pending if item.message_id not in checkpoint.consumed_message_ids]

    async def finish_candidate(self, checkpoint):
        self.candidates.append(checkpoint)
        return 'continue' if self.pending else 'candidate_ready'

    async def before_attempt(self, attempt):
        self.attempts.append(attempt)

    async def after_attempt(self, attempt, outcome):
        self.outcomes.append(outcome)


def loop_for(provider, control=None, **kwargs):
    control = control or Control()
    return AsyncAgentLoop(SimpleNamespace(llm=provider), control=control,
        provider_attempt_control=control, builtin_todo_tools=kwargs.pop('builtin_todo_tools', False), **kwargs)


async def execute(loop, stream, **kwargs):
    if stream:
        return [item async for item in loop.stream(**kwargs)]
    return await loop.run(**kwargs)


@pytest.mark.parametrize('stream', [False, True])
async def test_input_checkpoint_is_ack_authority_and_contains_safe_message(stream):
    control = Control()
    control.pending = [InboxMessage(message_id='m1', sender='peer', content='new facts')]
    provider = Provider([answer()])
    loop = loop_for(provider, control)
    await execute(loop, stream, user_input='task')
    first_boundary, first = control.saved[0]
    assert first_boundary == 'input'
    assert first.consumed_message_ids == ['m1']
    assert first.messages[-1]['role'] == 'user'
    assert 'new facts' in first.messages[-1]['content']
    assert provider.requests[0][0] == first.messages
    assert loop.checkpoint.step == 1
    assert [boundary for boundary, _ in control.saved] == ['input', 'candidate']
    assert all('control' not in kwargs and 'continuation' not in kwargs
               and 'provider_attempt_control' not in kwargs for _, kwargs in provider.requests)


@pytest.mark.parametrize('stream', [False, True])
async def test_message_arriving_during_tool_batch_waits_for_all_results(stream):
    active, release = asyncio.Event(), asyncio.Event()

    async def slow():
        active.set()
        await release.wait()
        return 'complete'

    control = Control()
    provider = Provider([answer(calls=[tool('t1', 'slow')]), answer('final')])
    loop = loop_for(provider, control, tools=[slow])
    task = asyncio.create_task(execute(loop, stream, user_input='task'))
    await active.wait()
    control.pending.append(InboxMessage(message_id='m2', content='arrived while tool active'))
    assert len(provider.requests) == 1
    release.set()
    await task
    post_tool = next(cp for boundary, cp in control.saved if boundary == 'tool_results')
    assert post_tool.messages[-1]['role'] == 'tool'
    assert post_tool.consumed_message_ids == []
    second_request = provider.requests[1][0]
    assert second_request[-2]['role'] == 'tool'
    assert second_request[-1]['role'] == 'user'
    assert 'arrived while tool active' in second_request[-1]['content']
    assert loop.checkpoint.step == 2


@pytest.mark.parametrize('stream', [False, True])
async def test_final_candidate_arrival_continues_and_preserves_prior_context(stream):
    class Arriving(Control):
        async def finish_candidate(self, checkpoint):
            if not self.candidates:
                self.pending.append(InboxMessage(message_id='late', content='correct the answer'))
            return await super().finish_candidate(checkpoint)

    control = Arriving()
    provider = Provider([answer('first candidate'), answer('updated final')])
    loop = loop_for(provider, control)
    result = await execute(loop, stream, user_input='task')
    assert loop.checkpoint.output_candidate == 'updated final'
    assert any(message.get('content') == 'first candidate' for message in provider.requests[1][0])
    assert loop.checkpoint.consumed_message_ids == ['late']
    assert loop.checkpoint.step == 2
    if not stream:
        assert result.content == 'updated final'


@pytest.mark.parametrize('where', ['before_turn', 'checkpoint', 'finish_candidate'])
async def test_control_failures_propagate_without_observer_suppression(where):
    control = Control()
    control.pending = [InboxMessage(message_id='m', content='input')]

    async def fail(*args):
        raise RuntimeError('authoritative failure')

    setattr(control, where, fail)
    provider = Provider([answer()])
    loop = loop_for(provider, control)
    with pytest.raises(RuntimeError, match='authoritative failure'):
        await loop.run('task')
    if where in ('before_turn', 'checkpoint'):
        assert provider.requests == []
        assert loop.checkpoint is None
        assert control.pending
    else:
        assert len(provider.requests) == 1


@pytest.mark.parametrize('stream', [False, True])
async def test_resume_preserves_dedup_history_usage_and_seen_ids(stream):
    effects = []

    async def paid(asset: str):
        effects.append(asset)
        return {'url': 'owned/' + asset}

    first_provider = Provider([answer(calls=[tool('t1', 'paid', asset='cover')]), answer('one')])
    first = loop_for(first_provider, tools=[paid], deduplicate=True)
    await execute(first, stream, user_input='task')
    checkpoint = AgentLoopCheckpoint.model_validate_json(first.checkpoint.model_dump_json())
    control = Control()
    control.pending = [InboxMessage(message_id='wake', content='resume')]
    second_provider = Provider([answer(calls=[tool('t2', 'paid', asset='cover')]), answer('two')])
    second = loop_for(second_provider, control, tools=[paid], deduplicate=True)
    await execute(second, stream, continuation=checkpoint)
    assert effects == ['cover']
    assert second.checkpoint.step == 4
    assert second.checkpoint.total_input_tokens == 12
    assert second.checkpoint.total_output_tokens == 8
    assert second.checkpoint.seen_tool_call_ids == ['t1', 't2']
    assert second.checkpoint.started_at == first.checkpoint.started_at
    assert second.checkpoint.messages[:len(checkpoint.messages)] == checkpoint.messages


async def test_restore_rejects_reused_tool_id_before_effect():
    effects = []

    async def effect():
        effects.append('called')

    first = loop_for(Provider([answer(calls=[tool('same', 'effect')]), answer()]), tools=[effect])
    await first.run('task')
    second = loop_for(Provider([answer(calls=[tool('same', 'effect')])]), tools=[effect])
    with pytest.raises(ValueError, match='Duplicate tool call ID'):
        await second.run(continuation=first.checkpoint)
    assert effects == ['called']


async def test_resume_restores_builtin_todo_state():
    todos = [{'id': 1, 'content': 'keep me', 'status': 'pending', 'priority': 'high'}]
    first = loop_for(Provider([answer(calls=[tool('write', 'todowrite', todos=todos)]), answer()]),
                     builtin_todo_tools=True)
    await first.run('task')
    provider = Provider([answer(calls=[tool('read', 'todoread')]), answer()])
    second = loop_for(provider, builtin_todo_tools=True)
    await second.run(continuation=first.checkpoint)
    result = next(message for message in provider.requests[1][0] if message.get('tool_call_id') == 'read')
    assert json.loads(result['content'])['todos'] == todos


async def test_restore_cannot_expand_cumulative_turn_budget():
    first = loop_for(Provider([answer()]), budget=AgentBudget(max_iterations=1))
    await first.run('task')
    provider = Provider([answer()])
    second = loop_for(provider, budget=AgentBudget(max_iterations=99))
    with pytest.raises(AgentBudgetExceeded) as caught:
        await second.run(continuation=first.checkpoint)
    assert caught.value.budget_type == 'max_iterations'
    assert provider.requests == []


async def test_absolute_utc_deadline_survives_restore_and_refuses_expired_work():
    now = [1000.0]
    first = loop_for(Provider([answer()]), budget=AgentBudget(wall_clock_timeout=5),
                     control_clock=lambda: now[0])
    await first.run('task')
    checkpoint = first.checkpoint
    assert checkpoint.absolute_deadline == '1970-01-01T00:16:45Z'
    assert 'monotonic' not in checkpoint.model_dump_json()
    now[0] = 1006.0
    provider = Provider([answer()])
    second = loop_for(provider, budget=AgentBudget(wall_clock_timeout=500), control_clock=lambda: now[0])
    with pytest.raises(AgentBudgetExceeded):
        await second.run(continuation=checkpoint)
    assert provider.requests == []


@pytest.mark.parametrize('mutation', ['version', 'version_bool', 'version_float', 'version_missing',
                                    'provider', 'manifest', 'unfinished', 'lost_id', 'unicode', 'digest'])
async def test_incompatible_checkpoint_rejected_before_dispatch(mutation):
    async def noop():
        return 'ok'

    first = loop_for(Provider([answer(calls=[tool('id', 'noop')]), answer()]), tools=[noop])
    await first.run('task')
    data = first.checkpoint.model_dump()
    if mutation == 'version':
        data['schema_version'] = 99
    elif mutation == 'version_bool':
        data['schema_version'] = True
    elif mutation == 'version_float':
        data['schema_version'] = 1.0
    elif mutation == 'version_missing':
        del data['schema_version']
    elif mutation == 'provider':
        data['provider'] = 'foreign'
    elif mutation == 'manifest':
        data['tool_manifest_digest'] = '0' * 64
    elif mutation == 'unfinished':
        data['messages'] = data['messages'][:-2]
    elif mutation == 'unicode':
        data['messages'][0]['content'] = '\ud800'
    elif mutation == 'digest':
        data['consumed_message_ids'] = ['m']
        data['message_digests'] = {'m': 'not-a-hash'}
    else:
        data['seen_tool_call_ids'] = []
    provider = Provider([answer()])
    second = loop_for(provider, tools=[noop])
    with pytest.raises(AgentControlError):
        await second.run(continuation=data)
    assert provider.requests == []


async def test_duplicate_delivery_is_not_reinjected_and_conflicts_fail():
    control = Control()
    message = InboxMessage(message_id='once', content='stable')
    control.pending = [message]
    first = loop_for(Provider([answer()]), control)
    await first.run('task')
    next_control = Control()
    next_control.pending = [message]
    second = loop_for(Provider([answer()]), next_control)
    await second.run(continuation=first.checkpoint)
    assert sum('stable' in str(item.get('content', '')) for item in second.checkpoint.messages) == 1
    conflict_control = Control()
    conflict_control.pending = [message.model_copy(update={'content': 'changed'})]
    provider = Provider([answer()])
    third = loop_for(provider, conflict_control)
    with pytest.raises(AgentControlError, match='conflicting'):
        await third.run(continuation=first.checkpoint)
    assert provider.requests == []


async def test_private_canonical_context_and_provider_replay_metadata_survive():
    def guard(context):
        context.chat.require_complete_context()
        context.chat.set_observer_projection(lambda chat: ModelChat())

    history = ModelChat()
    history.messages = [
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'previous', 'responses_output': [{'type': 'reasoning', 'id': 'r'}],
         'gemini_parts': [{'thoughtSignature': 'opaque'}]},
        {'role': 'user', 'content': 'private loaded instruction'},
    ]
    first = loop_for(Provider([answer()]), request_guard=guard)
    await first.run(initial_chat=history)
    assert first.checkpoint.messages[1] == history.messages[1]
    assert 'private loaded instruction' in first.checkpoint.model_dump_json()
    provider = Provider([answer()])
    unguarded = loop_for(provider)
    with pytest.raises(AgentControlError, match='reconstructed context guards'):
        await unguarded.run(continuation=first.checkpoint)
    assert provider.requests == []
    second = loop_for(provider, request_guard=guard)
    await second.run(continuation=first.checkpoint)
    assert provider.requests[0][0][:len(first.checkpoint.messages)] == first.checkpoint.messages


async def test_snapshot_is_detached_and_normal_run_is_fresh():
    provider = Provider([answer('one'), answer('two')])
    loop = loop_for(provider)
    await loop.run('first')
    snapshot = loop.checkpoint
    snapshot.messages.clear()
    assert loop.checkpoint.messages
    await loop.run('new task')
    assert loop.checkpoint.step == 1
    assert len(provider.requests[1][0]) == 1
    assert provider.requests[1][0][0]['content'] == 'new task'


@pytest.mark.parametrize('stream', [False, True])
async def test_magic_wrappers_forward_control_and_explicit_continuation(stream):
    client = MagicLLM.__new__(MagicLLM)
    client.llm = Provider([answer('one'), answer('two')])
    client._task_executor = None
    first = Control()
    method = client.run_agent_stream_async if stream else client.run_agent_async
    if stream:
        [item async for item in method('first', control=first, provider_attempt_control=first, builtin_todo_tools=False)]
    else:
        await method('first', control=first, provider_attempt_control=first, builtin_todo_tools=False)
    second = Control()
    kwargs = {'control': second, 'provider_attempt_control': second,
              'continuation': first.saved[-1][1], 'builtin_todo_tools': False}
    if stream:
        [item async for item in method('', **kwargs)]
    else:
        await method('', **kwargs)
    assert second.saved[-1][1].step == 2
    assert all('continuation' not in opts and 'control' not in opts for _, opts in client.llm.requests)


def test_loop_control_requires_attempt_admission_and_rejects_sync():
    with pytest.raises(AgentControlError):
        AsyncAgentLoop(SimpleNamespace(llm=Provider([])), control=Control())
    with pytest.raises(AgentControlError):
        AgentLoop(SimpleNamespace(llm=Provider([])), control=Control())


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('provider_name', ['openai', 'anthropic', 'google'])
async def test_native_result_batches_restore_complete_with_provider_metadata(stream, provider_name):
    async def noop():
        return 'complete'

    call = tool('native', 'noop')
    call['provider_metadata'] = {'gemini_part': {'thoughtSignature': 'signed'}}
    provider = Provider([answer(calls=[call], responses_output=[{'type': 'reasoning', 'id': 'r'}],
                                 gemini_parts=[{'thoughtSignature': 'signed'}]), answer()])
    provider.engine = provider_name
    first = loop_for(provider, tools=[noop])
    await execute(first, stream, user_input='task')
    checkpoint = first.checkpoint
    saved_call = next(message for message in checkpoint.messages if message.get('tool_calls'))
    assert saved_call['responses_output'] == [{'type': 'reasoning', 'id': 'r'}]
    assert saved_call['gemini_parts'] == [{'thoughtSignature': 'signed'}]
    # Streaming test fixture supplies replay metadata on the enclosing chunk.
    if not stream:
        assert saved_call['tool_calls'][0]['provider_metadata'] == call['provider_metadata']
    next_provider = Provider([answer()])
    next_provider.engine = provider_name
    second = loop_for(next_provider, tools=[noop])
    await execute(second, stream, continuation=checkpoint)
    assert next_provider.requests[0][0][:len(checkpoint.messages)] == checkpoint.messages


@pytest.mark.parametrize('stream', [False, True])
async def test_empty_candidate_preserves_replay_carrier(stream):
    provider = Provider([answer('', responses_output=[{'type': 'reasoning', 'id': 'opaque'}],
                               gemini_parts=[{'thoughtSignature': 'signed'}])])
    loop = loop_for(provider)
    await execute(loop, stream, user_input='task')
    assert loop.checkpoint.messages[-1] == {
        'role': 'assistant', 'content': '',
        'responses_output': [{'type': 'reasoning', 'id': 'opaque'}],
        'gemini_parts': [{'thoughtSignature': 'signed'}],
    }


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_tool_checkpoint_stops_before_next_provider_turn(stream):
    class FailingCheckpoint(Control):
        async def checkpoint(self, checkpoint, boundary):
            if boundary == 'tool_results':
                raise RuntimeError('persistence unavailable')
            await super().checkpoint(checkpoint, boundary)

    async def noop():
        return 'complete'

    provider = Provider([answer(calls=[tool('t', 'noop')]), answer('must not run')])
    loop = loop_for(provider, FailingCheckpoint(), tools=[noop])
    with pytest.raises(RuntimeError, match='persistence unavailable'):
        await execute(loop, stream, user_input='task')
    assert len(provider.requests) == 1
    assert loop.checkpoint.step == 0
    assert not any(message.get('tool_calls') for message in loop.checkpoint.messages)


@pytest.mark.parametrize('stream', [False, True])
async def test_cancelled_input_checkpoint_never_dispatches(stream):
    active = asyncio.Event()

    class PendingCheckpoint(Control):
        async def checkpoint(self, checkpoint, boundary):
            active.set()
            await asyncio.Event().wait()

    provider = Provider([answer('must not run')])
    control = PendingCheckpoint()
    control.pending = [InboxMessage(message_id='pending', content='pending ACK')]
    loop = loop_for(provider, control)
    task = asyncio.create_task(execute(loop, stream, user_input='task'))
    await active.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.requests == []
    assert loop.checkpoint is None
    assert control.pending
