"""Offline wire tests for generic complete-output and request-validation seams."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from magic_llm import MagicLLM
from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.request import AgentRequestContext
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall
from magic_llm.engine.engine_anthropic import EngineAnthropic
from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.engine.openai_adapters import ProviderOpenAI, ProviderDeepInfra, ProviderSambaNova
from magic_llm.engine.openai_adapters.responses import OpenAIResponsesAdapter
from magic_llm.engine.tooling import AnthropicStreamState, StreamIterationSummary, accumulate_stream_chunk, stream_summary_tool_calls
from magic_llm.exception.ChatException import ChatException, RequestValidationError
from magic_llm.model import ModelChat
from magic_llm.util.response_mapping import build_response, build_stream_chunk, build_tool_call, build_stream_tool_call

SCHEMA = {'type': 'function', 'function': {'name': 'read_definitions',
    'description': 'Read selected definitions.', 'parameters': {'type': 'object',
    'properties': {'ids': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['ids']}}}
FULL = {'schema_version': 1, 'skills': [{'id': 'one', 'prompt': 'PRIVATE_PROMPT_SENTINEL',
    'props': {'language': 'español', 'nested': {'value': 'PRIVATE_PROPS_SENTINEL'}}}]}
CALL = {'ids': ['one']}


def initial_chat():
    chat = ModelChat(system='Primary policy. Available: one | One | Use for supplied changes.')
    chat.add_user_message('Summarize the changes')
    return chat


class WireEngine:
    """Uses real adapters to prepare both requests; replaces only transport."""
    def __init__(self, provider):
        self.engine = provider
        self.model = 'fake-model'
        self.kwargs = {}
        self.requests = []
        if provider == 'anthropic':
            self.adapter = EngineAnthropic(api_key='test', model=self.model)
        elif provider == 'google':
            self.adapter = EngineGoogle(api_key='test', model=self.model)
        else:
            self.adapter = ProviderOpenAI(api_key='test', model=self.model)

    def record(self, chat, kwargs):
        assert 'request_guard' not in kwargs and 'tool_result_observer' not in kwargs
        if self.engine == 'google':
            body = self.adapter.prepare_data_sync(chat, **kwargs)[0]
        else:
            body = self.adapter.prepare_data(chat, **kwargs)[0]
        self.requests.append(json.loads(body))

    def response(self):
        if len(self.requests) == 1:
            if self.engine == 'google':
                return self.adapter.process_generate({'candidates': [{'content': {'parts': [{
                    'functionCall': {'id': 'call-load', 'name': 'read_definitions', 'args': CALL},
                    'thoughtSignature': 'opaque-signature'}]}, 'finishReason': 'STOP'}],
                    'usageMetadata': {}})
            return build_response(id='first', model=self.model, content=None,
                finish_reason='tool_calls', tool_calls=[build_tool_call(id='call-load',
                name='read_definitions', arguments=json.dumps(CALL))])
        return build_response(id='last', model=self.model, content='Applied the loaded guidance.', finish_reason='stop')

    def generate(self, chat, **kwargs):
        self.record(chat, kwargs)
        return self.response()

    async def async_generate(self, chat, **kwargs):
        self.record(chat, kwargs)
        return self.response()

    def stream_generate(self, chat, **kwargs):
        self.record(chat, kwargs)
        if len(self.requests) == 1:
            if self.engine == 'google':
                yield self.adapter.prepare_stream_response('data: '+json.dumps({'candidates': [{
                    'content': {'parts': [{'functionCall': {'id': 'call-load', 'name': 'read_definitions',
                    'args': CALL}, 'thoughtSignature': 'opaque-signature'}]}, 'finishReason': 'STOP'}],
                    'usageMetadata': {}}))
            else:
                yield build_stream_chunk(id='first', model=self.model, content='', finish_reason='tool_calls',
                    tool_calls=[build_stream_tool_call(id='call-load', name='read_definitions', arguments=json.dumps(CALL))])
        else:
            yield build_stream_chunk(id='last', model=self.model, content='Applied the loaded guidance.', finish_reason='stop')

    async def async_stream_generate(self, chat, **kwargs):
        for item in self.stream_generate(chat, **kwargs):
            yield item


def invoke(loop, mode, chat):
    if mode == 'sync':
        return loop.run(None, initial_chat=chat)
    if mode == 'sync-stream':
        return list(loop.stream(None, initial_chat=chat))
    if mode == 'async':
        return asyncio.run(loop.run(initial_chat=chat))
    async def collect():
        return [chunk async for chunk in loop.stream(initial_chat=chat)]
    return asyncio.run(collect())


@pytest.mark.parametrize('provider', ['openai', 'anthropic', 'google'])
@pytest.mark.parametrize('mode', ['sync', 'sync-stream', 'async', 'async-stream'])
def test_metadata_first_complete_second_wire_and_isolated_observer(provider, mode):
    engine = type('OfflineWireEngine', (WireEngine,), {'engine': provider})(provider)
    chat = initial_chat()
    original = copy.deepcopy(chat.messages)
    seen_contexts = []
    observed = []
    wire_validations = []

    def guard(context):
        assert isinstance(context, AgentRequestContext)
        seen_contexts.append(copy.deepcopy(context.messages))
        assert context.estimated_input_tokens() > 0
        context.chat.set_provider_payload_guard(lambda payload: wire_validations.append(copy.deepcopy(payload)))

    def read_definitions(ids):
        assert ids == ['one']
        return copy.deepcopy(FULL)
    read_definitions._require_complete_output = True

    def observer(result):
        result.content = '{"summary":"one loaded"}'
        return result
    hooks = SimpleNamespace(on_tool_complete=lambda result, state, **kwargs: observed.append(result))
    cls = AsyncAgentLoop if mode.startswith('async') else AgentLoop
    loop = cls(SimpleNamespace(llm=engine), tools=[SCHEMA], tool_functions={'read_definitions': read_definitions},
        request_guard=guard, tool_result_observer=observer, hooks=hooks, builtin_todo_tools=False)
    invoke(loop, mode, chat)
    assert len(engine.requests) == len(seen_contexts) == 2
    assert chat.messages == original
    assert 'PRIVATE_' not in json.dumps(engine.requests[0])
    assert 'PRIVATE_PROMPT_SENTINEL' in json.dumps(engine.requests[1])
    assert 'PRIVATE_PROPS_SENTINEL' in json.dumps(engine.requests[1])
    assert observed[0].content == '{"summary":"one loaded"}'
    assert all('PRIVATE_' not in item.content for item in observed)
    assert wire_validations[-1] == engine.requests[-1]
    second = engine.requests[1]
    if provider == 'anthropic':
        assert second['system'].startswith('Primary policy.')
        use = second['messages'][-2]['content'][0]
        result = second['messages'][-1]['content'][0]
        assert use == {'type': 'tool_use', 'id': 'call-load', 'name': 'read_definitions', 'input': CALL}
        assert result['tool_use_id'] == 'call-load'
        assert json.loads(result['content']) == FULL
    elif provider == 'google':
        call = second['contents'][-2]['parts'][0]
        assert call['thoughtSignature'] == 'opaque-signature'
        assert call['functionCall'] == {'id': 'call-load', 'name': 'read_definitions', 'args': CALL}
        result = second['contents'][-1]['parts'][0]['functionResponse']
        assert result['id'] == 'call-load' and json.loads(result['response']['output']) == FULL
    else:
        assert second['messages'][-2]['tool_calls'][0]['id'] == 'call-load'
        assert json.loads(second['messages'][-1]['content']) == FULL
    # The load remains only in tool history, with no second system injection.
    assert 'PRIVATE_' not in str(second.get('system', second.get('systemInstruction', second.get('messages', [])[0] if provider == 'openai' else '')))


@pytest.mark.parametrize('mode', ['sync', 'sync-stream', 'async', 'async-stream'])
def test_terminal_guard_prevents_second_provider_request(mode):
    engine = type('OfflineGoogleEngine', (WireEngine,), {'engine': 'google'})('google')
    def guard(context):
        if 'PRIVATE_PROMPT_SENTINEL' in json.dumps(context.messages):
            raise RuntimeError('COMPLETE_CONTEXT_LIMIT')
    cls = AsyncAgentLoop if mode.startswith('async') else AgentLoop
    loop = cls(SimpleNamespace(llm=engine), tools=[SCHEMA], tool_functions={'read_definitions': lambda ids: FULL},
        request_guard=guard, builtin_todo_tools=False)
    with pytest.raises(RuntimeError, match='COMPLETE_CONTEXT_LIMIT'):
        invoke(loop, mode, initial_chat())
    assert len(engine.requests) == 1


@pytest.mark.parametrize('adapter', [ProviderOpenAI, ProviderDeepInfra, ProviderSambaNova, EngineAnthropic, EngineGoogle])
def test_actual_final_payload_limit_is_terminal_before_transport(adapter):
    chat = initial_chat()
    chat.require_complete_context()
    seen = []
    def final_guard(payload):
        seen.append(copy.deepcopy(payload))
        if len(json.dumps(payload).encode('utf-8')) > 1:
            raise RuntimeError('FINAL_WIRE_CONTEXT_LIMIT')
    chat.set_provider_payload_guard(final_guard)
    engine = adapter(api_key='test', model='fake')
    with pytest.raises(RequestValidationError) as failure:
        if adapter is EngineGoogle:
            engine.prepare_data_sync(chat, tools=[SCHEMA])
        else:
            engine.prepare_data(chat, tools=[SCHEMA], stream=True)
    assert seen
    assert isinstance(failure.value.validation_error, RuntimeError)


def test_final_responses_payload_guard_sees_mapped_replay_and_schema():
    chat = initial_chat()
    chat.add_tool_call_message([{'id': 'call1', 'type': 'function', 'function': {
        'name': 'read_definitions', 'arguments': json.dumps(CALL)}}])
    chat.add_tool_result('call1', json.dumps(FULL))
    seen = []
    chat.set_provider_payload_guard(lambda payload: seen.append(copy.deepcopy(payload)))
    provider = ProviderOpenAI(api_key='test', model='fake')
    raw, _ = OpenAIResponsesAdapter(provider).transform_request(chat, tools=[SCHEMA])
    payload = json.loads(raw)
    assert seen[-1] == payload
    assert 'messages' not in payload
    assert any(item.get('type') == 'function_call_output' for item in payload['input'])
    assert payload['tools'][0]['name'] == 'read_definitions'


@pytest.mark.parametrize('adapter', [ProviderOpenAI, ProviderDeepInfra, EngineAnthropic, EngineGoogle])
def test_protected_provider_debug_never_logs_loaded_body(adapter, monkeypatch, caplog):
    monkeypatch.setenv('MAGIC_LLM_DEBUG_PAYLOAD', '1')
    monkeypatch.setenv('MAGIC_LLM_DEBUG_PAYLOAD_FULL', '1')
    chat = initial_chat()
    chat.require_complete_context()
    chat.add_user_message('PRIVATE_PROMPT_SENTINEL')
    engine = adapter(api_key='test', model='fake')
    with caplog.at_level('INFO'):
        if adapter is EngineGoogle:
            engine.prepare_data_sync(chat)
        else:
            engine.prepare_data(chat)
    assert 'MAGIC_LLM_DEBUG_PAYLOAD_REDACTED' in caplog.text
    assert 'PRIVATE_PROMPT_SENTINEL' not in caplog.text


def test_public_tool_limits_exact_unicode_output_and_atomic_exception_results():
    output = {'prompt': 'é'*30}
    serialized = ToolExecutor.serialize_output(output)
    assert len(serialized) > len(json.dumps(output, ensure_ascii=False))
    executor = ToolExecutor(max_content_size=len(serialized), max_content_sizes={'other': 90})
    executor.require_complete_output('load')
    executor.register('load', lambda: output)
    call = CanonicalToolCall('a', 'load', {})
    result = executor.execute(call)
    assert not result.is_error and result.content == serialized
    executor = executor.with_options({'max_content_size': len(serialized)-1})
    result = executor.execute(call)
    assert result.is_error and result.error_type == 'ToolOutputLimitError'
    assert 'é' not in result.content and 'TRUNCATED' not in result.content
    assert json.loads(result.content)['type'] == 'ToolOutputLimitError'
    executor.register('load', lambda: (_ for _ in ()).throw(ValueError('SECRET'*100)))
    result = executor.execute(call)
    assert result.is_error and 'SECRET' not in result.content
    client = MagicLLM.__new__(MagicLLM)
    client._task_executor = executor
    assert client.tool_content_limit('other', tool_executor_options={'max_content_size': 999}) == 90
    assert client.tool_content_limit('load', tool_executor_options={'max_content_size': 999}) == 999


def test_guarded_chat_counts_call_arguments_and_native_results_and_never_trims_pairs():
    chat = ModelChat(system='policy', max_input_tokens=40)
    chat.add_user_message('task')
    chat.add_tool_call_message([{'id': 'one', 'function': {'name': 'load', 'arguments': 'x '*500}}])
    chat.add_tool_result('one', json.dumps(FULL))
    messages = copy.deepcopy(chat.messages)
    count = chat.num_tokens_from_messages()
    assert count > 500
    chat.require_complete_context()
    with pytest.raises(ChatException) as exc:
        chat.get_messages()
    assert exc.value.error_code == 'COMPLETE_CONTEXT_EXCEEDS_TOKEN_LIMIT'
    assert chat.messages == messages
    native = ModelChat()
    native.messages = [{'role': 'user', 'content': [{'type': 'tool_result', 'content': 'x '*1000, 'tool_use_id': 'one'}]}]
    assert native.num_tokens_from_messages() > 1000


def test_anthropic_multifragment_arguments_are_not_duplicated_and_empty_calls_survive():
    engine = EngineAnthropic(api_key='test', model='fake')
    state = AnthropicStreamState()
    summary = StreamIterationSummary()
    events = [
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'tool_use', 'id': 'one', 'name': 'load', 'input': {}}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': '{"ids":'}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': '["one"]}'}},
        {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'tool_use', 'id': 'two', 'name': 'status', 'input': {}}},
    ]
    for event in events:
        chunk, _, _ = engine.prepare_chunk(event, 'response', None, state)
        if chunk is not None:
            accumulate_stream_chunk(summary, chunk)
    calls = stream_summary_tool_calls(summary)
    assert [(c.id, c.name, c.arguments) for c in calls] == [('one', 'load', {'ids': ['one']}), ('two', 'status', {})]


def test_anthropic_preparation_preserves_all_systems_nested_parts_and_none_choice():
    engine = EngineAnthropic(api_key='test', model='fake')
    chat = ModelChat(system='first')
    chat.add_user_message('hello')
    chat.add_system_message('second')
    chat.messages.append({'role': 'user', 'content': [{'type': 'text', 'text': 'original', 'image_url': None}]})
    original = copy.deepcopy(chat.messages)
    first = json.loads(engine.prepare_data(chat, tools=[SCHEMA], tool_choice='none')[0])
    second = json.loads(engine.prepare_data(chat, tools=[SCHEMA], tool_choice='none')[0])
    assert first == second and chat.messages == original
    assert first['system'] == 'first\n\nsecond'
    assert first['tool_choice'] == {'type': 'none'}


def test_google_async_final_payload_preserves_native_signature_and_canonical_result():
    engine = EngineGoogle(api_key='test', model='fake')
    chat = initial_chat()
    chat.add_tool_call_message([{'id': 'one', 'function': {'name': 'read_definitions', 'arguments': json.dumps(CALL)},
        'provider_metadata': {'gemini_part': {'functionCall': {'id': 'one', 'name': 'read_definitions', 'args': CALL},
        'thoughtSignature': 'opaque-signature'}}}])
    chat.add_tool_result('one', json.dumps(FULL))
    original = copy.deepcopy(chat.messages)
    seen = []
    chat.set_provider_payload_guard(lambda payload: seen.append(copy.deepcopy(payload)))
    raw, _, payload = asyncio.run(engine.prepare_data(chat, tools=[SCHEMA]))
    assert seen[-1] == json.loads(raw) == payload
    assert payload['contents'][-2]['parts'][0]['thoughtSignature'] == 'opaque-signature'
    assert payload['contents'][-1]['parts'][0]['functionResponse']['name'] == 'read_definitions'
    assert chat.messages == original


@pytest.mark.parametrize('mode', ['sync', 'sync-stream', 'async', 'async-stream'])
def test_final_payload_validation_bypasses_basechat_retry_callback_and_transport(mode):
    from magic_llm.engine.base_chat import BaseChat
    class HostContextError(RuntimeError):
        code = 'SKILLS_CONTEXT_LIMIT'
        outcome = SimpleNamespace(code='SKILLS_CONTEXT_LIMIT', retryable=False)
    class GuardedEngine(BaseChat):
        engine = 'openai'
        def __init__(self):
            super().__init__(model='fake', retries=3)
            self.preparations = 0
            self.transports = 0
        def prepare(self, chat):
            self.preparations += 1
            chat.validate_provider_payload({'messages': chat.get_messages(), 'tools': [{'overhead': 'x'*200}]})
            self.transports += 1
        @BaseChat.sync_intercept_generate
        def generate(self, chat, **kwargs):
            self.prepare(chat)
            return build_response(id='one', model='fake', content='done')
        @BaseChat.async_intercept_generate
        async def async_generate(self, chat, **kwargs):
            self.prepare(chat)
            return build_response(id='one', model='fake', content='done')
        @BaseChat.sync_intercept_stream_generate
        def stream_generate(self, chat, **kwargs):
            self.prepare(chat)
            yield build_stream_chunk(id='one', model='fake', content='done')
        @BaseChat.async_intercept_stream_generate
        async def async_stream_generate(self, chat, **kwargs):
            self.prepare(chat)
            yield build_stream_chunk(id='one', model='fake', content='done')
    engine = GuardedEngine()
    callback_calls=[]
    engine.callback=lambda *args: callback_calls.append(args)
    def guard(context):
        def final(payload):
            raise HostContextError('private body must not be logged')
        context.chat.set_provider_payload_guard(final)
    cls=AsyncAgentLoop if mode.startswith('async') else AgentLoop
    loop=cls(SimpleNamespace(llm=engine), request_guard=guard, builtin_todo_tools=False)
    with pytest.raises(RequestValidationError) as failure:
        invoke(loop,mode,initial_chat())
    assert failure.value.error_code == failure.value.code == 'SKILLS_CONTEXT_LIMIT'
    assert isinstance(failure.value.validation_error, HostContextError)
    assert failure.value.outcome.code == 'SKILLS_CONTEXT_LIMIT'
    assert failure.value.outcome is not HostContextError.outcome
    assert 'private body' not in str(failure.value)
    assert engine.preparations == 1 and engine.transports == 0 and callback_calls == []


def test_public_registered_names_include_tasks_without_mutable_registry_access():
    from magic_llm.agent.task_executor import TaskExecutor
    from magic_llm.agent.types import TaskManifest
    client=MagicLLM.__new__(MagicLLM)
    client._task_executor=TaskExecutor()
    client._task_executor.register_task(TaskManifest(id='read_definitions', name='Existing task',
        description='A task', input_schema={'type':'object'}), lambda: {})
    assert client.registered_tool_names() == frozenset({'read_definitions'})
    names=client._task_executor.registered_names()
    assert isinstance(names,frozenset)
    assert client._task_executor.fork().registered_names() == names


@pytest.mark.parametrize('mode', ['sync', 'async'])
def test_google_preserves_multiple_system_messages_in_order_without_mutation(mode):
    engine = EngineGoogle(api_key='test', model='fake')
    chat = ModelChat(system='primary policy')
    chat.add_user_message('hello')
    chat.messages.append({'role': 'system', 'content': [{'type':'text','text':'catalog guidance'}]})
    original = copy.deepcopy(chat.messages)
    if mode == 'async':
        raw, _, payload = asyncio.run(engine.prepare_data(chat, tools=[SCHEMA]))
    else:
        raw, _, payload = engine.prepare_data_sync(chat, tools=[SCHEMA])
    assert payload['systemInstruction'] == {'parts':[{'text':'primary policy\n\ncatalog guidance'}]}
    assert len(payload['contents']) == 1 and payload['contents'][0]['role'] == 'user'
    assert json.loads(raw) == payload and chat.messages == original


@pytest.mark.parametrize('mode', ['sync', 'sync-stream', 'async', 'async-stream'])
def test_google_signed_text_and_parallel_calls_replay_once_in_original_order(mode):
    native = [
        {'text':'Opaque reasoning preface.', 'thought':True, 'thoughtSignature':'text-signature'},
        {'functionCall':{'id':'first','name':'read_definitions','args':CALL}, 'thoughtSignature':'call-signature'},
        {'functionCall':{'id':'second','name':'read_definitions','args':CALL}},
        {'text':'','thoughtSignature':'tail-signature'},
    ]
    class SignedWireEngine(WireEngine):
        engine='google'
        def response(self):
            if len(self.requests) == 1:
                return self.adapter.process_generate({'candidates':[{'content':{'parts':copy.deepcopy(native)},
                    'finishReason':'STOP'}], 'usageMetadata':{}})
            return super().response()
        def stream_generate(self, chat, **kwargs):
            self.record(chat, kwargs)
            if len(self.requests) == 1:
                # Each functionCall is a separate chunk, still a distinct canonical call.
                for part in native:
                    yield self.adapter.prepare_stream_response('data: '+json.dumps({'candidates':[
                        {'content':{'parts':[copy.deepcopy(part)]},'finishReason':'STOP'}], 'usageMetadata':{}}))
            else:
                yield build_stream_chunk(id='last', model=self.model, content='done', finish_reason='stop')
    engine=SignedWireEngine('google')
    cls=AsyncAgentLoop if mode.startswith('async') else AgentLoop
    loop=cls(SimpleNamespace(llm=engine),tools=[SCHEMA],
        tool_functions={'read_definitions':lambda ids:copy.deepcopy(FULL)},
        request_guard=lambda context:None,builtin_todo_tools=False)
    invoke(loop,mode,initial_chat())
    assert len(engine.requests)==2
    second=engine.requests[1]
    assert second['contents'][1]['parts']==native
    assert sum('functionCall' in part for part in second['contents'][1]['parts'])==2
    responses=[part['functionResponse'] for message in second['contents'] for part in message['parts'] if 'functionResponse' in part]
    assert [item['id'] for item in responses]==['first','second']
    assert all(json.loads(item['response']['output'])==FULL for item in responses)


@pytest.mark.parametrize('mode', ['sync', 'sync-stream', 'async', 'async-stream'])
def test_basechat_observer_chat_projection_cannot_mutate_canonical_history(mode):
    from magic_llm.engine.base_chat import BaseChat
    class CallbackEngine(BaseChat):
        engine='openai'
        @BaseChat.sync_intercept_generate
        def generate(self, chat, **kwargs):
            return build_response(id='one',model='fake',content='done')
        @BaseChat.async_intercept_generate
        async def async_generate(self, chat, **kwargs):
            return build_response(id='one',model='fake',content='done')
        @BaseChat.sync_intercept_stream_generate
        def stream_generate(self, chat, **kwargs):
            yield build_stream_chunk(id='one',model='fake',content='done',finish_reason='stop')
        @BaseChat.async_intercept_stream_generate
        async def async_stream_generate(self, chat, **kwargs):
            yield build_stream_chunk(id='one',model='fake',content='done',finish_reason='stop')
    received=[]
    engine=CallbackEngine(model='fake',callback=lambda chat,*args:received.append(chat))
    chat=initial_chat()
    chat.add_tool_call_message([{'id':'one','function':{'name':'read_definitions','arguments':json.dumps(CALL)}}])
    chat.add_tool_result('one',json.dumps(FULL))
    original=copy.deepcopy(chat.messages)
    def project(observer_chat):
        observer_chat.messages[-1]['content']='{"loaded_ids":["one"]}'
        observer_chat.messages[0]['content']='observer copy only'
        return observer_chat
    chat.set_observer_projection(project)
    def guard(context):
        assert context.chat._observer_projection is project
    cls=AsyncAgentLoop if mode.startswith('async') else AgentLoop
    loop=cls(SimpleNamespace(llm=engine),request_guard=guard,builtin_todo_tools=False)
    invoke(loop,mode,chat)
    assert len(received)==1
    assert 'PRIVATE_' not in json.dumps(received[0].messages)
    assert received[0].messages[0]['content']=='observer copy only'
    assert chat.messages==original
    assert 'PRIVATE_PROMPT_SENTINEL' in json.dumps(loop.state.messages)


def test_observer_projection_failure_suppresses_callback_without_private_logging(caplog):
    from magic_llm.engine.base_chat import BaseChat
    received=[]
    chat=initial_chat()
    chat.require_complete_context()
    def reject(observer_chat):
        raise ValueError('PRIVATE_PROMPT_SENTINEL')
    chat.set_observer_projection(reject)
    # Use an existing concrete adapter's callback path.
    engine=EngineGoogle(api_key='test',model='fake',callback=lambda *args:received.append(args))
    engine._execute_callback_sync(chat,'answer',None,'fake',None)
    assert received==[]
    assert 'PRIVATE_PROMPT_SENTINEL' not in caplog.text



def test_protected_callback_failure_after_projection_never_logs_private_exception(caplog):
    chat=initial_chat()
    chat.require_complete_context()
    chat.set_observer_projection(lambda copied:copied)
    def fail(*args):
        raise RuntimeError('PRIVATE_PROMPT_SENTINEL')
    engine=EngineGoogle(api_key='test',model='fake',callback=fail)
    engine._execute_callback_sync(chat,'answer',None,'fake',None)
    assert 'PRIVATE_PROMPT_SENTINEL' not in caplog.text


def test_google_idless_function_calls_in_separate_stream_chunks_remain_distinct():
    engine=EngineGoogle(api_key='test',model='fake')
    summary=StreamIterationSummary()
    for _ in range(2):
        chunk=engine.prepare_stream_response('data: '+json.dumps({'candidates':[
            {'content':{'parts':[{'functionCall':{'name':'read_definitions','args':CALL},
                'thoughtSignature':'signature'}]},'finishReason':'STOP'}], 'usageMetadata':{}}))
        accumulate_stream_chunk(summary,chunk)
    calls=stream_summary_tool_calls(summary)
    assert len(calls)==2 and len({call.id for call in calls})==2
    assert summary.gemini_parts==[
        {'functionCall':{'name':'read_definitions','args':CALL},'thoughtSignature':'signature'},
        {'functionCall':{'name':'read_definitions','args':CALL},'thoughtSignature':'signature'},
    ]



def test_openai_wire_never_receives_native_google_replay_fields():
    chat=initial_chat()
    chat.add_assistant_message('Earlier answer',gemini_parts=[{'text':'Earlier answer','thoughtSignature':'opaque'}])
    chat.add_user_message('Continue')
    payload=json.loads(ProviderOpenAI(api_key='test',model='fake').prepare_data(chat)[0])
    assert 'gemini_parts' not in json.dumps(payload)
    assert 'thoughtSignature' not in json.dumps(payload)
    assert payload['messages'][-2]=={'role':'assistant','content':'Earlier answer'}


def test_opt_in_complete_output_invalidates_earlier_truncated_dedup_result():
    calls=[]
    def large():
        calls.append(1)
        return {'prompt':'x'*1000}
    executor=ToolExecutor(max_content_size=100)
    executor.register('large',large)
    first=executor.execute(CanonicalToolCall(id='one',name='large',arguments={}))
    assert '[TRUNCATED]' in first.content
    executor.require_complete_output('large')
    second=executor.execute(CanonicalToolCall(id='two',name='large',arguments={}))
    assert len(calls)==2
    assert second.error_type=='ToolOutputLimitError' and '[TRUNCATED]' not in second.content
    assert json.loads(second.content)['type']=='ToolOutputLimitError'
