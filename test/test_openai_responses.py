"""Responses contracts at the HTTP boundary; all engines and agent loops are real."""
import copy
import json

import pytest

from magic_llm import MagicLLM
from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.engine.tooling import StreamIterationSummary, accumulate_stream_chunk
from magic_llm.model import ModelChat


REASONING = {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque-test-state'}
CALL = {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1',
        'name': 'double', 'arguments': '{"value":3}', 'status': 'completed'}
TEXT = {'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'status': 'completed',
        'content': [{'type': 'output_text', 'text': 'Six.', 'annotations': []}]}
TOOL = {'type': 'function', 'function': {'name': 'double', 'description': 'Double a value',
        'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']}}}


def response(output=None, **overrides):
    return {'id': 'resp_1', 'created_at': 1, 'model': 'gpt-6-luna', 'status': 'completed',
            'output': copy.deepcopy(output if output is not None else [TEXT]),
            'usage': {'input_tokens': 10, 'output_tokens': 6, 'total_tokens': 16,
                      'input_tokens_details': {'cached_tokens': 4},
                      'output_tokens_details': {'reasoning_tokens': 2}}, **overrides}


def stream_events(reply):
    yield 'event: response.created'
    yield 'data: ' + json.dumps({'type': 'response.created', 'response': {**reply, 'output': [], 'usage': None}})
    for index, item in enumerate(reply['output']):
        if item['type'] == 'function_call':
            yield 'data: ' + json.dumps({'type': 'response.output_item.added', 'output_index': index,
                                        'item': {**item, 'arguments': ''}})
            for delta in ['{"value":', '3}']:
                yield 'data: ' + json.dumps({'type': 'response.function_call_arguments.delta',
                                            'output_index': index, 'delta': delta})
            yield 'data: ' + json.dumps({'type': 'response.function_call_arguments.done',
                                        'output_index': index, 'arguments': item['arguments']})
        elif item['type'] == 'message':
            for delta in ['Six', '.']:
                yield 'data: ' + json.dumps({'type': 'response.output_text.delta', 'delta': delta})
        yield 'data: ' + json.dumps({'type': 'response.output_item.done', 'output_index': index, 'item': item})
    yield 'data: ' + json.dumps({'type': 'response.completed', 'response': reply})


@pytest.fixture
def transport(monkeypatch):
    requests, replies = [], []

    def receive(url, data, **kwargs):
        requests.append({'url': url, 'body': json.loads(data), **kwargs})
        return replies.pop(0)

    class SyncHttp:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post_json(self, *, url, data, **kwargs): return receive(url, data, **kwargs)
        def stream_request(self, method, url, data, **kwargs):
            yield from stream_events(receive(url, data, **kwargs))

    class AsyncHttp:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post_json(self, *, url, data, **kwargs): return receive(url, data, **kwargs)
        async def post_stream(self, url, data, **kwargs):
            for event in stream_events(receive(url, data, **kwargs)):
                yield event.encode()

    monkeypatch.setattr('magic_llm.engine.engine_openai.HttpClient', SyncHttp)
    monkeypatch.setattr('magic_llm.engine.engine_openai.AsyncHttpClient', AsyncHttp)
    return requests, replies


def chat():
    result = ModelChat('Be helpful')
    result.add_user_message('Double three')
    return result


def engine(**kwargs):
    return EngineOpenAI(api_key='test-only', model='gpt-6-luna', endpoint='responses', **kwargs)


def test_request_maps_messages_images_tools_reasoning_and_json_schema_without_mutation():
    conversation = chat()
    conversation.add_user_message('Look', image='https://example.com/image.png')
    conversation.add_tool_call_message([{'id': 'old', 'function': {'name': 'double', 'arguments': '{"value":1}'}}])
    conversation.add_tool_result('old', '2')
    before = copy.deepcopy(conversation.messages)
    payload, _ = engine().transform_request(conversation, tools=[TOOL],
        tool_choice={'type': 'function', 'function': {'name': 'double'}}, reasoning_effort='high',
        max_tokens=2048, temperature=1, top_p=1, stream=True, timeout=12,
        response_format={'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': {'type': 'object'}}})
    data = json.loads(payload)
    assert data['input'][0] == {'role': 'system', 'content': 'Be helpful'}
    assert data['input'][2]['content'][1] == {'type': 'input_image', 'image_url': 'https://example.com/image.png', 'detail': 'auto'}
    assert data['input'][-2]['call_id'] == data['input'][-1]['call_id'] == 'old'
    assert data['input'][-1]['output'] == '2'
    assert data['tools'][0] == {'type': 'function', 'strict': False, **TOOL['function']}
    assert data['tool_choice'] == {'type': 'function', 'name': 'double'}
    assert data['max_output_tokens'] == 2048
    assert data['reasoning'] == {'effort': 'high'}
    assert data['text']['format'] == {'type': 'json_schema', 'name': 'answer', 'schema': {'type': 'object'}}
    assert data['store'] is False
    assert data['include'] == ['reasoning.encrypted_content']
    assert not {'messages', 'max_tokens', 'temperature', 'top_p', 'stream_options', 'timeout', 'endpoint'} & data.keys()
    assert conversation.messages == before


@pytest.mark.parametrize('params', [{'reasoning_effort': 'none'}, {'base_url': 'https://provider.example/v1'}])
def test_sampling_is_preserved_when_supported_or_using_custom_provider(params):
    data = json.loads(engine(**params).transform_request(chat(), temperature=.3, top_p=.8)[0])
    assert data['temperature'] == .3 and data['top_p'] == .8


def test_explicit_responses_options_win_over_legacy_aliases():
    data = json.loads(engine(reasoning_effort='high', max_tokens=32).transform_request(chat(), max_output_tokens=256,
        reasoning={'effort': 'none'}, text={'format': {'type': 'text'}}, json_output=True)[0])
    assert data['max_output_tokens'] == 256
    assert data['reasoning']['effort'] == 'none'
    assert data['text']['format']['type'] == 'text'


@pytest.mark.parametrize('mode', ['sync', 'async', 'sync_stream', 'async_stream'])
async def test_all_generation_paths_use_selected_url_and_normalize_usage(transport, mode):
    requests, replies = transport
    replies.append(response())
    client = engine(base_url='https://api.openai.com/v1/')
    if mode == 'sync': result = client.generate(chat())
    elif mode == 'async': result = await client.async_generate(chat())
    else:
        chunks = list(client.stream_generate(chat())) if mode == 'sync_stream' else [c async for c in client.async_stream_generate(chat())]
        assert ''.join(c.choices[0].delta.content or '' for c in chunks) == 'Six.'
        assert chunks[-1].choices[0].finish_reason == 'stop'
        result = chunks[-1]
    if not mode.endswith('stream'): assert result.content == 'Six.'
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 6
    assert result.usage.prompt_tokens_details.cached_tokens == 4
    assert result.usage.completion_tokens_details.reasoning_tokens == 2
    assert result.usage.provider_request_id == 'resp_1'
    assert requests[0]['url'] == 'https://api.openai.com/v1/responses'


@pytest.mark.parametrize('mode', ['sync', 'async', 'sync_stream', 'async_stream'])
async def test_real_agent_tool_round_trip_replays_reasoning_and_results(transport, mode):
    requests, replies = transport
    replies.extend([response([REASONING, CALL]), response(id='resp_2')])
    executed = []
    def double(value: int):
        executed.append(value)
        return value * 2
    client = MagicLLM(engine='openai', private_key='test-only', model='gpt-6-luna', endpoint='responses')
    options = {'tools': [TOOL], 'tool_functions': {'double': double}, 'max_iterations': 2, 'reasoning_effort': 'medium'}
    if mode == 'sync': result = client.run_agent('Double three', **options)
    elif mode == 'async': result = await client.run_agent_async('Double three', **options)
    elif mode == 'sync_stream': result = list(client.run_agent_stream('Double three', **options))
    else: result = [chunk async for chunk in client.run_agent_stream_async('Double three', **options)]
    assert executed == [3]
    assert len(requests) == 2
    history = requests[1]['body']['input']
    assert history[1:3] == [REASONING, CALL]
    assert history[3] == {'type': 'function_call_output', 'call_id': 'call_1', 'output': '6'}
    assert all(r['body']['reasoning']['effort'] == 'medium' for r in requests)
    if mode.endswith('stream'):
        assert ''.join(chunk.choices[0].delta.content or '' for chunk in result).strip() == 'Six.'
    else: assert result.content == 'Six.'


def test_stream_ignores_event_headers_and_done_snapshots_and_handles_sparse_indices():
    client, summary, context = engine(), StreamIterationSummary(), {}
    for raw in stream_events(response([REASONING, CALL])):
        chunk = client.transform_stream_chunk(raw, context)
        if chunk: accumulate_stream_chunk(summary, chunk)
    assert summary.tool_calls == [{'index': 1, 'id': 'call_1', 'function': {'name': 'double', 'arguments': '{"value":3}'}}]
    assert summary.responses_output == [REASONING, CALL]
    assert summary.finish_reason == 'tool_calls'


def test_refusal_incomplete_and_failure_are_not_silent_success():
    client = engine()
    assert client.transform_response(response(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})).finish_reason == 'length'
    refusal = {**TEXT, 'content': [{'type': 'refusal', 'refusal': 'Cannot help'}]}
    assert client.transform_response(response([refusal])).choices[0].message.refusal == 'Cannot help'
    with pytest.raises(ValueError, match='rejected'):
        client.transform_response(response(status='failed', error={'message': 'rejected'}))
    with pytest.raises(ValueError, match='rejected'):
        client.transform_stream_chunk({'type': 'error', 'message': 'rejected'}, {})


def test_legacy_default_and_chat_wire_stay_compatible():
    client = EngineOpenAI(api_key='test-only', model='gpt-4o')
    conversation = chat()
    conversation.add_assistant_message('Six.', responses_output=[TEXT])
    data = json.loads(client.transform_request(conversation)[0])
    assert client.chat_url.endswith('/chat/completions')
    assert data['messages'][-1] == {'role': 'assistant', 'content': 'Six.'}
    assert 'input' not in data and 'endpoint' not in data
    with pytest.raises(ValueError, match='endpoint'):
        EngineOpenAI(api_key='test-only', endpoint='typo')


def test_incomplete_stream_cannot_execute_partial_tool_arguments():
    client, context = engine(), {}
    client.transform_stream_chunk({'type': 'response.output_item.added', 'output_index': 0,
                                   'item': {**CALL, 'arguments': ''}}, context)
    with pytest.raises(ValueError, match='incomplete tool calls'):
        client.transform_stream_chunk({'type': 'response.incomplete', 'response': response(
            [CALL], status='incomplete', incomplete_details={'reason': 'max_output_tokens'})}, context)


@pytest.mark.parametrize('asynchronous', [False, True])
async def test_stream_without_terminal_event_fails(monkeypatch, transport, asynchronous):
    transport[1].append(response())
    complete_events = stream_events
    def truncated(reply):
        yield from list(complete_events(reply))[:-1]
    monkeypatch.setattr(__import__(__name__, fromlist=['stream_events']), 'stream_events', truncated)
    client = engine(retries=1)
    with pytest.raises(Exception, match='terminal event'):
        if asynchronous:
            [chunk async for chunk in client.async_stream_generate(chat())]
        else:
            list(client.stream_generate(chat()))


def test_stream_context_and_internal_reasoning_are_isolated():
    client = engine()
    first, second = {}, {}
    client.transform_stream_chunk({'type': 'response.created', 'response': response(id='resp_first')}, first)
    client.transform_stream_chunk({'type': 'response.created', 'response': response(id='resp_second')}, second)
    event = {'type': 'response.output_text.delta', 'delta': 'Hello'}
    assert client.transform_stream_chunk(event, first).id == 'resp_first'
    assert client.transform_stream_chunk(event, second).id == 'resp_second'
    final = client.transform_stream_chunk({'type': 'response.completed', 'response': response([REASONING, CALL])}, first)
    assert final.responses_output == [REASONING, CALL]
    assert 'responses_output' not in final.model_dump()


def test_per_call_effort_overrides_nested_client_default_and_preserves_summary():
    data = json.loads(engine(reasoning={'effort': 'low', 'summary': 'auto'}).transform_request(
        chat(), reasoning_effort='high')[0])
    assert data['reasoning'] == {'effort': 'high', 'summary': 'auto'}
