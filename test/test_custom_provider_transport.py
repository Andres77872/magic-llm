"""Custom OpenAI calls use guarded transport without provider discovery."""
import json

import pytest

from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.engine.openai_adapters.openai_base import ProviderOpenAI
from magic_llm.model import ModelChat
from magic_llm.util.http import HttpError


@pytest.fixture
def engine():
    clients = []
    def create(**kwargs):
        value = EngineOpenAI(api_key='FAKE_CUSTOM_KEY', model='never-listed/manual-id',
            base_url='https://custom.example/proxy/openrouter.ai/v1', private_transport=True,
            retries=1, **kwargs)
        clients.append(value)
        return value
    yield create
    for value in clients:
        value.executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.parametrize('endpoint', ['chat_completions', 'responses'])
@pytest.mark.parametrize('stream', [False, True])
async def test_custom_calls_use_the_guarded_client_and_literal_model(engine, monkeypatch, endpoint, stream):
    calls = []
    response = {'id': 'custom-response', 'created_at': 1, 'model': 'never-listed/manual-id',
        'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
            'content': [{'type': 'output_text', 'text': 'OK'}]}],
        'usage': {'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3}}
    completion = {'id': 'custom-response', 'object': 'chat.completion', 'created': 1,
        'model': 'never-listed/manual-id', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': 'OK'}}],
        'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}
    class Guarded:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def record(self, url, data, headers, **kwargs):
            calls.append((url, json.loads(data), headers))
        async def post_json(self, **kwargs):
            self.record(**kwargs)
            return response if endpoint == 'responses' else completion
        async def post_stream(self, url, **kwargs):
            self.record(url=url, **kwargs)
            if endpoint == 'responses':
                events = [{'type': 'response.created', 'response': response},
                    {'type': 'response.output_text.delta', 'delta': 'OK'},
                    {'type': 'response.completed', 'response': response}]
            else:
                events = [{**completion, 'object': 'chat.completion.chunk', 'choices': [
                    {'index': 0, 'finish_reason': 'stop', 'delta': {'content': 'OK'}}]}]
            for event in events:
                yield ('data: ' + json.dumps(event)).encode()
    def ordinary():
        raise AssertionError('Custom calls must not use the ordinary HTTP client')
    monkeypatch.setattr('magic_llm.engine.engine_openai.PrivateAsyncHttpClient', Guarded)
    monkeypatch.setattr('magic_llm.engine.engine_openai.AsyncHttpClient', ordinary)
    client = engine(endpoint=endpoint)
    chat = ModelChat(); chat.add_user_message('Hello')
    if stream:
        chunks = [chunk async for chunk in client.async_stream_generate(chat)]
        assert ''.join(chunk.choices[0].delta.content or '' for chunk in chunks) == 'OK'
    else:
        assert (await client.async_generate(chat)).choices[0].message.content == 'OK'
    assert type(client.base) is ProviderOpenAI
    assert len(calls) == 1
    url, body, headers = calls[0]
    resource = '/responses' if endpoint == 'responses' else '/chat/completions'
    assert url == 'https://custom.example/proxy/openrouter.ai/v1' + resource
    assert body['model'] == 'never-listed/manual-id'
    assert headers['Authorization'] == 'Bearer FAKE_CUSTOM_KEY'
    assert not {'private_transport', 'openai_adapter', 'include', 'stream_options'} & body.keys()
    if endpoint == 'responses': assert body['store'] is False


@pytest.mark.parametrize('url', ['http://provider.example/v1', 'https://localhost/v1',
    'https://127.0.0.1/v1', 'https://192.168.1.9/v1', 'https://169.254.169.254/v1',
    'https://[::1]/v1', 'https://provider.example/v1?api_key=secret'])
def test_private_destinations_fail_before_client_construction(url):
    with pytest.raises(HttpError):
        EngineOpenAI(api_key='FAKE_CUSTOM_KEY', model='manual', base_url=url, private_transport=True)
