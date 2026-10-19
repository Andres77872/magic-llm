"""Custom URLs retain their destination and do not match provider substrings."""
import json

import pytest

from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.engine.openai_adapters import (
    ProviderDeepInfra,
    ProviderDeepseek,
    ProviderFireworks,
    ProviderGroq,
    ProviderMistral,
    ProviderOpenAI,
    ProviderOpenRouter,
    ProviderSambaNova,
    ProviderTogether,
)
from magic_llm.model import ModelChat


@pytest.fixture
def make_engine():
    clients = []

    def create(**kwargs):
        client = EngineOpenAI(api_key='FAKE_ROUTING_KEY', model='my/manual-model', retries=1, **kwargs)
        clients.append(client)
        return client

    yield create
    for client in clients:
        client.executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.parametrize('endpoint', ['chat_completions', 'responses'])
@pytest.mark.parametrize('base_url', [
    'https://gateway.example/proxy/openrouter.ai/v1',
    'https://gateway.example/proxy/api.groq.com/v1',
    'https://gateway.example/api.deepinfra.com/v1/openai',
    'https://openrouter.ai.gateway.example/v1',
    'https://notopenrouter.ai/v1',
    'https://api.groq.com.gateway.example/v1',
])
def test_provider_names_outside_hostname_do_not_change_adapter_or_destination(make_engine, endpoint, base_url):
    client = make_engine(base_url=base_url, endpoint=endpoint)
    assert type(client.base) is ProviderOpenAI
    resource = '/responses' if endpoint == 'responses' else '/chat/completions'
    assert client.chat_url == base_url.rstrip('/') + resource


@pytest.mark.parametrize('url', [
    'https://openrouter.ai@gateway.example/v1',
    'https://gateway.example/v1?provider=openrouter.ai',
])
def test_url_metadata_is_not_provider_identity(make_engine, url):
    client = make_engine()
    assert client._get_provider_for_url(url) is ProviderOpenAI


@pytest.mark.parametrize('endpoint', ['chat_completions', 'responses'])
@pytest.mark.parametrize(('base_url', 'adapter'), [
    ('https://api.groq.com/openai/v1', ProviderGroq),
    ('https://api.sambanova.ai/v1', ProviderSambaNova),
    ('https://openrouter.ai/api/v1', ProviderOpenRouter),
    ('https://api.mistral.ai/v1', ProviderMistral),
    ('https://api.fireworks.ai/inference/v1', ProviderFireworks),
    ('https://api.deepseek.com/v1', ProviderDeepseek),
    ('https://api.deepinfra.com/v1/openai', ProviderDeepInfra),
    ('https://api.together.xyz/v1', ProviderTogether),
    ('https://OPENROUTER.AI/proxy-prefix/v1/', ProviderOpenRouter),
])
def test_known_provider_hosts_preserve_the_explicit_api_base(make_engine, endpoint, base_url, adapter):
    client = make_engine(base_url=base_url, endpoint=endpoint)
    assert type(client.base) is adapter
    resource = '/responses' if endpoint == 'responses' else '/chat/completions'
    assert client.chat_url == base_url.rstrip('/') + resource


@pytest.mark.parametrize('endpoint', ['chat_completions', 'responses'])
def test_custom_mode_can_explicitly_select_generic_adapter_on_a_known_host(make_engine, endpoint):
    base_url = 'https://openrouter.ai/custom-prefix/v1'
    client = make_engine(base_url=base_url, endpoint=endpoint, openai_adapter=ProviderOpenAI)
    assert type(client.base) is ProviderOpenAI
    resource = '/responses' if endpoint == 'responses' else '/chat/completions'
    assert client.chat_url == base_url + resource


def test_provider_api_prefix_matching_respects_path_boundaries(make_engine):
    client = make_engine(base_url='https://api.deepinfra.com/v10/custom')
    assert type(client.base) is ProviderOpenAI
    assert client.chat_url == 'https://api.deepinfra.com/v10/custom/chat/completions'


def test_custom_request_uses_the_exact_url_and_manual_model_id(make_engine, monkeypatch):
    requests = []
    base_url = 'https://gateway.example/proxy/openrouter.ai/v1'

    class Http:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post_json(self, *, url, data, headers, **kwargs):
            requests.append((url, json.loads(data), headers))
            return {'id': 'custom-reply', 'object': 'chat.completion', 'created': 0,
                    'model': 'my/manual-model',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'},
                                 'finish_reason': 'stop'}]}

    monkeypatch.setattr('magic_llm.engine.engine_openai.HttpClient', Http)
    client = make_engine(base_url=base_url)
    chat = ModelChat()
    chat.add_user_message('Hello')
    result = client.generate(chat)
    assert result.choices[0].message.content == 'OK'
    url, body, headers = requests.pop()
    assert not requests
    assert url == base_url + '/chat/completions'
    assert body['model'] == 'my/manual-model'
    assert headers['Authorization'] == 'Bearer FAKE_ROUTING_KEY'
