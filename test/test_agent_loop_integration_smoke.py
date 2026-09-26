"""Thin offline agent-loop seam using a real engine response shape."""

import json
from types import SimpleNamespace

import pytest

from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.types import AgentBudget
from magic_llm.engine.engine_openai import EngineOpenAI


class CapturingHttpClient:
    instances = []

    def __init__(self):
        self.calls = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def post_json(self, *, url, data=None, json=None, headers=None, timeout=30, **kwargs):
        payload = json_module_loads(data)
        self.calls.append({'url': url, 'data': payload, 'headers': headers, 'timeout': timeout, 'kwargs': kwargs})
        return {
            'id': 'chatcmpl-agent-smoke',
            'object': 'chat.completion',
            'created': 1700000000,
            'model': payload['model'],
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'real engine shape handled'},
                'finish_reason': 'stop',
            }],
            'usage': {'prompt_tokens': 4, 'completion_tokens': 5, 'total_tokens': 9},
        }


def json_module_loads(raw):
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return json.loads(raw)


@pytest.mark.integration
def test_agent_loop_consumes_real_openai_engine_response_shape(monkeypatch):
    """AgentLoop runs through EngineOpenAI with only the HTTP boundary mocked."""
    import magic_llm.engine.engine_openai as mod

    CapturingHttpClient.instances.clear()
    monkeypatch.setattr(mod, 'HttpClient', CapturingHttpClient)

    engine = EngineOpenAI(api_key='dummy-key', model='gpt-4o-mini')
    client = SimpleNamespace(llm=engine)
    loop = AgentLoop(client=client, tools=[], budget=AgentBudget(max_iterations=1))

    response = loop.run('Say hello once.')

    assert response.content == 'real engine shape handled'
    assert response.tool_calls is None
    assert response.finish_reason == 'stop'
    assert response.usage.total_tokens == 9
    call = CapturingHttpClient.instances[0].calls[0]
    assert call['url'] == 'https://api.openai.com/v1/chat/completions'
    assert call['data']['model'] == 'gpt-4o-mini'
    assert call['headers']['Authorization'].startswith('Bearer ')
