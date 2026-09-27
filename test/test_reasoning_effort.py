"""Effort survives provider wire conversion and per-request/client precedence."""
import copy
import json

import pytest

from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.model import ModelChat


def payload(endpoint='chat_completions', defaults=None, **kwargs):
    engine = EngineOpenAI(api_key='test-only', model='provider-model', endpoint=endpoint,
                         base_url='https://provider.example/v1', **(defaults or {}))
    chat = ModelChat()
    chat.add_user_message('Hello')
    raw, _ = engine.transform_request(chat, **kwargs)
    return json.loads(raw)


@pytest.mark.parametrize('endpoint', ['chat_completions', 'responses'])
@pytest.mark.parametrize('effort', ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'Vendor/Deep-v2'])
@pytest.mark.parametrize('stream', [False, True])
def test_provider_values_pass_through(endpoint, effort, stream):
    data = payload(endpoint, reasoning_effort=effort, stream=stream)
    if endpoint == 'responses':
        assert data['reasoning']['effort'] == effort
        assert 'reasoning_effort' not in data
    else:
        assert data['reasoning_effort'] == effort
    assert data['stream'] == stream


@pytest.mark.parametrize('endpoint', ['chat_completions', 'responses'])
def test_no_setting_leaves_provider_default(endpoint):
    data = payload(endpoint)
    assert 'reasoning_effort' not in data
    assert 'reasoning' not in data


def test_call_effort_overrides_nested_client_default_without_mutation():
    defaults = {'reasoning': {'effort': 'low', 'summary': 'auto'}}
    before = copy.deepcopy(defaults)
    assert payload('responses', defaults, reasoning_effort='high')['reasoning'] == {'effort': 'high', 'summary': 'auto'}
    assert defaults == before
    assert payload('responses', defaults)['reasoning']['effort'] == 'low'


def test_call_nested_effort_overrides_flat_client_default():
    assert payload('responses', {'reasoning_effort': 'low'}, reasoning={'effort': 'high'})['reasoning']['effort'] == 'high'


def test_nullable_nested_reasoning_accepts_flat_effort():
    assert payload('responses', {'reasoning': None}, reasoning_effort='high')['reasoning']['effort'] == 'high'


def test_partial_reasoning_options_preserve_client_effort():
    assert payload('responses', {'reasoning': {'effort': 'low'}}, reasoning={'summary': 'auto'})['reasoning'] == {
        'effort': 'low', 'summary': 'auto',
    }
