import json
from types import SimpleNamespace
import pytest
from magic_llm.util.http import HttpError, AsyncHttpClient
from magic_llm.util.provider_errors import provider_error_detail

KEY = 'FAKE_USER_BYOK_CANARY_NEVER_LOG'

@pytest.mark.parametrize('status,code,expected', [
    (401, 'invalid_api_key', 'BYOK_CREDENTIAL_REJECTED'),
    (400, 'API_KEY_INVALID', 'BYOK_CREDENTIAL_REJECTED'),
    (403, 'permission_denied', 'PROVIDER_PERMISSION_DENIED'),
    (429, 'rate_limit_error', 'PROVIDER_RATE_LIMITED'),
    (429, 'insufficient_quota', 'PROVIDER_QUOTA_EXCEEDED'),
    (500, KEY, 'PROVIDER_UNAVAILABLE'),
])
def test_provider_errors_keep_only_safe_classification(status, code, expected):
    error = HttpError('raw error ' + KEY, status,
        json.dumps({'error': {'message': KEY, 'code': code}}).encode())
    detail = provider_error_detail(error, byok=True)
    assert detail['code'] == expected
    assert KEY not in str(error) + repr(error) + repr(vars(error)) + str(detail)
    assert error.response_content is None

class Context:
    def __init__(self, response): self.response = response
    async def __aenter__(self): return self.response
    async def __aexit__(self, *args): pass

@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_http_error_body_never_enters_logs_or_exception(stream, caplog):
    async def read(): return json.dumps({'error': {'code': 'invalid_api_key', 'message': KEY}}).encode()
    response = SimpleNamespace(status=401, read=read)
    calls = []
    def request(*args, **kwargs):
        calls.append(kwargs)
        return Context(response)
    client = AsyncHttpClient()
    client.session = SimpleNamespace(request=request)
    with pytest.raises(HttpError) as raised:
        if stream:
            [chunk async for chunk in client.stream_request('POST', 'https://provider.invalid', headers={'Authorization': 'Bearer ' + KEY})]
        else:
            await client.request('POST', 'https://provider.invalid', headers={'Authorization': 'Bearer ' + KEY})
    assert provider_error_detail(raised.value, byok=True)['code'] == 'BYOK_CREDENTIAL_REJECTED'
    assert len(calls) == 1
    assert KEY not in caplog.text + repr(vars(raised.value)) + str(raised.value)


@pytest.mark.parametrize('responses', [False, True])
@pytest.mark.parametrize('stream', [False, True])
def test_error_payloads_over_http_success_keep_authentication_classification(responses, stream):
    from magic_llm.engine.openai_adapters.openai_base import ProviderOpenAI
    from magic_llm.engine.openai_adapters.responses import OpenAIResponsesAdapter
    provider = ProviderOpenAI(api_key=KEY)
    adapter = OpenAIResponsesAdapter(provider) if responses else provider
    payload = {'error': {'code': 'invalid_api_key', 'message': 'Provider echo ' + KEY}}
    if responses:
        payload.update(type='error', status='failed')
    with pytest.raises(HttpError) as raised:
        if stream:
            adapter.transform_stream_chunk('data: ' + json.dumps(payload), {})
        else:
            adapter.transform_response(payload)
    assert provider_error_detail(raised.value, byok=True)['code'] == 'BYOK_CREDENTIAL_REJECTED'
    assert KEY not in str(raised.value) + repr(vars(raised.value))
