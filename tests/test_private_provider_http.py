from types import SimpleNamespace

import pytest

from magic_llm.util.http import HttpError
from magic_llm.util.private_http import PrivateAsyncHttpClient, PublicProviderResolver, validate_private_provider_url


@pytest.mark.parametrize('url', [
    'http://provider.example/v1', 'https://localhost/v1', 'https://127.0.0.1/v1',
    'https://192.168.1.90/v1', 'https://169.254.169.254/v1', 'https://[::1]/v1',
    'https://[ff02::1]/v1', 'https://224.0.0.1/v1', 'https://secret@provider.example/v1',
    'https://provider.example/v1?api_key=FAKE_CANARY', 'https://provider.example:8443/v1',
])
def test_private_destinations_and_secret_urls_are_rejected(url):
    with pytest.raises(HttpError) as error:
        validate_private_provider_url(url)
    assert 'FAKE_CANARY' not in str(error.value)


class ResponseContext:
    def __init__(self, response): self.response = response
    async def __aenter__(self): return self.response
    async def __aexit__(self, *args): pass


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [301, 302, 307, 401, 403, 429, 500])
@pytest.mark.parametrize('stream', [False, True])
async def test_private_error_does_not_read_body_log_or_redirect(status, stream, caplog):
    received = []
    response = SimpleNamespace(status=status, headers={})
    # An error response intentionally has no read() method: it must not be read.
    def request(method, url, **kwargs):
        received.append(kwargs)
        return ResponseContext(response)
    client = PrivateAsyncHttpClient()
    client.session = SimpleNamespace(request=request)
    with pytest.raises(HttpError) as error:
        if stream:
            [line async for line in client.stream_request('POST', 'https://provider.example/v1', headers={'Authorization': 'Bearer FAKE_CANARY'})]
        else:
            await client.request('POST', 'https://provider.example/v1', headers={'Authorization': 'Bearer FAKE_CANARY'})
    assert error.value.status_code == status and error.value.response_content is None
    assert received[0]['allow_redirects'] is False
    assert received[0]['headers']['Accept-Encoding'] == 'identity'
    assert received[0]['timeout'].total == 120
    assert 'FAKE_CANARY' not in str(error.value) and 'FAKE_CANARY' not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('addresses', [['127.0.0.1'], ['10.0.0.1'], ['169.254.169.254'], ['8.8.8.8', '::1'], ['224.0.0.1']])
async def test_actual_dns_results_must_all_be_public(addresses):
    resolver = PublicProviderResolver()
    await resolver._resolver.close()
    async def resolve(*args): return [{'host': address} for address in addresses]
    resolver._resolver = SimpleNamespace(resolve=resolve)
    with pytest.raises(HttpError):
        await resolver.resolve('provider.example', 443)


@pytest.mark.asyncio
async def test_dns_uses_validated_public_records():
    resolver = PublicProviderResolver()
    await resolver._resolver.close()
    records = [{'host': '8.8.8.8', 'port': 443}]
    async def resolve(*args): return records
    resolver._resolver = SimpleNamespace(resolve=resolve)
    assert await resolver.resolve('provider.example', 443) == records


@pytest.mark.parametrize('timeout', [0, -1, 601, float('inf'), float('nan'), '120'])
def test_timeout_has_a_finite_total_limit(timeout):
    with pytest.raises(HttpError):
        PrivateAsyncHttpClient()._options('https://provider.example/v1', {'timeout': timeout})
