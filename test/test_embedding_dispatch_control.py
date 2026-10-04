"""Physical embedding admission uses the actual bounded HTTP path, without networking."""
import asyncio
import gzip
import json
from types import SimpleNamespace

import pytest

from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.util import http
from magic_llm.util.http import AsyncHttpClient, HttpError

pytestmark = pytest.mark.asyncio


class Control:
    def __init__(self): self.calls=[]; self.allowed=True
    async def call(self, kind, request, operation):
        self.calls.append((kind, request))
        if not self.allowed: raise PermissionError('revoked')
        return await operation()


class Transport:
    _retry_connection = True
    def __init__(self, chunks, *, status=200, encoding=None):
        self.chunks=chunks; self.status=status; self.calls=[]; self.reads=0; self.closed=0
        self.headers={'Content-Encoding':encoding} if encoding else {}
    def factory(self,*args,**kwargs): return self
    async def close(self): self.closed+=1
    def request(self,*args,**kwargs):self.calls.append((args,kwargs));return self
    async def __aenter__(self):return self
    async def __aexit__(self,*args):return None
    @property
    def content(self):return self
    async def iter_chunked(self,size):
        for chunk in self.chunks:
            self.reads+=1
            yield chunk
    async def read(self):self.reads+=1;return b''.join(self.chunks)


def body(model='override'):
    return json.dumps({'model':model,'data':[{'index':0,'embedding':[.25, .5]}],
                       'usage':{'prompt_tokens':2,'total_tokens':2}}).encode()


async def test_real_embedding_admits_resolved_override_and_private_headers_before_http(monkeypatch):
    wire=Transport([body()]);monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
    engine=EngineOpenAI(api_key='private-key',model='original',base_url='https://example.invalid/v1')
    control=Control();result=await engine.async_embedding('hello',model='override',dimensions=2,
        timeout=7,external_dispatch=control,external_kind='memory.embedding.write')
    assert result.model=='override' and result.data[0].embedding==[.25,.5]
    kind,request=control.calls[0]
    assert kind=='memory.embedding.write' and request['url']=='https://example.invalid/v1/embeddings'
    assert request['json']=={'input':'hello','model':'override','dimensions':2}
    assert request['headers']['Authorization']=='Bearer private-key'
    args,kwargs=wire.calls[0]
    assert args==('POST',request['url']) and json.loads(kwargs['data'])==request['json']
    assert kwargs['allow_redirects'] is False and kwargs['auto_decompress'] is False
    assert kwargs['timeout'].total==7 and 'external_dispatch' not in kwargs
    assert 'max_response_bytes' not in kwargs and wire.closed==1


async def test_denied_embedding_does_not_open_transport(monkeypatch):
    wire=Transport([body()]);monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
    control=Control();control.allowed=False
    with pytest.raises(PermissionError):
        await EngineOpenAI(api_key='test',model='original').async_embedding('hello',external_dispatch=control)
    assert not wire.calls and wire.closed==0


@pytest.mark.parametrize('timeout',[None,0,True,float('inf'),301])
async def test_controlled_embedding_rejects_unbounded_timeout_before_admission(timeout):
    control=Control()
    with pytest.raises(ValueError):
        await EngineOpenAI(api_key='test',model='original').async_embedding('hello',external_dispatch=control,timeout=timeout)
    assert not control.calls


async def test_legacy_embedding_preserves_payload_and_transport_defaults(monkeypatch):
    wire=Transport([body()]);monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
    await EngineOpenAI(api_key='test',model='original').async_embedding('hello',model='override',timeout=None)
    kwargs=wire.calls[0][1]
    assert json.loads(kwargs['data'])['timeout'] is None
    assert kwargs['timeout'].total is None and 'allow_redirects' not in kwargs
    assert 'auto_decompress' not in kwargs and wire.reads==1


@pytest.mark.parametrize('status',[302,503])
async def test_controlled_http_does_not_follow_redirect_or_repeat_failed_submission(monkeypatch,status):
    wire=Transport([b'private response'],status=status);monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
    with pytest.raises(HttpError,match=f'HTTP {status}'):
        await EngineOpenAI(api_key='test',model='original').async_embedding('hello',external_dispatch=Control())
    assert len(wire.calls)==1 and wire.calls[0][1]['allow_redirects'] is False


@pytest.mark.parametrize('encoding',['identity','gzip','deflate'])
async def test_controlled_http_bounds_encoded_and_decoded_streams(monkeypatch,encoding):
    import zlib
    decoded=b'12345'
    encoded=gzip.compress(decoded) if encoding=='gzip' else zlib.compress(decoded) if encoding=='deflate' else decoded
    wire=Transport([encoded],encoding=encoding);monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
    async with AsyncHttpClient() as client:
        assert await client.request('GET','https://example.invalid',max_response_bytes=128)==decoded
    # A tiny compressed request expands beyond the decoded bound. The decoder
    # itself receives max_length, before retaining the returned bytes.
    bomb=gzip.compress(b'x'*1000000)
    wire=Transport([bomb,b'not read'],encoding='gzip');monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
    async with AsyncHttpClient() as client:
        with pytest.raises(HttpError,match='byte limit'):
            await client.request('GET','https://example.invalid',max_response_bytes=2048)
    assert wire.reads==1 and wire.calls[0][1]['auto_decompress'] is False


async def test_controlled_http_stops_plain_body_and_rejects_unknown_encoding(monkeypatch):
    for wire in [Transport([b'1234',b'never']),Transport([b'never'],encoding='br')]:
        monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
        async with AsyncHttpClient() as client:
            with pytest.raises(HttpError):await client.request('GET','https://example.invalid',max_response_bytes=3)
        assert wire.reads<=1


async def test_request_snapshot_does_not_share_mutable_input_or_callback_state(monkeypatch):
    wire=Transport([body()]);monkeypatch.setattr(http.aiohttp,'ClientSession',wire.factory)
    original=['hello']
    class Mutating(Control):
        async def call(self,kind,request,operation):
            original[0]='changed';request['json']['input'][0]='changed again'
            request['headers'].clear()
            return await operation()
    await EngineOpenAI(api_key='test',model='original').async_embedding(original,external_dispatch=Mutating())
    kwargs=wire.calls[0][1]
    assert json.loads(kwargs['data'])['input']==['hello'] and kwargs['headers']['Authorization']=='Bearer test'


@pytest.mark.parametrize('controlled,expected',[(False,2),(True,1)])
async def test_actual_aiohttp_dispatch_loop_has_no_hidden_controlled_retry(controlled,expected):
    # Exercise ClientSession._request's real retry loop. Middleware fails before
    # networking, so this proves transport policy without sockets or providers.
    calls=[]
    async def disconnect(request,handler):
        calls.append(request.url)
        raise http.aiohttp.ServerDisconnectedError('ambiguous remote outcome')
    async with http.aiohttp.ClientSession(middlewares=[disconnect]) as session:
        client=AsyncHttpClient();client.session=session
        options={'max_response_bytes':100} if controlled else {}
        with pytest.raises(HttpError):await client.request('GET','https://example.invalid',**options)
    assert len(calls)==expected
