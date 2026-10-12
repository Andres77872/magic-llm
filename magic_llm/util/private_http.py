"""Request-scoped transport for transient provider credentials.

No redirects, proxy environment, raw error bodies or private-network DNS results.
The caller must also authorize the target before constructing the client.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import socket
from urllib.parse import urlsplit

import aiohttp

from magic_llm.util.http import AsyncHttpClient, HttpError, disable_automatic_retries, read_bounded_response


class PublicProviderResolver(aiohttp.abc.AbstractResolver):
    def __init__(self):
        self._resolver = aiohttp.resolver.DefaultResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        try:
            records = await self._resolver.resolve(host, port, family)
            if not records or any(not _public_address(row['host']) for row in records):
                raise HttpError('Provider destination is not public')
            return records
        except HttpError:
            raise
        except Exception:
            raise HttpError('Provider DNS resolution failed') from None

    async def close(self):
        await self._resolver.close()


def _public_address(value):
    address = ipaddress.ip_address(value)
    return address.is_global and not address.is_multicast and not address.is_reserved


def validate_private_provider_url(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.port not in (None, 443)):
            raise ValueError()
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            if parsed.hostname.lower() == 'localhost' or parsed.hostname.lower().endswith('.localhost'):
                raise ValueError()
        else:
            if not _public_address(address):
                raise ValueError()
    except (ValueError, TypeError):
        raise HttpError('Provider destination is not permitted') from None


class PrivateAsyncHttpClient(AsyncHttpClient):
    max_response_bytes = 16 * 1024 * 1024

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(resolver=PublicProviderResolver(), use_dns_cache=False)
        self.session = aiohttp.ClientSession(connector=connector, trust_env=False)
        try:
            disable_automatic_retries(self.session)
        except Exception:
            await self.session.close()
            self.session = None
            raise HttpError('Private provider transport is unavailable') from None
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await super().__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self.session = None

    def _options(self, url, kwargs):
        validate_private_provider_url(url)
        if any(key in kwargs for key in ('proxy', 'proxy_auth', 'ssl', 'server_hostname', 'auth')):
            raise HttpError('Provider transport override is not permitted')
        seconds = kwargs.pop('timeout', 120)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 600:
            raise HttpError('Provider timeout is invalid')
        kwargs.pop('max_response_bytes', None)
        kwargs['allow_redirects'] = False
        kwargs['auto_decompress'] = False
        kwargs['headers'] = {**kwargs.get('headers', {}), 'Accept-Encoding': 'identity'}
        kwargs['timeout'] = aiohttp.ClientTimeout(total=seconds, sock_read=seconds, connect=min(seconds, 30))
        return kwargs

    async def request(self, method, url, **kwargs):
        self._ensure_session()
        options = self._options(url, kwargs)
        try:
            async with self.session.request(method, url, **options) as response:
                if response.status != 200:
                    raise HttpError(f'Provider HTTP {response.status}', response.status)
                return await read_bounded_response(response, self.max_response_bytes)
        except HttpError:
            raise
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise HttpError('Provider transport failed') from None

    async def post_json(self, url, **kwargs):
        raw = await self.request('POST', url, **kwargs)
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError):
            raise HttpError('Provider response is invalid JSON') from None

    async def stream_request(self, method, url, **kwargs):
        self._ensure_session()
        options = self._options(url, kwargs)
        try:
            async with self.session.request(method, url, **options) as response:
                if response.status != 200:
                    raise HttpError(f'Provider HTTP {response.status}', response.status)
                if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                    raise HttpError('Provider stream encoding is unsupported', response.status)
                buffer = b''
                size = 0
                async for chunk in response.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise HttpError('Provider stream exceeds its byte limit', response.status)
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if line:
                            yield line
                if buffer.strip():
                    yield buffer
        except HttpError:
            raise
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise HttpError('Provider stream transport failed') from None
