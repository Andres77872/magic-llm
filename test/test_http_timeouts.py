"""Tests for HTTP client timeout defaults."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from magic_llm.util.http import AsyncHttpClient, HttpClient


class TestAsyncHttpClientTimeouts:
    """Test AsyncHttpClient timeout behavior."""

    @pytest.mark.asyncio
    async def test_default_timeout_applied_when_none_provided(self):
        """AsyncHttpClient.request() uses total=30 when no timeout kwarg."""
        client = AsyncHttpClient()
        client.session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"ok")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        client.session.request = MagicMock(return_value=mock_response)

        await client.request("GET", "http://example.com")

        call_kwargs = client.session.request.call_args[1]
        timeout = call_kwargs["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == 30

    @pytest.mark.asyncio
    async def test_explicit_timeout_override(self):
        """AsyncHttpClient.request() respects explicit timeout=120."""
        client = AsyncHttpClient()
        client.session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"ok")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        client.session.request = MagicMock(return_value=mock_response)

        await client.request("GET", "http://example.com", timeout=120)

        call_kwargs = client.session.request.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.total == 120

    @pytest.mark.asyncio
    async def test_explicit_none_timeout_respected(self):
        """AsyncHttpClient.request() respects timeout=None (no timeout)."""
        client = AsyncHttpClient()
        client.session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"ok")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        client.session.request = MagicMock(return_value=mock_response)

        await client.request("GET", "http://example.com", timeout=None)

        call_kwargs = client.session.request.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.total is None

    @pytest.mark.asyncio
    async def test_stream_request_default_timeout(self):
        """AsyncHttpClient.stream_request() uses total=30 default."""
        client = AsyncHttpClient()
        client.session = MagicMock()

        async def mock_stream():
            yield b"chunk1"
            yield b"chunk2"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        client.session.request = MagicMock(return_value=mock_response)

        chunks = []
        async for chunk in client.stream_request("POST", "http://example.com"):
            chunks.append(chunk)

        call_kwargs = client.session.request.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.total == 30

    @pytest.mark.asyncio
    async def test_stream_request_explicit_timeout(self):
        """AsyncHttpClient.stream_request() respects explicit timeout."""
        client = AsyncHttpClient()
        client.session = MagicMock()

        async def mock_stream():
            yield b"chunk"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        client.session.request = MagicMock(return_value=mock_response)

        async for _ in client.stream_request("POST", "http://example.com", timeout=60):
            pass

        call_kwargs = client.session.request.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.total == 60


class TestSyncHttpClientTimeouts:
    """Test HttpClient timeout behavior."""

    def test_default_timeout_applied_when_none_provided(self):
        """HttpClient.request() uses timeout=30 when no timeout kwarg."""
        client = HttpClient()
        client.session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"ok"
        client.session.request = MagicMock(return_value=mock_response)

        client.request("GET", "http://example.com")

        call_kwargs = client.session.request.call_args[1]
        assert call_kwargs["timeout"] == 30

    def test_explicit_timeout_override(self):
        """HttpClient.request() respects explicit timeout=120."""
        client = HttpClient()
        client.session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"ok"
        client.session.request = MagicMock(return_value=mock_response)

        client.request("GET", "http://example.com", timeout=120)

        call_kwargs = client.session.request.call_args[1]
        assert call_kwargs["timeout"] == 120

    def test_explicit_none_timeout_respected(self):
        """HttpClient.request() respects timeout=None."""
        client = HttpClient()
        client.session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"ok"
        client.session.request = MagicMock(return_value=mock_response)

        client.request("GET", "http://example.com", timeout=None)

        call_kwargs = client.session.request.call_args[1]
        assert call_kwargs["timeout"] is None
