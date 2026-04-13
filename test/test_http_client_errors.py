"""Tests for HTTP client error paths (magic_llm.util.http).

Slice 17: HttpClient (sync) error paths
Slice 18: AsyncHttpClient (async) error paths
"""

import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from requests import RequestException

from magic_llm.util.http import HttpError, HttpClient, AsyncHttpClient


# ─── Slice 17: HttpClient sync error paths ─────────────────────────────────

class TestHttpClientSyncErrors:
    """HttpClient error wrapping — sync client."""

    def test_non_200_response_raises_http_error(self):
        """Non-200 response → HttpError with status_code and response_content."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b'{"error": "not found"}'

        mock_session = MagicMock()
        mock_session.request.return_value = mock_response

        client = HttpClient()
        client.session = mock_session

        with pytest.raises(HttpError) as exc_info:
            client.request("GET", "https://example.com/api")

        assert exc_info.value.status_code == 404
        assert exc_info.value.response_content == b'{"error": "not found"}'
        assert "404" in str(exc_info.value)

    def test_post_json_invalid_json_propagates_decode_error(self):
        """Invalid JSON from post_json → json.JSONDecodeError propagated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'not valid json at all'

        mock_session = MagicMock()
        mock_session.request.return_value = mock_response

        client = HttpClient()
        client.session = mock_session

        with pytest.raises(json.JSONDecodeError):
            client.post_json("https://example.com/api")

    def test_request_exception_with_response_wrapped(self):
        """RequestException with response → HttpError with status/content."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.content = b'bad gateway'

        exc = RequestException("connection issue")
        exc.response = mock_response

        mock_session = MagicMock()
        mock_session.request.side_effect = exc

        client = HttpClient()
        client.session = mock_session

        with pytest.raises(HttpError) as exc_info:
            client.request("POST", "https://example.com/api")

        assert exc_info.value.status_code == 502
        assert exc_info.value.response_content == b'bad gateway'
        assert "connection issue" in str(exc_info.value)

    def test_request_exception_without_response_message_only(self):
        """RequestException without response → HttpError with message only."""
        exc = RequestException("network unreachable")

        mock_session = MagicMock()
        mock_session.request.side_effect = exc

        client = HttpClient()
        client.session = mock_session

        with pytest.raises(HttpError) as exc_info:
            client.request("GET", "https://example.com/api")

        assert exc_info.value.status_code is None
        assert exc_info.value.response_content is None
        assert "network unreachable" in str(exc_info.value)

    def test_successful_request_returns_content(self):
        """200 response → returns content bytes."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"ok": true}'

        mock_session = MagicMock()
        mock_session.request.return_value = mock_response

        client = HttpClient()
        client.session = mock_session

        result = client.request("GET", "https://example.com/api")
        assert result == b'{"ok": true}'

    def test_post_json_success(self):
        """post_json with valid JSON → parsed dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"key": "value"}'

        mock_session = MagicMock()
        mock_session.request.return_value = mock_response

        client = HttpClient()
        client.session = mock_session

        result = client.post_json("https://example.com/api")
        assert result == {"key": "value"}

    def test_session_not_initialized_raises(self):
        """Using client without session → RuntimeError."""
        client = HttpClient()
        with pytest.raises(RuntimeError, match="Session not initialized"):
            client.request("GET", "https://example.com")


# ─── Slice 18: AsyncHttpClient async error paths ───────────────────────────


class _AsyncCtxManager:
    """Helper to make a mock work as an async context manager."""
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class TestAsyncHttpClientErrors:
    """AsyncHttpClient error wrapping — async client."""

    @pytest.mark.asyncio
    async def test_non_200_response_raises_http_error(self):
        """Non-200 response → HttpError with status_code and response_content."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.read = AsyncMock(return_value=b'{"error": "server error"}')

        mock_session = MagicMock()
        mock_session.request.return_value = _AsyncCtxManager(mock_response)

        client = AsyncHttpClient()
        client.session = mock_session

        with pytest.raises(HttpError) as exc_info:
            await client.request("POST", "https://example.com/api")

        assert exc_info.value.status_code == 500
        assert exc_info.value.response_content == b'{"error": "server error"}'
        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_post_json_invalid_json_propagates_decode_error(self):
        """Invalid JSON from post_json → json.JSONDecodeError propagated."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b'not json')

        mock_session = MagicMock()
        mock_session.request.return_value = _AsyncCtxManager(mock_response)

        client = AsyncHttpClient()
        client.session = mock_session

        with pytest.raises(json.JSONDecodeError):
            await client.post_json("https://example.com/api")

    @pytest.mark.asyncio
    async def test_aiohttp_client_error_wrapped(self):
        """aiohttp.ClientError → HttpError with message only."""
        import aiohttp

        mock_session = MagicMock()
        mock_session.request.side_effect = aiohttp.ClientError("connection refused")

        client = AsyncHttpClient()
        client.session = mock_session

        with pytest.raises(HttpError) as exc_info:
            await client.request("GET", "https://example.com/api")

        assert exc_info.value.status_code is None
        assert exc_info.value.response_content is None
        assert "connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_successful_request_returns_content(self):
        """200 response → returns content bytes."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b'{"ok": true}')

        mock_session = MagicMock()
        mock_session.request.return_value = _AsyncCtxManager(mock_response)

        client = AsyncHttpClient()
        client.session = mock_session

        result = await client.request("GET", "https://example.com/api")
        assert result == b'{"ok": true}'

    @pytest.mark.asyncio
    async def test_post_json_success(self):
        """post_json with valid JSON → parsed dict."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b'{"key": "value"}')

        mock_session = MagicMock()
        mock_session.request.return_value = _AsyncCtxManager(mock_response)

        client = AsyncHttpClient()
        client.session = mock_session

        result = await client.post_json("https://example.com/api")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_session_not_initialized_raises(self):
        """Using client without session → RuntimeError."""
        client = AsyncHttpClient()
        with pytest.raises(RuntimeError, match="Session not initialized"):
            await client.request("GET", "https://example.com")
