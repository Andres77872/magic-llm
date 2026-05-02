"""Groq discovery adapter tests.

OpenAI-compatible adapter — tests follow the canonical pattern:
- Default URL is ``https://api.groq.com/openai/v1/models`` (URL quirk)
- Auth header is ``Authorization: Bearer {key}``
- Groq uses ``/openai/v1/models`` (NOT standard ``/v1/models``)
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.groq import (
    GroqDiscoveryAdapter,
)

GROQ_PAYLOAD = {
    "data": [
        {"id": "llama-3.3-70b-versatile", "object": "model"},
        {"id": "mixtral-8x7b-32768", "object": "model"},
    ]
}


class TestGroqDefaults:
    def test_default_url_uses_openai_v1_path(self):
        """Groq uses ``/openai/v1/models`` — NOT plain ``/v1/models``."""
        adapter = GroqDiscoveryAdapter(api_key="gk-test")
        url = adapter._get_endpoint_url()
        assert url == "https://api.groq.com/openai/v1/models"
        assert "/openai/v1/" in url

    def test_default_headers(self):
        adapter = GroqDiscoveryAdapter(api_key="gk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer gk-test"


class TestGroqDiscover:
    @pytest.fixture
    def adapter(self):
        return GroqDiscoveryAdapter(api_key="gk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(GROQ_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "groq"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": "Bearer gk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


class TestGroqAsync:
    @pytest.fixture
    def adapter(self):
        return GroqDiscoveryAdapter(api_key="gk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(GROQ_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 2


class TestGroqRegistry:
    def test_get_adapter_returns_groq(self):
        cls = get_adapter("groq")
        assert cls is GroqDiscoveryAdapter
        assert cls is not None
