"""Cerebras discovery adapter tests.

OpenAI-compatible adapter — tests follow the canonical pattern:
- Default URL is ``https://api.cerebras.ai/v1/models``
- Auth header is ``Authorization: Bearer {key}``
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.cerebras import (
    CerebrasDiscoveryAdapter,
)

CEREBRAS_PAYLOAD = {
    "data": [
        {"id": "cerebras/Llama-3.3-70B", "object": "model"},
    ]
}


class TestCerebrasDefaults:
    def test_default_url(self):
        adapter = CerebrasDiscoveryAdapter(api_key="ck-test")
        assert adapter._get_endpoint_url() == "https://api.cerebras.ai/v1/models"

    def test_default_headers(self):
        adapter = CerebrasDiscoveryAdapter(api_key="ck-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer ck-test"


class TestCerebrasDiscover:
    @pytest.fixture
    def adapter(self):
        return CerebrasDiscoveryAdapter(api_key="ck-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(CEREBRAS_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 1
        assert result[0].provider == "cerebras"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.cerebras.ai/v1/models",
            headers={
                "Authorization": "Bearer ck-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


class TestCerebrasAsync:
    @pytest.fixture
    def adapter(self):
        return CerebrasDiscoveryAdapter(api_key="ck-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(CEREBRAS_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 1


class TestCerebrasRegistry:
    def test_get_adapter_returns_cerebras(self):
        cls = get_adapter("cerebras")
        assert cls is CerebrasDiscoveryAdapter
        assert cls is not None
