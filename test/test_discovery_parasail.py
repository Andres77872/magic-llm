"""Parasail discovery adapter tests.

OpenAI-compatible adapter — tests follow the canonical pattern:
- Default URL is ``https://api.parasail.io/v1/models``
- Auth header is ``Authorization: Bearer {key}``
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.parasail import (
    ParasailDiscoveryAdapter,
)

PARASAIL_PAYLOAD = {
    "data": [
        {"id": "parasail/Meta-Llama-3.1-8B-Instruct", "object": "model"},
    ]
}


class TestParasailDefaults:
    def test_default_url(self):
        adapter = ParasailDiscoveryAdapter(api_key="pk-test")
        assert adapter._get_endpoint_url() == "https://api.parasail.io/v1/models"

    def test_default_headers(self):
        adapter = ParasailDiscoveryAdapter(api_key="pk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer pk-test"


class TestParasailDiscover:
    @pytest.fixture
    def adapter(self):
        return ParasailDiscoveryAdapter(api_key="pk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(PARASAIL_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 1
        assert result[0].provider == "parasail"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.parasail.io/v1/models",
            headers={
                "Authorization": "Bearer pk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


class TestParasailAsync:
    @pytest.fixture
    def adapter(self):
        return ParasailDiscoveryAdapter(api_key="pk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(PARASAIL_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 1


class TestParasailRegistry:
    def test_get_adapter_returns_parasail(self):
        cls = get_adapter("parasail")
        assert cls is ParasailDiscoveryAdapter
        assert cls is not None
