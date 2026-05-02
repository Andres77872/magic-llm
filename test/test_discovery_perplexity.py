"""Perplexity discovery adapter tests.

OpenAI-compatible adapter — tests follow the canonical pattern:
- Default URL is ``https://api.perplexity.ai/v1/models``
- Auth header is ``Authorization: Bearer {key}``
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.perplexity import (
    PerplexityDiscoveryAdapter,
)

PERPLEXITY_PAYLOAD = {
    "data": [
        {"id": "sonar-pro", "object": "model"},
        {"id": "sonar-small", "object": "model"},
    ]
}


class TestPerplexityDefaults:
    def test_default_url(self):
        adapter = PerplexityDiscoveryAdapter(api_key="pk-test")
        assert adapter._get_endpoint_url() == "https://api.perplexity.ai/v1/models"

    def test_default_headers(self):
        adapter = PerplexityDiscoveryAdapter(api_key="pk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer pk-test"

    def test_no_auth_when_no_key(self):
        adapter = PerplexityDiscoveryAdapter(api_key=None)
        headers = adapter._get_headers()
        assert "Authorization" not in headers


class TestPerplexityDiscover:
    @pytest.fixture
    def adapter(self):
        return PerplexityDiscoveryAdapter(api_key="pk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(PERPLEXITY_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "perplexity"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.perplexity.ai/v1/models",
            headers={
                "Authorization": "Bearer pk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


class TestPerplexityAsync:
    @pytest.fixture
    def adapter(self):
        return PerplexityDiscoveryAdapter(api_key="pk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(PERPLEXITY_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 2


class TestPerplexityRegistry:
    def test_get_adapter_returns_perplexity(self):
        cls = get_adapter("perplexity")
        assert cls is PerplexityDiscoveryAdapter
        assert cls is not None
