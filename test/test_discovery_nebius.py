"""Nebius AI Studio discovery adapter tests.

OpenAI-compatible adapter — tests follow the canonical pattern:
- Default URL is ``https://api.studio.nebius.ai/v1/models`` (subdomain quirk)
- Auth header is ``Authorization: Bearer {key}``
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.nebius import (
    NebiusDiscoveryAdapter,
)

NEBIUS_PAYLOAD = {
    "data": [
        {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "object": "model"},
    ]
}


class TestNebiusDefaults:
    def test_default_url_has_studio_subdomain(self):
        """Nebius uses ``api.studio.nebius.ai`` (not plain ``api.nebius.ai``)."""
        adapter = NebiusDiscoveryAdapter(api_key="nk-test")
        url = adapter._get_endpoint_url()
        assert url == "https://api.studio.nebius.ai/v1/models"
        assert "studio" in url  # subdomain quirk

    def test_default_headers(self):
        adapter = NebiusDiscoveryAdapter(api_key="nk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer nk-test"


class TestNebiusDiscover:
    @pytest.fixture
    def adapter(self):
        return NebiusDiscoveryAdapter(api_key="nk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(NEBIUS_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 1
        assert result[0].provider == "nebius"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.studio.nebius.ai/v1/models",
            headers={
                "Authorization": "Bearer nk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


class TestNebiusAsync:
    @pytest.fixture
    def adapter(self):
        return NebiusDiscoveryAdapter(api_key="nk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(NEBIUS_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 1


class TestNebiusRegistry:
    def test_get_adapter_returns_nebius(self):
        cls = get_adapter("nebius")
        assert cls is NebiusDiscoveryAdapter
        assert cls is not None
