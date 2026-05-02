"""Novita discovery adapter tests.

OpenAI-compatible adapter — tests follow the canonical pattern:
- Default URL is ``https://api.novita.ai/v3/openai/models`` (URL quirk)
- Auth header is ``Authorization: Bearer {key}``
- The URL uses ``/v3/openai/models`` — NOT the standard ``/v1/models``
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.novita import (
    NovitaDiscoveryAdapter,
)

NOVITA_PAYLOAD = {
    "data": [
        {"id": "llama-3.1-8b-instruct", "object": "model"},
        {"id": "mistral-7b-instruct", "object": "model"},
    ]
}


class TestNovitaDefaults:
    def test_default_url_uses_v3_openai_path(self):
        """Novita uses ``/v3/openai/models`` — NOT standard ``/v1/models``."""
        adapter = NovitaDiscoveryAdapter(api_key="nk-test")
        url = adapter._get_endpoint_url()
        assert url == "https://api.novita.ai/v3/openai/models"
        assert "/v3/openai/" in url
        assert "/v1/models" not in url  # regression guard

    def test_default_headers(self):
        adapter = NovitaDiscoveryAdapter(api_key="nk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer nk-test"


class TestNovitaDiscover:
    @pytest.fixture
    def adapter(self):
        return NovitaDiscoveryAdapter(api_key="nk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(NOVITA_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "novita"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.novita.ai/v3/openai/models",
            headers={
                "Authorization": "Bearer nk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


class TestNovitaAsync:
    @pytest.fixture
    def adapter(self):
        return NovitaDiscoveryAdapter(api_key="nk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(NOVITA_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 2


class TestNovitaRegistry:
    def test_get_adapter_returns_novita(self):
        cls = get_adapter("novita")
        assert cls is NovitaDiscoveryAdapter
        assert cls is not None
