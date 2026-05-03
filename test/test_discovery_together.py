"""Together AI discovery adapter tests.

OpenAI-compatible adapter — tests follow the canonical pattern:
- Default URL is ``https://api.together.xyz/v1/models``
- Auth header is ``Authorization: Bearer {key}``
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.together import (
    TogetherDiscoveryAdapter,
)

TOGETHER_PAYLOAD = {
    "data": [
        {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "object": "model"},
        {"id": "meta-llama/Llama-2-70b-chat-hf", "object": "model"},
    ]
}

TOGETHER_BARE_ARRAY = [
    {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "object": "model"},
    {"id": "meta-llama/Llama-2-70b-chat-hf", "object": "model"},
]


class TestTogetherDefaults:
    def test_default_url(self):
        adapter = TogetherDiscoveryAdapter(api_key="tk-test")
        assert adapter._get_endpoint_url() == "https://api.together.xyz/v1/models"

    def test_default_headers(self):
        adapter = TogetherDiscoveryAdapter(api_key="tk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer tk-test"


class TestTogetherBareArray:
    """Regression: Together API returns a bare JSON array, not ``{"data": [...]}``."""

    @pytest.fixture
    def adapter(self):
        return TogetherDiscoveryAdapter(api_key="tk-test")

    @pytest.fixture
    def mock_client_bare(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(TOGETHER_BARE_ARRAY).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_bare_array_does_not_crash(self, adapter, mock_client_bare):
        result = adapter.discover()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "together"


class TestTogetherDiscover:
    @pytest.fixture
    def adapter(self):
        return TogetherDiscoveryAdapter(api_key="tk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(TOGETHER_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "together"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.together.xyz/v1/models",
            headers={
                "Authorization": "Bearer tk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


class TestTogetherAsync:
    @pytest.fixture
    def adapter(self):
        return TogetherDiscoveryAdapter(api_key="tk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(TOGETHER_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 2


class TestTogetherRegistry:
    def test_get_adapter_returns_together(self):
        cls = get_adapter("together")
        assert cls is TogetherDiscoveryAdapter
        assert cls is not None
