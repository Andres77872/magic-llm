"""DeepInfra discovery regression tests.

Verifies:
- Default URL is ``https://api.deepinfra.com/v1/models`` (NOT
  ``…/v1/openai/v1/models`` — the double-``/v1`` bug)
- Auth header is ``Authorization: Bearer {key}``
- Sync and async produce identical results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_compatible.deepinfra import (
    DeepInfraDiscoveryAdapter,
)

DEEPINFRA_PAYLOAD = {
    "data": [
        {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "object": "model"},
        {"id": "meta-llama/Llama-2-70b-chat-hf", "object": "model"},
    ]
}


class TestDeepInfraDefaults:
    """URL composition — the critical regression."""

    def test_default_url_is_correct(self):
        """Default URL must NOT contain /v1/openai — the old double-/v1 bug."""
        adapter = DeepInfraDiscoveryAdapter(api_key="dk-test")
        url = adapter._get_endpoint_url()
        assert url == "https://api.deepinfra.com/v1/models"
        # Regression assertions
        assert "/v1/openai/" not in url
        assert "/openai/v1/" not in url

    def test_default_headers(self):
        adapter = DeepInfraDiscoveryAdapter(api_key="dk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer dk-test"

    def test_user_url_override(self):
        """Explicit base_url is honored (proxy scenarios)."""
        adapter = DeepInfraDiscoveryAdapter(
            api_key="dk-test",
            base_url="https://proxy.internal.example.com/dpi/v1/models",
        )
        url = adapter._get_endpoint_url()
        assert url == "https://proxy.internal.example.com/dpi/v1/models"


class TestDeepInfraDiscover:
    """Mocked HTTP discover test — asserts exact request URL."""

    @pytest.fixture
    def adapter(self):
        return DeepInfraDiscoveryAdapter(api_key="dk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(DEEPINFRA_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_request_url_is_exact(self, adapter, mock_client):
        """Assert the EXACT URL hit — regression for double-/v1 bug."""
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.deepinfra.com/v1/models",
            headers={
                "Authorization": "Bearer dk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    def test_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "deepinfra"


class TestDeepInfraAsync:
    """Async variant with aioresponses-style mock."""

    @pytest.fixture
    def adapter(self):
        return DeepInfraDiscoveryAdapter(api_key="dk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(DEEPINFRA_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 2


class TestDeepInfraRegistry:
    def test_get_adapter_returns_deepinfra(self):
        cls = get_adapter("deepinfra")
        assert cls is DeepInfraDiscoveryAdapter
        assert cls is not None
