"""Anthropic discovery adapter tests.

Anthropic uses custom auth headers (``x-api-key``, ``anthropic-version``)
instead of Bearer.  This test file validates the non-OpenAI-compatible path.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.anthropic_discovery import (
    AnthropicDiscoveryAdapter,
)

ANTHROPIC_PAYLOAD = {
    "data": [
        {
            "id": "claude-3-5-sonnet-20241022",
            "display_name": "Claude 3.5 Sonnet",
            "max_tokens": 8192,
            "capabilities": {
                "image_input": {"supported": True},
                "thinking": {"supported": False},
            },
        },
        {
            "id": "claude-3-haiku-20240307",
            "display_name": "Claude 3 Haiku",
            "max_tokens": 4096,
            "capabilities": {
                "image_input": {"supported": True},
                "thinking": {"supported": False},
            },
        },
    ]
}


class TestAnthropicDefaults:
    def test_default_url(self):
        adapter = AnthropicDiscoveryAdapter(api_key="ak-test")
        assert adapter._get_endpoint_url() == "https://api.anthropic.com/v1/models"

    def test_custom_auth_headers(self):
        adapter = AnthropicDiscoveryAdapter(api_key="ak-test")
        headers = adapter._get_headers()
        assert headers["x-api-key"] == "ak-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in headers  # NOT Bearer

    def test_no_auth_when_no_key(self):
        adapter = AnthropicDiscoveryAdapter(api_key=None)
        headers = adapter._get_headers()
        assert "x-api-key" not in headers
        assert "anthropic-version" in headers  # version header always sent


class TestAnthropicNormalization:
    def test_capabilities_mapped_correctly(self):
        adapter = AnthropicDiscoveryAdapter(api_key="ak-test")
        result = adapter._normalize_response(ANTHROPIC_PAYLOAD)
        assert len(result) == 2
        sonnet = result[0]
        assert sonnet.external_id == "claude-3-5-sonnet-20241022"
        assert sonnet.display_name == "Claude 3.5 Sonnet"
        assert sonnet.capabilities.vision is True  # image_input.supported
        assert sonnet.capabilities.reasoning is False  # thinking.supported
        assert sonnet.capabilities.chat is True
        assert sonnet.max_output_tokens == 8192


class TestAnthropicDiscover:
    @pytest.fixture
    def adapter(self):
        return AnthropicDiscoveryAdapter(api_key="ak-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(ANTHROPIC_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_hits_correct_url_and_headers(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.anthropic.com/v1/models",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": "ak-test",
            },
        )

    def test_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2


class TestAnthropicRegistry:
    def test_get_adapter_returns_anthropic(self):
        cls = get_adapter("anthropic")
        assert cls is AnthropicDiscoveryAdapter
