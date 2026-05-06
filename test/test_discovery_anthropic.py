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
            "max_input_tokens": 200000,
            "max_tokens": 8192,
            "capabilities": {
                "image_input": {"supported": True},
                "thinking": {"supported": False},
            },
        },
        {
            "id": "claude-3-haiku-20240307",
            "display_name": "Claude 3 Haiku",
            "max_input_tokens": 200000,
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
        assert sonnet.context_window == 200000  # Claude-name heuristic wins
        assert sonnet.max_input_tokens == 200000
        assert sonnet.max_output_tokens == 8192


    def test_unified_fallback_for_unknown_claude(self):
        """Non-Claude-3/2 model with max_input_tokens → context_window from fallback.

        Models like claude-opus-4-* or claude-sonnet-4-* don't match any heuristic,
        so context_window falls back to max_input_tokens per the unified contract.
        """
        adapter = AnthropicDiscoveryAdapter(api_key="ak-test")
        payload = {
            "data": [
                {
                    "id": "claude-opus-4-20250514",
                    "display_name": "Claude Opus 4",
                    "max_input_tokens": 1000000,
                    "max_tokens": 8192,
                    "capabilities": {"image_input": {"supported": True}},
                },
                {
                    "id": "claude-sonnet-4-20250514",
                    "display_name": "Claude Sonnet 4",
                    "max_input_tokens": 1000000,
                    "max_tokens": 16384,
                    "capabilities": {"image_input": {"supported": True}},
                },
            ],
        }
        result = adapter._normalize_response(payload)
        assert len(result) == 2
        opus = result[0]
        sonnet4 = result[1]
        # No heuristic match for claude-opus-4 or claude-sonnet-4 → fallback to max_input_tokens
        assert opus.context_window == 1000000
        assert opus.max_input_tokens == 1000000
        assert opus.max_output_tokens == 8192
        assert sonnet4.context_window == 1000000
        assert sonnet4.max_input_tokens == 1000000
        assert sonnet4.max_output_tokens == 16384

    def test_no_defaults_when_fields_absent(self):
        """When max_input_tokens and max_tokens are absent, all three fields stay None."""
        adapter = AnthropicDiscoveryAdapter(api_key="ak-test")
        payload = {
            "data": [
                {
                    "id": "some-unknown-model",
                    "display_name": "Unknown Model",
                },
            ],
        }
        result = adapter._normalize_response(payload)
        assert len(result) == 1
        mdl = result[0]
        assert mdl.context_window is None
        assert mdl.max_input_tokens is None
        assert mdl.max_output_tokens is None


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
