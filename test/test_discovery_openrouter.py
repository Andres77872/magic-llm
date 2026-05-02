"""OpenRouter discovery adapter tests.

OpenRouter's models endpoint is public — no auth required.  This file
tests the no-auth-variant adapter and validates pricing extraction.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openrouter_discovery import (
    OpenRouterDiscoveryAdapter,
)

OPENROUTER_PAYLOAD = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "description": "OpenAI GPT-4o",
            "context_length": 128000,
            "pricing": {
                "prompt": "0.0000025",
                "completion": "0.00001",
            },
            "architecture": {
                "modality": {"input": ["text", "image"], "output": ["text"]},
                "tokenizer": "GPT",
            },
        },
        {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "description": "Anthropic Claude 3.5 Sonnet",
            "context_length": 200000,
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
            },
            "architecture": {
                "modality": {"input": ["text"], "output": ["text"]},
                "tokenizer": "Claude",
            },
        },
    ]
}


class TestOpenRouterDefaults:
    def test_default_url(self):
        adapter = OpenRouterDiscoveryAdapter(api_key=None)
        assert adapter._get_endpoint_url() == "https://openrouter.ai/api/v1/models"

    def test_no_auth_headers(self):
        """OpenRouter public listing — no auth required."""
        adapter = OpenRouterDiscoveryAdapter(api_key=None)
        headers = adapter._get_headers()
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/json"


class TestOpenRouterNormalization:
    def test_pricing_extraction(self):
        adapter = OpenRouterDiscoveryAdapter(api_key=None)
        result = adapter._normalize_response(OPENROUTER_PAYLOAD)
        assert len(result) == 2
        gpt4o = result[0]
        assert gpt4o.external_id == "openai/gpt-4o"
        # Pricing: prompt=0.0000025 → 2.5 per million, completion=0.00001 → 10 per million
        assert gpt4o.pricing is not None
        assert gpt4o.pricing.input_per_million == 2.5
        assert gpt4o.pricing.output_per_million == 10.0
        assert gpt4o.context_window == 128000
        assert gpt4o.capabilities.vision is True

    def test_no_auth_discover(self):
        adapter = OpenRouterDiscoveryAdapter(api_key=None)
        result = adapter._normalize_response(OPENROUTER_PAYLOAD)
        assert len(result) == 2


class TestOpenRouterDiscover:
    @pytest.fixture
    def adapter(self):
        return OpenRouterDiscoveryAdapter(api_key=None)

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(OPENROUTER_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_public_endpoint_no_auth(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once()
        args, kwargs = mock_client.return_value.request.call_args
        assert args[1] == "https://openrouter.ai/api/v1/models"
        # Verify no Authorization header
        assert "Authorization" not in kwargs["headers"]


class TestOpenRouterRegistry:
    def test_get_adapter_returns_openrouter(self):
        cls = get_adapter("openrouter")
        assert cls is OpenRouterDiscoveryAdapter
