"""Google AI Studio discovery adapter tests.

Google uses a different endpoint shape (``/v1beta/models``) and auth header
(``x-goog-api-key``) — distinct from the OpenAI-compatible pattern.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.google_discovery import (
    GoogleDiscoveryAdapter,
)

GOOGLE_PAYLOAD = {
    "models": [
        {
            "name": "models/gemini-1.5-pro",
            "displayName": "Gemini 1.5 Pro",
            "description": "Best for complex tasks",
            "inputTokenLimit": 128000,
            "outputTokenLimit": 8192,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens",
            ],
        },
        {
            "name": "models/gemini-1.5-flash",
            "displayName": "Gemini 1.5 Flash",
            "description": "Best for fast tasks",
            "inputTokenLimit": 128000,
            "outputTokenLimit": 8192,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens",
            ],
        },
        {
            "name": "models/text-embedding-004",
            "displayName": "Text Embedding 004",
            "description": "Embedding model",
            "inputTokenLimit": 2048,
            "supportedGenerationMethods": [
                "embedContent",
            ],
        },
    ]
}


class TestGoogleDefaults:
    def test_default_url(self):
        adapter = GoogleDiscoveryAdapter(api_key="gk-test")
        assert adapter._get_endpoint_url() == "https://generativelanguage.googleapis.com/v1beta/models"

    def test_custom_auth_header(self):
        adapter = GoogleDiscoveryAdapter(api_key="gk-test")
        headers = adapter._get_headers()
        assert headers["x-goog-api-key"] == "gk-test"
        assert "Authorization" not in headers  # NOT Bearer

    def test_user_url_override(self):
        adapter = GoogleDiscoveryAdapter(
            api_key="gk-test",
            base_url="https://google-proxy.example.com",
        )
        assert adapter._get_endpoint_url() == "https://google-proxy.example.com/v1beta/models"


class TestGoogleNormalization:
    def test_normalized_models_have_correct_fields(self):
        adapter = GoogleDiscoveryAdapter(api_key="gk-test")
        result = adapter._normalize_response(GOOGLE_PAYLOAD)
        assert len(result) == 3
        pro = result[0]
        assert pro.external_id == "gemini-1.5-pro"
        assert pro.display_name == "Gemini 1.5 Pro"
        assert pro.capabilities.chat is True
        assert pro.context_window == 128000
        assert pro.max_output_tokens == 8192

    def test_embedding_model(self):
        adapter = GoogleDiscoveryAdapter(api_key="gk-test")
        result = adapter._normalize_response(GOOGLE_PAYLOAD)
        embedding = result[2]
        assert embedding.capabilities.chat is False
        assert embedding.capabilities.embedding is True


class TestGoogleDiscover:
    @pytest.fixture
    def adapter(self):
        return GoogleDiscoveryAdapter(api_key="gk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(GOOGLE_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_url_and_headers(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-goog-api-key": "gk-test",
            },
        )


class TestGoogleRegistry:
    def test_get_adapter_returns_google(self):
        cls = get_adapter("google")
        assert cls is GoogleDiscoveryAdapter
