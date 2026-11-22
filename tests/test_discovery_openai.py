"""Tests for OpenAI Discovery Adapter (canonical shape).

Tests:
- ``OpenAIDiscoveryAdapter`` resolves the correct default URL
- Sync ``discover()`` returns ``List[NormalizedDiscoveredModel]``
- Async ``async_discover()`` returns identical results
- Auth header is ``Authorization: Bearer {key}``
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.openai_discovery import OpenAIDiscoveryAdapter


# ── Fixtures ───────────────────────────────────────────────────────────────

OPENAI_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai",
        },
        {
            "id": "gpt-4-turbo",
            "object": "model",
            "created": 1700000001,
            "owned_by": "openai",
        },
        {
            "id": "text-embedding-3-small",
            "object": "model",
            "created": 1700000002,
            "owned_by": "openai",
        },
    ],
}


@pytest.fixture
def adapter():
    return OpenAIDiscoveryAdapter(api_key="sk-test")


# ── URL / Header Tests (no HTTP) ──────────────────────────────────────────

class TestOpenAIDefaults:
    """Verify adapter owns its configuration — no external URL table."""

    def test_default_base_url(self):
        """Default base_url is the canonical OpenAI discovery endpoint."""
        adapter = OpenAIDiscoveryAdapter(api_key="sk-test")
        assert adapter._get_endpoint_url() == "https://api.openai.com/v1/models"

    def test_default_headers(self):
        """Auth header is Bearer with the provided api_key."""
        adapter = OpenAIDiscoveryAdapter(api_key="sk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    def test_no_auth_when_no_key(self):
        """No Authorization header when api_key is None."""
        adapter = OpenAIDiscoveryAdapter(api_key=None)
        headers = adapter._get_headers()
        assert "Authorization" not in headers

    def test_user_base_url_override(self):
        """Explicit base_url is honored verbatim."""
        adapter = OpenAIDiscoveryAdapter(
            api_key="sk-test",
            base_url="https://proxy.example.com/v1/models",
        )
        assert adapter._get_endpoint_url() == "https://proxy.example.com/v1/models"


# ── Normalization Tests (no HTTP) ─────────────────────────────────────────

class TestOpenAINormalization:
    """Verify _normalize_response produces correct NormalizedDiscoveredModel."""

    def test_returns_list_of_normalized_models(self, adapter):
        result = adapter._normalize_response(OPENAI_PAYLOAD)
        assert len(result) == 3
        for m in result:
            assert m.external_id
            assert m.provider == "openai"
            assert isinstance(m.raw, dict)

    def test_gpt4o_has_vision(self, adapter):
        result = adapter._normalize_response(OPENAI_PAYLOAD)
        gpt4o = next(m for m in result if m.external_id == "gpt-4o")
        assert gpt4o.capabilities.vision is True
        assert gpt4o.capabilities.chat is True

    def test_embedding_model_not_chat(self, adapter):
        result = adapter._normalize_response(OPENAI_PAYLOAD)
        embedding = next(m for m in result if m.external_id == "text-embedding-3-small")
        assert embedding.capabilities.chat is False
        assert embedding.capabilities.embedding is True


# ── Sync discover() Mocked Tests ──────────────────────────────────────────

class TestOpenAIDiscover:
    """Mocked HTTP tests for the full discover() flow."""

    @pytest.fixture
    def mock_client(self):
        """Patch HttpClient to return a controlled response."""
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(OPENAI_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 3
        for m in result:
            assert m.external_id
            assert m.provider == "openai"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.openai.com/v1/models",
            headers={
                "Authorization": "Bearer sk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


# ── Async discover() Mocked Tests ─────────────────────────────────────────

class TestOpenAIAsyncDiscover:
    """Mocked HTTP tests for async_discover()."""

    @pytest.fixture
    def mock_async_client(self):
        """Patch AsyncHttpClient to return a controlled response."""
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(OPENAI_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover_returns_normalized_models(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 3
        for m in result:
            assert m.external_id
            assert m.provider == "openai"

    @pytest.mark.asyncio
    async def test_async_discover_same_result_as_sync(self, adapter, mock_async_client):
        """Sync and async return identical normalized output for same payload."""
        result = await adapter.async_discover()
        # Sync with mocked client
        with patch("magic_llm.engine.discovery.base_discovery.HttpClient") as sync_mock:
            sync_instance = MagicMock()
            sync_instance.request.return_value = json.dumps(OPENAI_PAYLOAD).encode("utf-8")
            sync_instance.__enter__.return_value = sync_instance
            sync_mock.return_value = sync_instance

            sync_result = adapter.discover()

        assert len(sync_result) == len(result)
        for s, r in zip(sync_result, result):
            assert s.external_id == r.external_id
            assert s.provider == r.provider
            assert s.capabilities == r.capabilities
            assert s.context_window == r.context_window
            assert s.max_input_tokens == r.max_input_tokens
            assert s.max_output_tokens == r.max_output_tokens
            assert s.pricing == r.pricing


# ── Registry Access Test ──────────────────────────────────────────────────

class TestOpenAIRegistry:
    """Verify get_adapter resolves the correct class."""

    def test_get_adapter_returns_openai(self):
        cls = get_adapter("openai")
        assert cls is OpenAIDiscoveryAdapter
        assert cls is not None
