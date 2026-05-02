"""SambaNova discovery adapter tests.

SambaNova is an OpenAI-compatible adapter with ``BaseDiscoveryAdapter``
subclass (NOT ``OpenAICompatibleAdapter`` — different constructor).

Tests:
- URL correctness: ``https://api.sambanova.ai/v1/models``
- Standard Bearer auth
- ``DiscoveryPolicy`` resilience: graceful 404→[], 5xx propagates (Error Policy B)
- Auth 401→propagates
- Sync/async equivalence
- ``discover()`` is inherited from ``BaseDiscoveryAdapter`` (override removed)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.base_discovery import (
    BaseDiscoveryAdapter,
    DiscoveryError,
)
from magic_llm.engine.discovery.sambanova_discovery import (
    SambaNovaDiscoveryAdapter,
)
from magic_llm.util.http import HttpError

SAMBA_PAYLOAD = {
    "data": [
        {"id": "Meta-Llama-3.1-8B-Instruct", "object": "model"},
        {"id": "Meta-Llama-3.1-70B-Instruct", "object": "model"},
    ],
}


# ── Defaults ──────────────────────────────────────────────────────────────

class TestSambaNovaDefaults:
    def test_default_url(self):
        adapter = SambaNovaDiscoveryAdapter(api_key="sk-test")
        assert adapter._get_endpoint_url() == "https://api.sambanova.ai/v1/models"

    def test_default_headers(self):
        adapter = SambaNovaDiscoveryAdapter(api_key="sk-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"

    def test_no_auth_when_no_key(self):
        adapter = SambaNovaDiscoveryAdapter(api_key=None)
        headers = adapter._get_headers()
        assert "Authorization" not in headers


# ── Override Removal ──────────────────────────────────────────────────────

class TestSambaNovaOverrideRemoved:
    def test_discover_is_inherited_from_base(self):
        """SambaNova no longer overrides discover() — it has been removed."""
        assert SambaNovaDiscoveryAdapter.discover is BaseDiscoveryAdapter.discover

    def test_async_discover_is_inherited_from_base(self):
        assert (
            SambaNovaDiscoveryAdapter.async_discover
            is BaseDiscoveryAdapter.async_discover
        )


# ── Normalization ─────────────────────────────────────────────────────────

class TestSambaNovaNormalization:
    def test_returns_normalized_models(self):
        adapter = SambaNovaDiscoveryAdapter(api_key="sk-test")
        result = adapter._normalize_response(SAMBA_PAYLOAD)
        assert len(result) == 2
        assert result[0].external_id == "Meta-Llama-3.1-8B-Instruct"
        assert result[0].provider == "sambanova"

    def test_empty_payload(self):
        adapter = SambaNovaDiscoveryAdapter(api_key="sk-test")
        result = adapter._normalize_response({"data": []})
        assert result == []


# ── Sync discover() ───────────────────────────────────────────────────────

class TestSambaNovaDiscover:
    @pytest.fixture
    def adapter(self):
        return SambaNovaDiscoveryAdapter(api_key="sk-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(SAMBA_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2
        assert result[0].provider == "sambanova"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.sambanova.ai/v1/models",
            headers={
                "Authorization": "Bearer sk-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


# ── Error Policy B ────────────────────────────────────────────────────────

class TestSambaNovaErrorPolicy:
    """Error Policy B: 404→[], 5xx→propagates, auth 401→propagates."""

    @pytest.fixture
    def adapter(self):
        return SambaNovaDiscoveryAdapter(api_key="sk-test")

    def test_404_returns_empty_list(self, adapter):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Not found", status_code=404, response_content=b'{}'
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            result = adapter.discover()
            assert result == []

    def test_5xx_propagates(self, adapter):
        """SambaNova no longer swallows 5xx — behavior CHANGE."""
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Server error", status_code=502, response_content=b'bad gateway'
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            with pytest.raises(DiscoveryError) as exc:
                adapter.discover()
            assert exc.value.status_code == 502

    def test_401_propagates(self, adapter):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Unauthorized", status_code=401, response_content=b'{}'
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            from magic_llm.engine.discovery.base_discovery import DiscoveryAuthError
            with pytest.raises(DiscoveryAuthError):
                adapter.discover()


# ── Async ─────────────────────────────────────────────────────────────────

class TestSambaNovaAsync:
    @pytest.fixture
    def adapter(self):
        return SambaNovaDiscoveryAdapter(api_key="sk-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(SAMBA_PAYLOAD).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 2


# ── Registry ───────────────────────────────────────────────────────────────

class TestSambaNovaRegistry:
    def test_get_adapter_returns_sambanova(self):
        cls = get_adapter("sambanova")
        assert cls is SambaNovaDiscoveryAdapter
        assert cls is not None
