"""Cohere discovery adapter tests.

Non-OpenAI-compatible adapter — Cohere uses custom response formats:
- ``models`` key, raw list, or ``data`` key response shapes
- ``name`` as primary model identifier (then ``id``)
- Capabilities inferred from ``endpoints[]``/``features[]`` arrays
- Bearer auth via ``Authorization`` header
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import get_adapter
from magic_llm.engine.discovery.cohere_discovery import CohereDiscoveryAdapter

# ── Fixtures ───────────────────────────────────────────────────────────────

COHERE_PAYLOAD_MODELS_KEY = {
    "models": [
        {
            "id": "command-r",
            "name": "Command R",
            "endpoints": ["chat"],
            "features": ["tool_use"],
        },
        {
            "id": "embed-english-v3.0",
            "name": "Embed English",
            "endpoints": ["embed"],
            "features": [],
        },
    ],
}

COHERE_PAYLOAD_DATA_KEY = {
    "data": [
        {
            "id": "command-r",
            "name": "Command R",
            "endpoints": ["chat"],
            "features": ["tool_use"],
        },
    ],
}

COHERE_PAYLOAD_RAW_LIST = [
    {
        "id": "command-r",
        "name": "Command R",
        "endpoints": ["chat"],
        "features": ["tool_use"],
    },
]


# ── Defaults ──────────────────────────────────────────────────────────────

class TestCohereDefaults:
    def test_default_url(self):
        adapter = CohereDiscoveryAdapter(api_key="ck-test")
        assert adapter._get_endpoint_url() == "https://api.cohere.com/v1/models"

    def test_default_headers(self):
        adapter = CohereDiscoveryAdapter(api_key="ck-test")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer ck-test"

    def test_no_auth_when_no_key(self):
        adapter = CohereDiscoveryAdapter(api_key=None)
        headers = adapter._get_headers()
        assert "Authorization" not in headers


# ── Response Format Handling ─────────────────────────────────────────────

class TestCohereResponseFormats:
    def test_models_key_format(self):
        adapter = CohereDiscoveryAdapter(api_key="ck-test")
        result = adapter._normalize_response(COHERE_PAYLOAD_MODELS_KEY)
        assert len(result) == 2
        assert result[0].external_id == "Command R"
        assert result[0].capabilities.chat is True
        assert result[0].capabilities.function_calling is True
        assert result[1].capabilities.embedding is True
        assert result[1].capabilities.chat is False  # Cohere defaults

    def test_data_key_format(self):
        adapter = CohereDiscoveryAdapter(api_key="ck-test")
        result = adapter._normalize_response(COHERE_PAYLOAD_DATA_KEY)
        assert len(result) == 1
        assert result[0].external_id == "Command R"

    def test_raw_list_format(self):
        adapter = CohereDiscoveryAdapter(api_key="ck-test")
        result = adapter._normalize_response(COHERE_PAYLOAD_RAW_LIST)
        assert len(result) == 1
        assert result[0].external_id == "Command R"


# ── Sync discover() ───────────────────────────────────────────────────────

class TestCohereDiscover:
    @pytest.fixture
    def adapter(self):
        return CohereDiscoveryAdapter(api_key="ck-test")

    @pytest.fixture
    def mock_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(
                COHERE_PAYLOAD_MODELS_KEY
            ).encode("utf-8")
            instance.__enter__.return_value = instance
            mock.return_value = instance
            yield mock

    def test_discover_returns_normalized_models(self, adapter, mock_client):
        result = adapter.discover()
        assert len(result) == 2
        assert result[0].provider == "cohere"

    def test_discover_hits_correct_url(self, adapter, mock_client):
        adapter.discover()
        mock_client.return_value.request.assert_called_once_with(
            "GET",
            "https://api.cohere.com/v1/models",
            headers={
                "Authorization": "Bearer ck-test",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )


# ── Async discover() ──────────────────────────────────────────────────────

class TestCohereAsync:
    @pytest.fixture
    def adapter(self):
        return CohereDiscoveryAdapter(api_key="ck-test")

    @pytest.fixture
    def mock_async_client(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock:
            instance = MagicMock()
            instance.request = AsyncMock(
                return_value=json.dumps(COHERE_PAYLOAD_MODELS_KEY).encode("utf-8")
            )
            instance.__aenter__.return_value = instance
            mock.return_value = instance
            yield mock

    @pytest.mark.asyncio
    async def test_async_discover(self, adapter, mock_async_client):
        result = await adapter.async_discover()
        assert len(result) == 2


# ── Error Policy B Tests ──────────────────────────────────────────────────

class TestCohereErrorPolicy:
    @pytest.fixture
    def adapter(self):
        return CohereDiscoveryAdapter(api_key="ck-test")

    def test_404_returns_empty_list(self, adapter):
        from magic_llm.util.http import HttpError
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
        from magic_llm.engine.discovery.base_discovery import DiscoveryError
        from magic_llm.util.http import HttpError
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Server error", status_code=500, response_content=b'internal error'
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            with pytest.raises(DiscoveryError) as exc:
                adapter.discover()
            assert exc.value.status_code == 500


# ── Registry ───────────────────────────────────────────────────────────────

class TestCohereRegistry:
    def test_get_adapter_returns_cohere(self):
        cls = get_adapter("cohere")
        assert cls is CohereDiscoveryAdapter
        assert cls is not None
