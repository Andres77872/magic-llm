"""Tests for Azure discovery adapters — OpenAI data plane + Foundry management plane.

Per spec.md Domain 4 "Azure Adapter Split":
- ``AzureOpenAIDiscoveryAdapter``: data plane, api-key auth, ``{data: [...]}`` response
- ``AzureFoundryDiscoveryAdapter``: management plane, Entra Bearer auth, ``{value: [...]}`` response
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery.base_discovery import (
    DiscoveryAuthError,
    DiscoveryError,
    HttpError,
)
from magic_llm.engine.discovery.azure_discovery import (
    AzureFoundryDiscoveryAdapter,
    AzureOpenAIDiscoveryAdapter,
)
from magic_llm.model.discovery import NormalizedDiscoveredModel, PricingInfo


# =============================================================================
# AzureOpenAIDiscoveryAdapter — Data Plane
# =============================================================================


class TestAzureOpenAIDiscoveryAdapter:
    """Azure OpenAI data plane adapter tests."""

    DATA_PLANE_PAYLOAD = {
        "data": [
            {
                "id": "gpt-4o",
                "display_name": "GPT-4o",
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "embeddings": False,
                },
                "context_window": 128000,
            },
            {
                "id": "text-embedding-3-small",
                "display_name": "Text Embedding 3 Small",
                "capabilities": {
                    "vision": False,
                    "function_calling": False,
                    "embeddings": True,
                },
                "context_window": 8192,
            },
        ]
    }

    # ── Construction & URL ─────────────────────────────────────────────

    def test_constructs_data_plane_url_from_resource_name(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        url = adapter._get_endpoint_url()
        assert url == (
            "https://my-resource.openai.azure.com/openai/models"
            "?api-version=2024-10-21"
        )

    def test_constructs_data_plane_url_from_base_url(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            base_url="https://custom.azure.com", api_key="sk-test"
        )
        url = adapter._get_endpoint_url()
        assert url == (
            "https://custom.azure.com/openai/models?api-version=2024-10-21"
        )

    def test_raises_value_error_when_no_resource_or_base_url(self):
        adapter = AzureOpenAIDiscoveryAdapter(api_key="sk-test")
        with pytest.raises(ValueError, match="resource_name or base_url"):
            adapter._get_endpoint_url()

    # ── Auth Headers ───────────────────────────────────────────────────

    def test_uses_api_key_header_not_bearer(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        headers = adapter._get_headers()
        assert headers.get("api-key") == "sk-test"
        assert "Authorization" not in headers

    def test_no_api_key_when_not_provided(self):
        adapter = AzureOpenAIDiscoveryAdapter(resource_name="my-resource")
        headers = adapter._get_headers()
        assert "api-key" not in headers

    # ── Response Normalisation ─────────────────────────────────────────

    def test_normalize_response_data_plane(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        result = adapter._normalize_response(self.DATA_PLANE_PAYLOAD)
        assert len(result) == 2
        assert isinstance(result[0], NormalizedDiscoveredModel)
        assert result[0].external_id == "gpt-4o"
        assert result[0].provider == "azure"
        assert result[0].capabilities.vision is True
        assert result[0].capabilities.chat is True
        assert result[0].capabilities.embedding is False
        assert result[0].context_window == 128000
        assert result[0].max_input_tokens is None
        assert result[1].external_id == "text-embedding-3-small"
        assert result[1].capabilities.embedding is True

    def test_empty_data_plane_payload(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        result = adapter._normalize_response({"data": []})
        assert result == []

    # ── Error Handling (Error Policy B) ────────────────────────────────

    def test_404_returns_empty_list(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Not Found", status_code=404
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            result = adapter.discover()
            assert result == []

    def test_401_raises_auth_error(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Unauthorized", status_code=401
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            with pytest.raises(DiscoveryAuthError):
                adapter.discover()

    def test_5xx_raises_discovery_error(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Server Error", status_code=503
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            with pytest.raises(DiscoveryError):
                adapter.discover()

    # ── Sync / Async Equivalence ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sync_async_equivalence(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        payload = self.DATA_PLANE_PAYLOAD

        # Sync
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock_sync:
            inst = MagicMock()
            inst.request.return_value = json.dumps(payload).encode("utf-8")
            inst.__enter__.return_value = inst
            mock_sync.return_value = inst
            sync_result = adapter.discover()

        # Async
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock_async:
            inst = AsyncMock()
            inst.request = AsyncMock(
                return_value=json.dumps(payload).encode("utf-8")
            )
            inst.__aenter__.return_value = inst
            mock_async.return_value = inst
            async_result = await adapter.async_discover()

        assert len(sync_result) == len(async_result)
        for s, a in zip(sync_result, async_result):
            assert s.external_id == a.external_id
            assert s.provider == a.provider
            assert s.capabilities == a.capabilities
            assert s.context_window == a.context_window

    # ── Pipeline — discover() end-to-end ───────────────────────────────

    def test_discover_returns_normalized_models(self):
        adapter = AzureOpenAIDiscoveryAdapter(
            resource_name="my-resource", api_key="sk-test"
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.return_value = json.dumps(
                self.DATA_PLANE_PAYLOAD
            ).encode("utf-8")
            inst.__enter__.return_value = inst
            mock.return_value = inst
            result = adapter.discover()
            assert len(result) == 2
            assert result[0].external_id == "gpt-4o"
            assert result[0].provider == "azure"
            # Verify exact URL was hit
            mock.return_value.request.assert_called_once()
            args = mock.return_value.request.call_args
            assert "openai.azure.com" in args[0][1]


# =============================================================================
# AzureFoundryDiscoveryAdapter — Management Plane
# =============================================================================


class TestAzureFoundryDiscoveryAdapter:
    """Azure AI Foundry management plane adapter tests."""

    FOUNDRY_PAYLOAD = {
        "value": [
            {
                "name": "gpt-4o",
                "display_name": "GPT-4o",
                "kind": "OpenAI",
                "model": "gpt-4o",
                "limits": {"context": 128000},
                "skus": [
                    {
                        "cost": [
                            {
                                "meterRate": {
                                    "input": 0.000_003,
                                    "output": 0.000_015,
                                }
                            }
                        ]
                    }
                ],
            },
            {
                "name": "text-embedding-3-small",
                "display_name": "Text Embedding 3 Small",
                "kind": "Embeddings",
                "model": "text-embedding-3-small",
                "limits": {"context": 8192},
                "skus": [],
            },
        ]
    }

    # ── Construction & URL ─────────────────────────────────────────────

    def test_constructs_management_plane_url(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        url = adapter._get_endpoint_url()
        assert url == (
            "https://management.azure.com/subscriptions/sub-123"
            "/providers/Microsoft.CognitiveServices/locations/eastus"
            "/models?api-version=2025-06-01"
        )

    def test_raises_value_error_when_subscription_id_missing(self):
        adapter = AzureFoundryDiscoveryAdapter(
            location="eastus", entra_token="et-test"
        )
        with pytest.raises(ValueError, match="subscription_id"):
            adapter._get_endpoint_url()

    def test_raises_value_error_when_location_missing(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123", entra_token="et-test"
        )
        with pytest.raises(ValueError, match="location"):
            adapter._get_endpoint_url()

    # ── Auth Headers ───────────────────────────────────────────────────

    def test_uses_bearer_token_not_api_key(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        headers = adapter._get_headers()
        assert headers.get("Authorization") == "Bearer et-test"
        assert "api-key" not in headers

    def test_no_auth_header_when_token_not_provided(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123", location="eastus"
        )
        headers = adapter._get_headers()
        assert "Authorization" not in headers
        assert "api-key" not in headers

    # ── Response Normalisation ─────────────────────────────────────────

    def test_normalize_response_foundry(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        result = adapter._normalize_response(self.FOUNDRY_PAYLOAD)
        assert len(result) == 2
        assert isinstance(result[0], NormalizedDiscoveredModel)
        assert result[0].external_id == "gpt-4o"
        assert result[0].provider == "azure-foundry"
        assert result[0].capabilities.chat is True
        assert result[0].capabilities.embedding is False
        assert result[0].context_window == 128000
        assert result[0].max_input_tokens is None
        assert result[1].external_id == "text-embedding-3-small"
        assert result[1].capabilities.chat is False
        assert result[1].capabilities.embedding is True

    def test_empty_foundry_payload(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        result = adapter._normalize_response({"value": []})
        assert result == []

    # ── Pricing Extraction ─────────────────────────────────────────────

    def test_extracts_pricing_from_foundry_response(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        result = adapter._normalize_response(self.FOUNDRY_PAYLOAD)
        # First model has skus with cost data
        pricing = result[0].pricing
        assert pricing is not None
        assert isinstance(pricing, PricingInfo)
        # 0.000003 * 1_000_000 = 3.0
        assert pricing.input_per_million == 3.0
        # 0.000015 * 1_000_000 = 15.0
        assert pricing.output_per_million == 15.0
        # Second model has empty skus — no pricing
        assert result[1].pricing is None

    def test_no_pricing_when_no_skus(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        result = adapter._normalize_response({"value": [{"name": "no-skus"}]})
        assert result[0].pricing is None

    # ── Sync / Async Equivalence ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sync_async_equivalence(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        payload = self.FOUNDRY_PAYLOAD

        # Sync
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock_sync:
            inst = MagicMock()
            inst.request.return_value = json.dumps(payload).encode("utf-8")
            inst.__enter__.return_value = inst
            mock_sync.return_value = inst
            sync_result = adapter.discover()

        # Async
        with patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as mock_async:
            inst = AsyncMock()
            inst.request = AsyncMock(
                return_value=json.dumps(payload).encode("utf-8")
            )
            inst.__aenter__.return_value = inst
            mock_async.return_value = inst
            async_result = await adapter.async_discover()

        assert len(sync_result) == len(async_result)
        for s, a in zip(sync_result, async_result):
            assert s.external_id == a.external_id
            assert s.provider == a.provider
            assert s.capabilities == a.capabilities
            assert s.pricing == a.pricing

    # ── Pipeline — discover() end-to-end ───────────────────────────────

    def test_discover_returns_normalized_models(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.return_value = json.dumps(
                self.FOUNDRY_PAYLOAD
            ).encode("utf-8")
            inst.__enter__.return_value = inst
            mock.return_value = inst
            result = adapter.discover()
            assert len(result) == 2
            assert result[0].external_id == "gpt-4o"
            assert result[0].provider == "azure-foundry"
            # Verify exact URL was hit
            mock.return_value.request.assert_called_once()
            args = mock.return_value.request.call_args
            assert "management.azure.com" in args[0][1]

    # ── Error Handling (Error Policy B) ────────────────────────────────

    def test_404_returns_empty_list(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Not Found", status_code=404
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            result = adapter.discover()
            assert result == []

    def test_401_raises_auth_error(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Unauthorized", status_code=401
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            with pytest.raises(DiscoveryAuthError):
                adapter.discover()

    def test_5xx_raises_discovery_error(self):
        adapter = AzureFoundryDiscoveryAdapter(
            subscription_id="sub-123",
            location="eastus",
            entra_token="et-test",
        )
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as mock:
            inst = MagicMock()
            inst.request.side_effect = HttpError(
                "Server Error", status_code=502
            )
            inst.__enter__.return_value = inst
            mock.return_value = inst
            with pytest.raises(DiscoveryError):
                adapter.discover()
