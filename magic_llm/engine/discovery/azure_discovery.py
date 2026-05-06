"""Azure discovery adapters — split into OpenAI data plane + Foundry management plane.

Per spec.md Domain 4 "Azure Adapter Split":
- ``AzureOpenAIDiscoveryAdapter``: Azure OpenAI data plane (api-key auth)
- ``AzureFoundryDiscoveryAdapter``: Azure AI Foundry management plane (Entra Bearer auth)

Each adapter extends ``BaseDiscoveryAdapter`` directly with NO shared constructor
heuristic.  No ``surface=`` kwarg, no runtime surface switching.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter
from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    ProviderDefaultsStrategy,
)
from magic_llm.model.discovery import ModelCapabilities, PricingInfo

logger = logging.getLogger(__name__)

# =============================================================================
# Azure OpenAI Data Plane Adapter
# =============================================================================


class AzureOpenAIDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for Azure OpenAI (data plane).

    Uses ``api-key`` header auth and the OpenAI-compatible data plane endpoint.
    No ``surface=`` kwarg — this adapter handles ONLY Azure OpenAI, not Foundry.

    Per spec.md:
    - Data plane: GET {resource}.openai.azure.com/openai/models with api-key header
    - API version: 2024-10-21
    """

    PROVIDER = "azure"
    AZURE_OPENAI_API_VERSION = "2024-10-21"

    _capability_strategy = CompositeCapabilityInference([
        ProviderDefaultsStrategy(),
    ])

    def __init__(
        self,
        resource_name: str = None,
        api_key: str = None,
        base_url: str = None,
        **kwargs,
    ):
        """Initialise Azure OpenAI discovery adapter.

        Args:
            resource_name: Azure OpenAI resource name (e.g. ``"my-resource"``).
                Used to construct ``https://{resource_name}.openai.azure.com``.
            api_key: Azure OpenAI API key.  Sent as ``api-key`` header.
            base_url: Optional explicit base URL.  Overrides resource_name
                construction when provided.
            **kwargs: Additional parameters forwarded to ``BaseDiscoveryAdapter``.
        """
        actual_base = base_url or ""
        super().__init__(provider="azure", base_url=actual_base, **kwargs)
        self.resource_name = resource_name
        self.api_key = api_key

    def _get_endpoint_url(self) -> str:
        """Construct the Azure OpenAI data plane endpoint URL.

        Returns:
            Full URL for model listing.

        Raises:
            ValueError: If neither ``resource_name`` nor ``base_url`` was provided.
        """
        if self.resource_name:
            base = f"https://{self.resource_name}.openai.azure.com"
        elif self.base_url:
            base = self.base_url.rstrip("/")
        else:
            raise ValueError(
                "AzureOpenAIDiscoveryAdapter requires resource_name or base_url"
            )
        return f"{base}/openai/models?api-version={self.AZURE_OPENAI_API_VERSION}"

    def _get_headers(self) -> Dict[str, str]:
        """Return auth headers for Azure OpenAI data plane (``api-key``, NOT Bearer).

        Returns:
            Headers dict with ``api-key`` (if available) and content-type headers.
        """
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    # ── Normalization pipeline overrides ─────────────────────────────────

    def _extract_raw_models(self, raw_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Azure OpenAI data plane uses the ``data`` key."""
        return raw_response.get("data", [])

    def _infer_capabilities(self, raw_model: Dict[str, Any]) -> ModelCapabilities:
        """Infer capabilities from Azure OpenAI capability fields.

        Azure OpenAI exposes a ``capabilities{}`` object in the API response.
        """
        capabilities = raw_model.get("capabilities", {})
        return ModelCapabilities(
            chat=True,
            vision=capabilities.get("vision", False),
            streaming=True,
            function_calling=capabilities.get("function_calling", True),
            embedding=capabilities.get("embeddings", False),
            completion=False,
            audio_input=False,
            reasoning=False,
        )

    # _infer_context_window is inherited from BaseDiscoveryAdapter.
    # Default alias chain (context_window → context_length → max_context → ...)
    # handles Azure OpenAI's ``context_window`` and ``max_context`` fields.


# =============================================================================
# Azure AI Foundry Management Plane Adapter
# =============================================================================


class AzureFoundryDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for Azure AI Foundry (management plane).

    Uses Entra ID ``Bearer`` token auth and the ARM management-plane endpoint.
    No ``surface=`` kwarg — this adapter handles ONLY Foundry.

    Per spec.md:
    - Management plane: GET management.azure.com/subscriptions/... with Bearer auth
    - API version: 2025-06-01
    """

    PROVIDER = "azure-foundry"
    AZURE_FOUNDRY_API_VERSION = "2025-06-01"

    _capability_strategy = CompositeCapabilityInference([
        ProviderDefaultsStrategy(),
    ])

    def __init__(
        self,
        subscription_id: str = None,
        location: str = None,
        entra_token: str = None,
        **kwargs,
    ):
        """Initialise Azure Foundry discovery adapter.

        Args:
            subscription_id: Azure subscription ID for the management plane URL.
            location: Azure region (e.g. ``"eastus"``).
            entra_token: Azure Entra ID Bearer token for auth.
            **kwargs: Additional parameters forwarded to ``BaseDiscoveryAdapter``.
        """
        super().__init__(
            provider="azure-foundry",
            base_url="https://management.azure.com",
            **kwargs,
        )
        self.subscription_id = subscription_id
        self.location = location
        self.entra_token = entra_token

    def _get_endpoint_url(self) -> str:
        """Construct the Azure AI Foundry management plane endpoint URL.

        Returns:
            Full URL for model listing.

        Raises:
            ValueError: If ``subscription_id`` or ``location`` is missing.
        """
        if not self.subscription_id:
            raise ValueError(
                "AzureFoundryDiscoveryAdapter requires subscription_id"
            )
        if not self.location:
            raise ValueError(
                "AzureFoundryDiscoveryAdapter requires location"
            )
        return (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/providers/Microsoft.CognitiveServices/locations/{self.location}"
            f"/models?api-version={self.AZURE_FOUNDRY_API_VERSION}"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Return auth headers for Azure Foundry management plane (Bearer, NOT api-key).

        Returns:
            Headers dict with ``Authorization: Bearer`` (if token available)
            and content-type headers.
        """
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.entra_token:
            headers["Authorization"] = f"Bearer {self.entra_token}"
        return headers

    # ── Normalization pipeline overrides ─────────────────────────────────

    def _extract_raw_models(self, raw_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Azure Foundry management plane uses the ``value`` key."""
        return raw_response.get("value", [])

    def _extract_model_id(self, raw_model: Dict[str, Any]) -> str:
        """Azure Foundry uses ``name`` or ``model`` for the model identifier."""
        return raw_model.get("name") or raw_model.get("model", "")

    def _extract_display_name(self, raw_model: Dict[str, Any]) -> Optional[str]:
        """Azure Foundry model display name."""
        return raw_model.get("display_name") or raw_model.get("name")

    def _infer_capabilities(self, raw_model: Dict[str, Any]) -> ModelCapabilities:
        """Infer capabilities from Azure Foundry management plane fields.

        Foundry uses ``kind`` to distinguish model types (``"OpenAI"``,
        ``"Chat"``, ``"Embeddings"``).
        """
        kind = raw_model.get("kind", "")
        return ModelCapabilities(
            chat=kind in ("OpenAI", "Chat"),
            vision="vision" in raw_model.get("model", "").lower(),
            streaming=True,
            function_calling=True,
            embedding=kind == "Embeddings",
            completion=False,
            audio_input=False,
            reasoning=False,
        )

    # Azure Foundry nests context under ``limits.context``.  Custom prefix
    # ensures the nested path is probed first, then falls back to defaults.
    _context_window_aliases = ["limits.context", "context_window", "context_length", "max_context"]

    def _extract_pricing(self, raw_model: Dict[str, Any]) -> Optional[PricingInfo]:
        """Extract pricing from Azure Foundry management plane response.

        Pricing lives in the ``skus[].cost[]`` array.
        """
        return self._extract_pricing_from_foundry(raw_model)

    @staticmethod
    def _extract_pricing_from_foundry(
        model_data: Dict[str, Any],
    ) -> Optional[PricingInfo]:
        """Extract pricing from Azure Foundry ``skus[].cost[].meterRate``.

        Per spec.md:
        - ``skus[].cost[].meterRate.input`` → ``PricingInfo.input_per_million``
        - ``skus[].cost[].meterRate.output`` → ``PricingInfo.output_per_million``

        Prices from meterRate are raw per-unit values; multiply by 1,000,000
        to convert to per-million-token pricing.
        """
        skus = model_data.get("skus", [])
        if not skus:
            return None

        for sku in skus:
            costs = sku.get("cost", [])
            for cost in costs:
                meter_rate = cost.get("meterRate", {})
                if not meter_rate:
                    continue
                input_price = meter_rate.get("input")
                output_price = meter_rate.get("output")
                if input_price is not None or output_price is not None:
                    return PricingInfo(
                        input_per_million=(
                            float(input_price) * 1_000_000
                            if input_price is not None
                            else None
                        ),
                        output_per_million=(
                            float(output_price) * 1_000_000
                            if output_price is not None
                            else None
                        ),
                    )
        return None


# =============================================================================
# Registry — two independent registrations
# =============================================================================

register_adapter("azure", AzureOpenAIDiscoveryAdapter)
register_adapter("azure-foundry", AzureFoundryDiscoveryAdapter)


# =============================================================================
# Azure Speech Services — out of v1 scope
# =============================================================================


class AzureSpeechDiscoveryAdapter(BaseDiscoveryAdapter):
    """Placeholder adapter for Azure Speech Services (OUT of v1 scope).

    Per spec.md Section "Azure Speech Services discovery excluded":
    - Raises ``NotImplementedError``
    - Indicates Azure Speech not supported for discovery in v1
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "Azure Speech Services discovery is not supported in v1. "
            "Use 'azure' engine for Azure OpenAI or 'azure-foundry' for Azure AI Foundry."
        )


# NOTE: azure-speech is intentionally NOT registered.
# AzureSpeechDiscoveryAdapter remains as a stub class for a future implementation.
