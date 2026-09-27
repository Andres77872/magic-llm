"""OpenRouter discovery adapter for Group F.

Per spec.md Section "OpenRouter Public No-Auth Discovery":
- Endpoint: GET https://openrouter.ai/api/v1/models
- NO authentication required — public endpoint
- Pricing available from OpenRouter's pricing fields
- Context window from context_length
- Capabilities from architecture.modality
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter, versioned_endpoint
from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    DeclaredFieldStrategy,
    ProviderFieldStrategy,
    ProviderDefaultsStrategy,
)
from magic_llm.engine.discovery.openai_compatible.base import extract_listing_pricing
from magic_llm.model.discovery import PricingInfo

logger = logging.getLogger(__name__)


class OpenRouterDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for OpenRouter models.

    Per spec.md Section "OpenRouter Public No-Auth Discovery":
    - Public endpoint — no auth required
    - Rich metadata including pricing, context, capabilities
    - Pricing extracted from OpenRouter's pricing fields
    """

    HOSTS = ("openrouter.ai",)

    # Capability inference: API fields (Tier 1) + provider defaults (Tier 3).
    # DeclaredFieldStrategy reads architecture.input/output_modalities and
    # supported_parameters (tools, reasoning); ProviderFieldStrategy keeps the
    # legacy ``architecture.modality`` dict shape working.
    _capability_strategy = CompositeCapabilityInference([
        DeclaredFieldStrategy(),
        ProviderFieldStrategy(),
        ProviderDefaultsStrategy(),
    ])

    def __init__(
        self,
        provider: str = "openrouter",
        base_url: str = "https://openrouter.ai/api",
        api_key: str = None,  # Optional, not required for discovery
        **kwargs
    ):
        """Initialize OpenRouter discovery adapter.

        Args:
            provider: Provider identifier
            base_url: OpenRouter API base URL
            api_key: Optional API key (not required for public endpoint)
            **kwargs: Additional parameters
        """
        super().__init__(
            provider=provider,
            base_url=base_url or "https://openrouter.ai/api",
            **kwargs
        )
        self.api_key = api_key

    def _get_endpoint_url(self) -> str:
        """Get OpenRouter public models endpoint.

        Per spec.md:
        - GET https://openrouter.ai/api/v1/models
        - No auth required

        Providers store the chat base URL ``https://openrouter.ai/api/v1``;
        both it and the bare ``https://openrouter.ai/api`` resolve here.
        """
        return versioned_endpoint(self.base_url, "v1")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter request.

        OpenRouter models endpoint is public — no auth required.
        Optionally include API key if provided.
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        # API key optional for public endpoint
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── Token alias profiles ──────────────────────────────────────────────
    # OpenRouter uses ``context_length`` as the authoritative context field
    # (not ``context_window``). The custom prefix reverses the default order
    # so that ``context_length`` is probed first.

    _context_window_aliases = ["context_length", "context_window", "top_provider.context_length"]

    # ── Pipeline overrides ────────────────────────────────────────────────

    def _extract_pricing(self, model_data: Dict[str, Any]) -> Optional[PricingInfo]:
        """Extract pricing from OpenRouter model data.

        ``pricing.prompt`` / ``pricing.completion`` are USD per token (strings
        such as ``"0.0000025"``) and are converted to per-1M. Router entries
        (``openrouter/auto``) publish ``"-1"`` because the price depends on the
        routed model; those become unknown instead of negative prices.
        """
        return extract_listing_pricing(model_data)

    # _normalize_response is inherited from BaseDiscoveryAdapter — default
    # ``data`` key for _extract_raw_models, default ``id`` for
    # _extract_model_id, default ``display_name→name→id`` for
    # _extract_display_name (OpenRouter uses ``name``).


# Register adapter
register_adapter("openrouter", OpenRouterDiscoveryAdapter)
