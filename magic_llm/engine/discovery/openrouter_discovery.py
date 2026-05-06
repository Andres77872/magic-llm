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

from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter
from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    ProviderFieldStrategy,
    ProviderDefaultsStrategy,
)
from magic_llm.model.discovery import PricingInfo

logger = logging.getLogger(__name__)


class OpenRouterDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for OpenRouter models.

    Per spec.md Section "OpenRouter Public No-Auth Discovery":
    - Public endpoint — no auth required
    - Rich metadata including pricing, context, capabilities
    - Pricing extracted from OpenRouter's pricing fields
    """

    # Capability inference: API fields (Tier 1) + provider defaults (Tier 3)
    # ProviderFieldStrategy reads architecture.modality for vision/audio_input
    _capability_strategy = CompositeCapabilityInference([
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
        """
        return f"{self.base_url}/v1/models"

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

    _context_window_aliases = ["context_length", "context_window"]

    # ── Pipeline overrides ────────────────────────────────────────────────

    def _extract_pricing(self, model_data: Dict[str, Any]) -> Optional[PricingInfo]:
        """Extract pricing from OpenRouter model data.

        OpenRouter provides pricing in model.pricing object:
        - pricing.prompt: price per prompt token (may be string or number)
        - pricing.completion: price per completion token
        """
        pricing_data = model_data.get("pricing", {})

        if not pricing_data:
            return None

        # OpenRouter prices may be strings like "0.000001" or numbers
        # Convert to per-million prices
        input_price = pricing_data.get("prompt") or pricing_data.get("input")
        output_price = pricing_data.get("completion") or pricing_data.get("output")

        if input_price is None and output_price is None:
            return None

        # Convert to float if string
        def to_float(val):
            if val is None:
                return None
            if isinstance(val, str):
                try:
                    return float(val)
                except ValueError:
                    return None
            return float(val)

        input_per_million = to_float(input_price)
        output_per_million = to_float(output_price)

        # OpenRouter prices are typically per-token, convert to per-million
        # If price is very small (< 1), it's likely per-token
        if input_per_million and input_per_million < 1:
            input_per_million = input_per_million * 1_000_000
        if output_per_million and output_per_million < 1:
            output_per_million = output_per_million * 1_000_000

        return PricingInfo(
            input_per_million=input_per_million,
            output_per_million=output_per_million,
        )

    # _normalize_response is inherited from BaseDiscoveryAdapter — default
    # ``data`` key for _extract_raw_models, default ``id`` for
    # _extract_model_id, default ``display_name→name→id`` for
    # _extract_display_name (OpenRouter uses ``name``).


# Register adapter
register_adapter("openrouter", OpenRouterDiscoveryAdapter)
