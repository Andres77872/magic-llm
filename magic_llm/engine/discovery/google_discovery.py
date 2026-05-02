"""Google AI Studio discovery adapter for Group D.

Per spec.md Section "Google AI Studio provider discovery":
- Endpoint: GET https://generativelanguage.googleapis.com/v1beta/models
- Header: x-goog-api-key: {api_key}
- Normalization: chat from supportedGenerationMethods, context_window from inputTokenLimit
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter
from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    ProviderFieldStrategy,
    ProviderDefaultsStrategy,
)

logger = logging.getLogger(__name__)


class GoogleDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for Google AI Studio (Gemini) models.

    Per spec.md Section "Google AI Studio provider discovery":
    - Endpoint: GET https://generativelanguage.googleapis.com/v1beta/models
    - Header: x-goog-api-key (NOT Authorization Bearer)
    - Chat inferred from supportedGenerationMethods containing "generateContent"
    - Context window from inputTokenLimit
    """

    # Capability inference: API fields (Tier 1) + provider defaults (Tier 3)
    # ProviderFieldStrategy reads supportedGenerationMethods for chat/embedding.
    # Vision is inferred separately via Google-specific patterns below.
    _capability_strategy = CompositeCapabilityInference([
        ProviderFieldStrategy(),
        ProviderDefaultsStrategy(),
    ])

    def __init__(
        self,
        provider: str = "google",
        base_url: str = "https://generativelanguage.googleapis.com",
        api_key: str = None,
        **kwargs
    ):
        """Initialize Google discovery adapter.

        Args:
            provider: Provider identifier
            base_url: Google AI Studio API base URL
            api_key: Google API key
            **kwargs: Additional parameters
        """
        super().__init__(
            provider=provider,
            base_url=base_url,
            **kwargs
        )
        self.api_key = api_key

    def _get_endpoint_url(self) -> str:
        """Get Google AI Studio models listing endpoint.

        Per spec.md:
        - GET https://generativelanguage.googleapis.com/v1beta/models
        """
        return f"{self.base_url}/v1beta/models"

    def _get_headers(self) -> Dict[str, str]:
        """Get Google auth headers.

        Per spec.md Section "Google AI Studio provider discovery":
        - x-goog-api-key header (NOT Authorization Bearer)
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        return headers

    # ── Pipeline overrides ────────────────────────────────────────────────

    def _extract_raw_models(self, raw_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Google returns models in ``models`` key (NOT ``data``)."""
        return raw_response.get("models", [])

    def _extract_model_id(self, raw_model: Dict[str, Any]) -> str:
        """Strip ``models/`` prefix from Google's full-path names."""
        full_name = raw_model.get("name", "")
        if full_name.startswith("models/"):
            return full_name[7:]  # Remove "models/" prefix
        elif full_name.startswith("publishers/google/models/"):
            return full_name.split("/")[-1]
        return full_name

    def _extract_display_name(self, raw_model: Dict[str, Any]) -> Optional[str]:
        """Google uses camelCase ``displayName`` (not snake_case)."""
        return raw_model.get("displayName")

    def _infer_context_window(self, raw_model: Dict[str, Any]) -> Optional[int]:
        """Google provides ``inputTokenLimit`` directly in model data."""
        return raw_model.get("inputTokenLimit")

    def _infer_max_output_tokens(self, raw_model: Dict[str, Any]) -> Optional[int]:
        """Google provides ``outputTokenLimit`` directly in model data."""
        return raw_model.get("outputTokenLimit")

    def _infer_vision_capability(self, model_data: Dict[str, Any]) -> bool:
        """Infer vision capability from model name.

        Gemini Pro Vision and newer Gemini models support images.
        """
        model_name = model_data.get("name", "")
        display_name = model_data.get("displayName", "")

        # Gemini Pro Vision and Gemini 1.5+ support images
        if "vision" in model_name.lower() or "vision" in display_name.lower():
            return True
        if "gemini-1.5" in model_name.lower() or "gemini-2" in model_name.lower():
            return True
        if "gemini-pro-vision" in model_name.lower():
            return True

        return False

    def _infer_capabilities(self, raw_model: Dict[str, Any]):
        """Infer capabilities — strategy baseline + Google-specific vision."""
        caps = super()._infer_capabilities(raw_model)
        # Google-specific vision inference on top of strategy result
        if self._infer_vision_capability(raw_model):
            caps.vision = True
        return caps

    # _normalize_response is inherited from BaseDiscoveryAdapter — it uses
    # the pipeline overrides above (_extract_raw_models, _extract_model_id,
    # _extract_display_name, _infer_context_window, _infer_max_output_tokens).


# Register adapter
register_adapter("google", GoogleDiscoveryAdapter)
