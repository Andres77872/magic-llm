"""Cohere discovery adapter for Group D.

Per spec.md Section "Cohere provider discovery":
- Endpoint: GET https://api.cohere.com/v1/models
- Normalization: chat from endpoints array, function_calling from features array
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional
from urllib.parse import quote

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter
from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    ProviderFieldStrategy,
    ProviderDefaultsStrategy,
)

logger = logging.getLogger(__name__)


class CohereDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for Cohere models.

    Per spec.md Section "Cohere provider discovery":
    - Endpoint: GET https://api.cohere.com/v1/models
    - Bearer auth via Authorization header
    - Capabilities inferred from endpoints/features arrays
    """

    # Capability inference: API fields (Tier 1) + provider defaults (Tier 3)
    # ProviderFieldStrategy reads endpoints[]/features[] for chat/function_calling/embedding
    _capability_strategy = CompositeCapabilityInference([
        ProviderFieldStrategy(),
        ProviderDefaultsStrategy(),
    ])

    def __init__(
        self,
        provider: str = "cohere",
        base_url: str = "https://api.cohere.com",
        api_key: str = None,
        **kwargs
    ):
        """Initialize Cohere discovery adapter.

        Args:
            provider: Provider identifier
            base_url: Cohere API base URL
            api_key: Cohere API key
            **kwargs: Additional parameters
        """
        super().__init__(
            provider=provider,
            base_url=base_url or "https://api.cohere.com",
            **kwargs
        )
        self.api_key = api_key

    HOSTS = ("api.cohere.com", "api.cohere.ai")

    # Cohere pages with ``page_size`` (default 20, max 1000) and ``page_token``.
    _PAGE_SIZE = 1000

    def _get_endpoint_url(self) -> str:
        """Get Cohere models listing endpoint.

        The listing only exists on the v1 API, but providers often store the
        v2 chat base URL (``https://api.cohere.com/v2``); any trailing API
        version is replaced with ``/v1``.
        """
        url = self.base_url.rstrip("/")
        if url.endswith("/models"):
            return url
        return re.sub(r"/v\d+$", "", url) + "/v1/models"

    def _first_page_url(self) -> str:
        return f"{self._get_endpoint_url()}?page_size={self._PAGE_SIZE}"

    def _next_page_url(self, raw_response: Any, current_url: str) -> Optional[str]:
        token = raw_response.get("next_page_token") if isinstance(raw_response, dict) else None
        if token:
            return f"{self._get_endpoint_url()}?page_size={self._PAGE_SIZE}&page_token={quote(str(token), safe='')}"
        return None

    def _get_headers(self) -> Dict[str, str]:
        """Get Cohere auth headers with Bearer token."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── Pipeline overrides ────────────────────────────────────────────────

    def _extract_raw_models(self, raw_response: Any) -> List[Dict[str, Any]]:
        """Cohere response format varies — handles list, ``models`` key, and ``data`` key."""
        if isinstance(raw_response, list):
            return raw_response
        model_list = raw_response.get("models", [])
        if not model_list and "data" in raw_response:
            model_list = raw_response.get("data", [])
        return model_list

    def _extract_model_id(self, raw_model: Dict[str, Any]) -> str:
        """Cohere uses ``name`` as the primary identifier, then ``id``."""
        return raw_model.get("name", raw_model.get("id", ""))

    # _normalize_response is inherited from BaseDiscoveryAdapter — default
    # ``context_length`` in _infer_context_window matches Cohere's field name.


# Register adapter
register_adapter("cohere", CohereDiscoveryAdapter)
