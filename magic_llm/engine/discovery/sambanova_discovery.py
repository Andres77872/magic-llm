"""SambaNova discovery adapter for OpenAI-compatible models listing.

Per spec.md Section "SambaNova Discovery":
- Primary: GET https://api.sambanova.ai/v1/models (OpenAI-compatible)
- Standard Bearer auth
- Error Policy B: 404 returns [], all other errors propagate
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter
from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    ModelNameRegexStrategy,
    ProviderDefaultsStrategy,
)

logger = logging.getLogger(__name__)


class SambaNovaDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for SambaNova models.

    Uses the official GET /v1/models endpoint (OpenAI-compatible format).
    Error Policy B: 404 returns [], 5xx and auth errors propagate.
    """

    # Capability inference: regex-on-name + provider defaults
    _capability_strategy = CompositeCapabilityInference([
        ModelNameRegexStrategy(),
        ProviderDefaultsStrategy(),
    ])

    # Default base URL used when _resolve_discovery_adapter passes base_url=None
    _DEFAULT_BASE_URL = "https://api.sambanova.ai"

    def __init__(
        self,
        provider: str = "sambanova",
        base_url: Optional[str] = None,
        api_key: str = None,
        **kwargs
    ):
        super().__init__(
            provider=provider,
            base_url=base_url or self._DEFAULT_BASE_URL,
            **kwargs
        )
        self.api_key = api_key

    def _get_endpoint_url(self) -> str:
        return f"{self.base_url}/v1/models"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── Pipeline overrides ────────────────────────────────────────────────

    # ── Llama-name heuristic hook ─────────────────────────────────────────

    @staticmethod
    def _llama_name_heuristic(raw_model: Dict[str, Any]) -> Optional[int]:
        """Fallback heuristic for context window based on Llama model name.

        SambaNova's Llama models have well-known context windows:
        - Llama 3 with 70B/405B: 128K
        - Other Llama 3 models: 8K
        """
        model_id = raw_model.get("id", "")
        if "llama-3" in model_id.lower() or "llama3" in model_id.lower():
            if "70b" in model_id.lower() or "405b" in model_id.lower():
                return 128000
            return 8192
        return None

    _context_window_hook = _llama_name_heuristic

    def _extract_raw_models(self, raw_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """SambaNova response may be a list, ``data`` key, or ``models`` key."""
        if isinstance(raw_response, list):
            return raw_response
        model_list = raw_response.get("data", [])
        if not model_list:
            model_list = raw_response.get("models", [])
        return model_list

    def _extract_model_id(self, raw_model: Dict[str, Any]) -> str:
        """SambaNova may return ``id``, ``model``, or ``name`` as identifier."""
        return raw_model.get("id") or raw_model.get("model") or raw_model.get("name", "")

    # _normalize_response is inherited from BaseDiscoveryAdapter.


# Register adapter
register_adapter("sambanova", SambaNovaDiscoveryAdapter)
