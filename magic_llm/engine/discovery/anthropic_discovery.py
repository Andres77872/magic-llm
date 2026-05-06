"""Anthropic discovery adapter for Group D.

Per spec.md Section "Anthropic provider discovery":
- Endpoint: GET https://api.anthropic.com/v1/models
- Headers: x-api-key: {api_key}, anthropic-version: 2023-06-01
- Normalization: map capabilities{} object directly
- Anthropic returns capability info directly in response
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

logger = logging.getLogger(__name__)


class AnthropicDiscoveryAdapter(BaseDiscoveryAdapter):
    """Discovery adapter for Anthropic Claude models.

    Per spec.md Section "Anthropic provider discovery":
    - Endpoint: GET https://api.anthropic.com/v1/models
    - Headers: x-api-key header (NOT Bearer auth)
    - anthropic-version header required
    - Capabilities extracted from response capabilities{} object
    """

    # Anthropic API version header required for all requests
    ANTHROPIC_VERSION = "2023-06-01"

    # Capability inference: API fields (Tier 1) + provider defaults (Tier 3)
    _capability_strategy = CompositeCapabilityInference([
        ProviderFieldStrategy(),
        ProviderDefaultsStrategy(),
    ])

    def __init__(
        self,
        provider: str = "anthropic",
        base_url: str = "https://api.anthropic.com",
        api_key: str = None,
        **kwargs
    ):
        """Initialize Anthropic discovery adapter.

        Args:
            provider: Provider identifier
            base_url: Anthropic API base URL
            api_key: Anthropic API key
            **kwargs: Additional parameters
        """
        super().__init__(
            provider=provider,
            base_url=base_url or "https://api.anthropic.com",
            **kwargs
        )
        self.api_key = api_key

    def _get_endpoint_url(self) -> str:
        """Get Anthropic models listing endpoint.

        Per spec.md:
        - GET https://api.anthropic.com/v1/models
        """
        return f"{self.base_url}/v1/models"

    def _get_headers(self) -> Dict[str, str]:
        """Get Anthropic auth headers.

        Per spec.md Section "Anthropic provider discovery":
        - x-api-key header (NOT Authorization Bearer)
        - anthropic-version header required
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-version": self.ANTHROPIC_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    # ── Pipeline overrides ────────────────────────────────────────────────

    # ── Claude-name heuristic hook ────────────────────────────────────────

    @staticmethod
    def _claude_name_heuristic(raw_model: Dict[str, Any]) -> Optional[int]:
        """Fallback heuristic for Claude context window based on model name.

        Anthropic models have well-known context windows:
        - Claude 3.5 and Claude 3: 200K
        - Claude 2: 100K

        For unknown model names (e.g., Claude 4.x, Claude Opus), returns
        ``None`` — the alias chain will pick up ``max_input_tokens`` instead.
        """
        model_id = raw_model.get("id", "")
        if "claude-3-5" in model_id or "claude-3.5" in model_id:
            return 200000
        elif "claude-3" in model_id:
            return 200000
        elif "claude-2" in model_id:
            return 100000
        return None

    _context_window_hook = _claude_name_heuristic

    # _normalize_response is inherited from BaseDiscoveryAdapter — it uses the
    # default ``data`` key for _extract_raw_models, default ``id`` for
    # _extract_model_id, default ``display_name→name→id`` for
    # _extract_display_name, base ``_infer_max_input_tokens`` for max_input_tokens,
    # and default ``max_tokens`` fallback for _infer_max_output_tokens (Anthropic
    # returns ``max_tokens``).


# Register adapter
register_adapter("anthropic", AnthropicDiscoveryAdapter)
