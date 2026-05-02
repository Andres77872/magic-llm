"""Shared base class for OpenAI-compatible discovery adapters.

Concrete subclasses set two class attributes::

    PROVIDER         — engine name (must match ``register_adapter()`` key)
    DEFAULT_BASE_URL — full discovery endpoint URL (NOT a host or base)

Base_url IS the full endpoint — the adapter owns its URL.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional

from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter
from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    ModelNameRegexStrategy,
    ProviderDefaultsStrategy,
)


class OpenAICompatibleAdapter(BaseDiscoveryAdapter):
    """Shared template for OpenAI-compatible discovery (13+ providers).

    Subclasses MUST set::

        PROVIDER         — engine name string
        DEFAULT_BASE_URL — full discovery endpoint URL
    """

    PROVIDER: str = ""          # set by subclass
    DEFAULT_BASE_URL: str = ""  # set by subclass

    # Capability inference: regex-on-name (Tier 2) + provider defaults (Tier 3)
    _capability_strategy = CompositeCapabilityInference([
        ModelNameRegexStrategy(),
        ProviderDefaultsStrategy(),
    ])

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        if not self.PROVIDER or not self.DEFAULT_BASE_URL:
            raise TypeError(
                f"{type(self).__name__} must set PROVIDER and DEFAULT_BASE_URL"
            )
        super().__init__(
            provider=self.PROVIDER,
            base_url=base_url or self.DEFAULT_BASE_URL,
            **kwargs,
        )
        self.api_key = api_key

    # ── Extension points ──────────────────────────────────────────────────

    def _get_endpoint_url(self) -> str:
        """Return the full discovery endpoint URL.

        Idempotent normalization: if ``base_url`` already targets the
        ``/models`` listing endpoint we use it as-is; otherwise we append
        ``/models``. This lets callers pass the chat-completions base URL
        (e.g. ``https://api.openai.com/v1`` or
        ``https://openrouter.ai/api/v1``) — which is what providers
        typically store — without needing to know the discovery suffix.

        Backward compatible: URLs that already end in ``/models`` (or
        ``/models/``) are returned unchanged.
        """
        url = (self.base_url or "").rstrip("/")
        if not url:
            return self.base_url
        # Already pointing at the models listing endpoint
        if url.endswith("/models") or "/models?" in url or url.endswith("/models/"):
            return url
        return f"{url}/models"

    def _get_headers(self) -> Dict[str, str]:
        """Standard Bearer auth + JSON content-type headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── Pipeline overrides ────────────────────────────────────────────────

    def _infer_context_window(self, raw_model: Dict[str, Any]) -> Optional[int]:
        """Infer context window from model name via CONTEXT_WINDOW_MAP."""
        model_id = raw_model.get("id", "")
        from magic_llm.engine.discovery.capabilities.models import CONTEXT_WINDOW_MAP
        for pattern, window in CONTEXT_WINDOW_MAP.items():
            if re.search(pattern, model_id, re.IGNORECASE):
                return window
        return None

    # _normalize_response is inherited from BaseDiscoveryAdapter — it iterates
    # ``_extract_raw_models()`` (default: ``data`` key) and calls
    # ``_normalize_single_model()`` for each model record.
