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
    DeclaredFieldStrategy,
    ModelNameRegexStrategy,
    ProviderDefaultsStrategy,
)
from magic_llm.model.discovery import PricingInfo


def _price(value: Any) -> Optional[float]:
    """Parse a price that may arrive as a number or a numeric string.

    Negative values are sentinels (OpenRouter uses ``"-1"`` for routers whose
    price depends on the routed model) and are treated as unknown.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price >= 0 else None


def _scaled(value: Any, factor: float) -> Optional[float]:
    price = _price(value)
    return None if price is None else round(price * factor, 6)


def extract_listing_pricing(raw_model: Dict[str, Any]) -> Optional[PricingInfo]:
    """Extract per-1M-token USD pricing from the shapes OpenAI-compatible hosts use.

    Verified against live listings:

    * ``pricing.prompt`` / ``pricing.completion`` as per-token strings
      (OpenRouter, Groq, SambaNova) → multiplied by 1e6
    * ``pricing.prompt.price_per_m_decimal`` (Novita) → already per 1M
    * ``pricing.input`` / ``pricing.output`` (Together) → already per 1M
    * ``metadata.pricing.input_tokens`` / ``output_tokens`` (DeepInfra) → per 1M
    * ``input_price`` / ``output_price`` (Hyperbolic) → per 1M
    * ``prompt_text_token_price`` / ``completion_text_token_price`` (xAI)
      → USD cents per 100M tokens, i.e. divided by 1e4
    """
    inp: Optional[float] = None
    out: Optional[float] = None

    pricing = raw_model.get("pricing")
    if isinstance(pricing, dict):
        prompt, completion = pricing.get("prompt"), pricing.get("completion")
        if isinstance(prompt, dict) or isinstance(completion, dict):
            inp = _price((prompt or {}).get("price_per_m_decimal"))
            out = _price((completion or {}).get("price_per_m_decimal"))
        elif prompt is not None or completion is not None:
            inp, out = _scaled(prompt, 1e6), _scaled(completion, 1e6)
        elif "input" in pricing or "output" in pricing:
            inp, out = _price(pricing.get("input")), _price(pricing.get("output"))

    metadata = raw_model.get("metadata")
    if inp is None and out is None and isinstance(metadata, dict) and isinstance(metadata.get("pricing"), dict):
        inp = _price(metadata["pricing"].get("input_tokens"))
        out = _price(metadata["pricing"].get("output_tokens"))

    if inp is None and out is None and ("input_price" in raw_model or "output_price" in raw_model):
        inp, out = _price(raw_model.get("input_price")), _price(raw_model.get("output_price"))

    if inp is None and out is None and "prompt_text_token_price" in raw_model:
        inp = _scaled(raw_model.get("prompt_text_token_price"), 1e-4)
        out = _scaled(raw_model.get("completion_text_token_price"), 1e-4)

    if inp is None and out is None:
        return None
    return PricingInfo(input_per_million=inp, output_per_million=out)


class OpenAICompatibleAdapter(BaseDiscoveryAdapter):
    """Shared template for OpenAI-compatible discovery (13+ providers).

    Subclasses MUST set::

        PROVIDER         — engine name string
        DEFAULT_BASE_URL — full discovery endpoint URL
    """

    PROVIDER: str = ""  # set by subclass
    DEFAULT_BASE_URL: str = ""  # set by subclass

    # Capability inference: declared per-model fields (Tier 1) + regex-on-name
    # (Tier 2) + provider defaults (Tier 3)
    _capability_strategy = CompositeCapabilityInference([
        DeclaredFieldStrategy(),
        ModelNameRegexStrategy(),
        ProviderDefaultsStrategy(),
    ])

    # Token-limit fields used by OpenAI-compatible hosts beyond the default
    # chain: Mistral ``max_context_length``, Novita ``context_size``, DeepInfra
    # ``metadata.context_length``, OpenRouter ``top_provider.*``, Groq
    # ``max_output_length``.
    _context_window_aliases = [
        "context_window",
        "context_length",
        "max_context_length",
        "context_size",
        "metadata.context_length",
        "top_provider.context_length",
    ]
    _max_output_tokens_aliases = [
        "max_output_tokens",
        "max_tokens",
        "max_completion_tokens",
        "max_output_length",
        "top_provider.max_completion_tokens",
    ]

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

    # ── Heuristic: CONTEXT_WINDOW_MAP regex fallback ──────────────────────

    @staticmethod
    def _context_window_map_fallback(raw_model: Dict[str, Any]) -> Optional[int]:
        """Fallback: match model ID against CONTEXT_WINDOW_MAP regex patterns.

        Invoked after the alias chain returns ``None`` for context_window.
        """
        model_id = raw_model.get("id", "")
        from magic_llm.engine.discovery.capabilities.models import CONTEXT_WINDOW_MAP
        for pattern, window in CONTEXT_WINDOW_MAP.items():
            if re.search(pattern, model_id, re.IGNORECASE):
                return window
        return None

    _context_window_hook = _context_window_map_fallback

    def _extract_pricing(self, raw_model: Dict[str, Any]) -> Optional[PricingInfo]:
        return extract_listing_pricing(raw_model)

    # _normalize_response is inherited from BaseDiscoveryAdapter — it iterates
    # ``_extract_raw_models()`` (default: ``data`` key) and calls
    # ``_normalize_single_model()`` for each model record.
