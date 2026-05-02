"""Capability inference strategies — composable, tiered, stateless.

Each strategy implements ``infer(provider, model_id, model_data)`` and
returns a ``Dict[str, Any]`` of capability field overrides.

CompositeCapabilityInference orchestrates the tiered fallback chain:
    Tier 1 (highest priority):  ProviderFieldStrategy
    Tier 2:                     ModelNameRegexStrategy
    Tier 3 (baseline):          ProviderDefaultsStrategy
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from magic_llm.engine.discovery.capabilities.models import (
    VISION_PATTERNS,
    EMBEDDING_PATTERNS,
    FUNCTION_CALLING_PATTERNS,
    CONTEXT_WINDOW_MAP,
)
from magic_llm.model.discovery import ModelCapabilities


# =============================================================================
# Protocol / Abstract Base
# =============================================================================


class CapabilityInferenceStrategy(ABC):
    """Given provider, model_id, and raw model data, return capability overrides.

    Returns only fields the strategy can determine (empty dict = no inference).
    Keys are ``ModelCapabilities`` field names; values are the inferred booleans.
    """

    @abstractmethod
    def infer(
        self,
        provider: str,
        model_id: str,
        model_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


# =============================================================================
# Tier 3 — Provider-Level Baseline Defaults
# =============================================================================


class ProviderDefaultsStrategy(CapabilityInferenceStrategy):
    """Tier 3: sensible baseline defaults per provider.

    Returns a dict with baseline values for each known provider.
    Unknown providers return ``{}`` (all defaults from ``ModelCapabilities``).
    """

    # Provider → baseline defaults (fields omitted use ModelCapabilities default)
    PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
        # OpenAI-compatible providers — all support chat, streaming, function_calling
        "openai": {"chat": True, "streaming": True, "function_calling": True},
        "deepinfra": {"chat": True, "streaming": True, "function_calling": True},
        "groq": {"chat": True, "streaming": True, "function_calling": True},
        "novita": {"chat": True, "streaming": True, "function_calling": True},
        "perplexity": {"chat": True, "streaming": True, "function_calling": True},
        "together": {"chat": True, "streaming": True, "function_calling": True},
        "mistral": {"chat": True, "streaming": True, "function_calling": True},
        "deepseek": {"chat": True, "streaming": True, "function_calling": True},
        "hyperbolic": {"chat": True, "streaming": True, "function_calling": True},
        "cerebras": {"chat": True, "streaming": True, "function_calling": True},
        "xai": {"chat": True, "streaming": True, "function_calling": True},
        "parasail": {"chat": True, "streaming": True, "function_calling": True},
        "nebius": {"chat": True, "streaming": True, "function_calling": True},
        # SambaNova — OpenAI-compatible in practice
        "sambanova": {"chat": True, "streaming": True, "function_calling": True},
        # Anthropic — all Claude models support chat, streaming, function_calling
        "anthropic": {"chat": True, "streaming": True, "function_calling": True},
        # Google Gemini — supports chat and streaming
        "google": {"chat": True, "streaming": True},
        # Cohere — supports chat and streaming
        "cohere": {"chat": True, "streaming": True},
        # OpenRouter — supports chat and streaming
        "openrouter": {"chat": True, "streaming": True, "function_calling": True},
        # Azure — all models support chat and streaming
        "azure": {"chat": True, "streaming": True},
        "azure-foundry": {"chat": True, "streaming": True},
    }

    def infer(
        self,
        provider: str,
        model_id: str,
        model_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return dict(self.PROVIDER_DEFAULTS.get(provider, {}))


# =============================================================================
# Tier 2 — Regex-on-Model-Name Inference
# =============================================================================


class ModelNameRegexStrategy(CapabilityInferenceStrategy):
    """Tier 2: infer capabilities via regex on ``model_data.get('id', '')``.

    Centralises all regex patterns that were previously scattered across
    13 OpenAI-compatible adapter files.  Returns a dict with only the
    fields it can definitively determine (empty dict = no match).
    """

    def infer(
        self,
        provider: str,
        model_id: str,
        model_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Match ``model_id`` against known patterns and return overrides."""
        result: Dict[str, Any] = {}

        # Vision — first pattern match wins
        for pattern in VISION_PATTERNS:
            if re.search(pattern, model_id, re.IGNORECASE):
                result["vision"] = True
                break

        # Embedding — when detected, also disable chat (matching old
        # OpenAICompatibleAdapter._infer_chat_capability behaviour)
        for pattern in EMBEDDING_PATTERNS:
            if re.search(pattern, model_id, re.IGNORECASE):
                result["embedding"] = True
                result["chat"] = False
                break

        # Function-calling
        for pattern in FUNCTION_CALLING_PATTERNS:
            if re.search(pattern, model_id, re.IGNORECASE):
                result["function_calling"] = True
                break

        return result


# =============================================================================
# Tier 1 — Provider API Response Fields
# =============================================================================


class ProviderFieldStrategy(CapabilityInferenceStrategy):
    """Tier 1: read capabilities from provider API response fields.

    Each provider entry maps a capability field name to a callable that
    extracts the value from ``model_data``.  Only fields that the provider
    definitively exposes are included — the strategy does NOT guess.
    """

    # Provider → {capability_field: extractor_callable(model_data) → bool}
    PROVIDER_FIELDS: Dict[str, Dict[str, Any]] = {
        "anthropic": {
            "vision": lambda d: (
                d.get("capabilities", {})
                .get("image_input", {})
                .get("supported", False)
            ),
            "reasoning": lambda d: (
                d.get("capabilities", {})
                .get("thinking", {})
                .get("supported", False)
            ),
        },
        "cohere": {
            "chat": lambda d: "chat" in d.get("endpoints", []),
            "function_calling": lambda d: "tool_use" in d.get("features", []),
            "embedding": lambda d: "embed" in d.get("endpoints", []),
        },
        "google": {
            "chat": lambda d: "generateContent"
            in d.get("supportedGenerationMethods", []),
            "embedding": lambda d: "embedContent"
            in d.get("supportedGenerationMethods", []),
        },
        "openrouter": {
            "vision": lambda d: "image"
            in d.get("architecture", {})
            .get("modality", {})
            .get("input", []),
            "audio_input": lambda d: "audio"
            in d.get("architecture", {})
            .get("modality", {})
            .get("input", []),
        },
    }

    def infer(
        self,
        provider: str,
        model_id: str,
        model_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return capability overrides from provider API response fields."""
        result: Dict[str, Any] = {}
        fields = self.PROVIDER_FIELDS.get(provider, {})
        for cap, extractor in fields.items():
            try:
                result[cap] = extractor(model_data)
            except (KeyError, IndexError, TypeError, AttributeError):
                pass
        return result


# =============================================================================
# Composite — Tiered Orchestrator
# =============================================================================


class CompositeCapabilityInference(CapabilityInferenceStrategy):
    """Orchestrates Tier 1 → Tier 2 → Tier 3 fallback chain.

    Strategies are provided in priority order: ``[Tier1, Tier2, Tier3]``.
    The composite applies them in REVERSE (Tier3 baseline → Tier2 override
    → Tier1 override), merging via inclusive ``dict.update()``.

    The result is always a complete ``ModelCapabilities`` instance — any
    fields not set by any strategy use the ``ModelCapabilities`` defaults.
    """

    def __init__(self, strategies: List[CapabilityInferenceStrategy]) -> None:
        self._strategies = strategies  # highest priority first

    def infer(
        self,
        provider: str,
        model_id: str,
        model_data: Dict[str, Any],
    ) -> ModelCapabilities:
        """Infer capabilities via tiered strategy merge.

        Applies strategies in reverse priority order — last strategy in
        the list (Tier 3) seeds the baseline, first strategy (Tier 1)
        provides the final override.
        """
        all_values: Dict[str, Any] = {}
        for strategy in reversed(self._strategies):
            all_values.update(strategy.infer(provider, model_id, model_data))
        return ModelCapabilities(**all_values)
