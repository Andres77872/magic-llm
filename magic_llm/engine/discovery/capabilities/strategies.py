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
from typing import Any, Dict, List

from magic_llm.engine.discovery.capabilities.models import (
    VISION_PATTERNS,
    EMBEDDING_PATTERNS,
    FUNCTION_CALLING_PATTERNS,
    AUDIO_INPUT_PATTERNS,
    AUDIO_OUTPUT_PATTERNS,
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

        # Audio input / STT — narrow verified patterns only
        for pattern in AUDIO_INPUT_PATTERNS:
            if re.search(pattern, model_id, re.IGNORECASE):
                result["audio_input"] = True
                break

        # Audio output / TTS — narrow verified patterns only
        for pattern in AUDIO_OUTPUT_PATTERNS:
            if re.search(pattern, model_id, re.IGNORECASE):
                result["audio_output"] = True
                break

        if result.get("audio_output") and re.search(r"(tts|text-to-speech|sonic)", model_id, re.IGNORECASE):
            # TTS/audio-output model names such as gpt-4o-mini-tts should not
            # inherit broad chat-vision regex claims from gpt-4o.
            result.pop("vision", None)

        return result


# =============================================================================
# Tier 1 — Provider API Response Fields
# =============================================================================


def _openrouter_modalities(model_data: Dict[str, Any], direction: str) -> List[str]:
    """OpenRouter modalities in either API shape.

    Current API: ``architecture.input_modalities`` / ``output_modalities``
    lists (``architecture.modality`` is a display string such as
    ``"text+image->text"``). Legacy shape: ``architecture.modality`` as a
    ``{"input": [...], "output": [...]}`` dict.
    """
    arch = model_data.get("architecture") or {}
    listed = arch.get(f"{direction}_modalities")
    if isinstance(listed, list):
        return listed
    modality = arch.get("modality")
    if isinstance(modality, dict):
        return list(modality.get(direction) or [])
    return []


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
            # ``features`` may be null. Current API names: "tools" (formerly
            # "tool_use"), "vision"; transcription models expose only the
            # "transcriptions" endpoint.
            "chat": lambda d: "chat" in (d.get("endpoints") or []),
            "function_calling": lambda d: any(
                f in (d.get("features") or []) for f in ("tools", "tool_use")
            ),
            "embedding": lambda d: "embed" in (d.get("endpoints") or []),
            "vision": lambda d: "vision" in (d.get("features") or []),
            "audio_input": lambda d: "transcriptions" in (d.get("endpoints") or []),
        },
        "google": {
            "chat": lambda d: "generateContent"
                              in d.get("supportedGenerationMethods", []),
            "embedding": lambda d: "embedContent"
                                   in d.get("supportedGenerationMethods", []),
        },
        "openrouter": {
            "vision": lambda d: "image" in _openrouter_modalities(d, "input"),
            "audio_input": lambda d: "audio" in _openrouter_modalities(d, "input"),
            "audio_output": lambda d: "audio" in _openrouter_modalities(d, "output"),
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
# Tier 1 (provider-agnostic) — Declared-field Strategy
# =============================================================================

# Together ``type`` values that are served by the chat completions endpoint.
_CHAT_MODEL_TYPES = {"chat", "language", "code"}
# ``type`` / ``model_type`` values for models that are NOT chat models.
_NON_CHAT_MODEL_TYPES = {
    "embedding", "embeddings", "rerank", "moderation", "image",
    "audio", "transcribe", "transcription", "tts", "video",
}


def _as_list(value: Any) -> List[str]:
    return [str(v).lower() for v in value] if isinstance(value, (list, tuple)) else []


class DeclaredFieldStrategy(CapabilityInferenceStrategy):
    """Tier 1: read capabilities that OpenAI-compatible hosts declare per model.

    Many hosts extend the bare OpenAI ``/models`` record with explicit
    capability metadata, each in its own shape. This strategy reads every
    well-known shape it finds and only reports what the record states:

    * ``supports_chat`` / ``supports_image_input`` / ``supports_tools`` (Fireworks, Hyperbolic)
    * ``capabilities`` object (Mistral: ``completion_chat``, ``function_calling``, ``vision`` …)
    * ``input_modalities`` / ``output_modalities`` (Groq, DeepSeek, OpenRouter ``architecture``)
    * ``type`` / ``model_type`` (Together, Novita)
    * ``supported_parameters`` / ``supported_features`` / ``features`` (OpenRouter, Groq, Novita)
    * ``metadata.tags`` (DeepInfra)

    Records without any of these fields yield ``{}``, so name-regex and
    provider defaults still apply to bare OpenAI-style listings.
    """

    def infer(
        self,
        provider: str,
        model_id: str,
        model_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        d = model_data or {}
        result: Dict[str, Any] = {}

        # Explicit booleans (Fireworks, Hyperbolic)
        for field, cap in (("supports_chat", "chat"),
                           ("supports_image_input", "vision"),
                           ("supports_tools", "function_calling")):
            if isinstance(d.get(field), bool):
                result[cap] = d[field]

        # Mistral-style capability object
        caps = d.get("capabilities")
        if isinstance(caps, dict) and caps and all(isinstance(v, bool) for v in caps.values()):
            mapping = {
                "completion_chat": "chat",
                "function_calling": "function_calling",
                "vision": "vision",
                "reasoning": "reasoning",
                "audio_transcription": "audio_input",
                "audio_speech": "audio_output",
            }
            for field, cap in mapping.items():
                if field in caps:
                    result[cap] = caps[field]
            if caps.get("audio"):
                result["audio_input"] = True

        # Modalities (Groq, DeepSeek, OpenRouter architecture)
        arch = d.get("architecture") if isinstance(d.get("architecture"), dict) else {}
        inputs = _as_list(d.get("input_modalities")) or _as_list(arch.get("input_modalities"))
        outputs = _as_list(d.get("output_modalities")) or _as_list(arch.get("output_modalities"))
        if inputs:
            result["vision"] = "image" in inputs
            result["audio_input"] = "audio" in inputs
            if "text" not in inputs:
                # Audio-only input (e.g. Whisper on Groq): transcription, not chat.
                result["chat"] = False
        if outputs:
            # Groq labels TTS output "speech"; OpenRouter/DeepSeek use "audio".
            result["audio_output"] = "audio" in outputs or "speech" in outputs
            result["image_output"] = "image" in outputs
            if "text" not in outputs:
                result["chat"] = False

        # Declared feature lists
        features = (_as_list(d.get("supported_parameters"))
                    + _as_list(d.get("supported_features"))
                    + _as_list(d.get("features")))
        if features:
            result["function_calling"] = any(f in ("tools", "function-calling", "function_calling")
                                             for f in features)
            if any(f in ("reasoning", "include_reasoning", "reasoning_effort") for f in features):
                result["reasoning"] = True

        # DeepInfra tags
        metadata = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        tags = _as_list(metadata.get("tags"))
        if tags:
            if "vision" in tags or "vlm" in tags:
                result["vision"] = True
            if any(t.startswith("reasoning") for t in tags):
                result["reasoning"] = True
            if "embeddings" in tags or "embedding" in tags:
                result["embedding"] = True
                result["chat"] = False

        # Model type (Together ``type``, Novita ``model_type``)
        model_type = str(d.get("type") or d.get("model_type") or "").lower()
        if model_type in _CHAT_MODEL_TYPES:
            result.setdefault("chat", True)
        elif model_type in _NON_CHAT_MODEL_TYPES:
            result["chat"] = False
            if model_type in ("embedding", "embeddings"):
                result["embedding"] = True
            elif model_type in ("audio", "tts"):
                result["audio_output"] = True
            elif model_type in ("transcribe", "transcription"):
                result["audio_input"] = True
            elif model_type == "image":
                result["image_output"] = True

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
        if all_values.get("chat") is False:
            # Tool calling is a chat-completions feature; provider-wide chat
            # defaults must not leak onto embedding/transcription/TTS models.
            all_values["function_calling"] = False
            if all_values.get("embedding"):
                all_values["streaming"] = False
        return ModelCapabilities(**all_values)
