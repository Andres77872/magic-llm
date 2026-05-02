"""Capability inference strategy tests.

Tests:
- ``CapabilityInferenceStrategy`` protocol conformance
- ``ModelNameRegexStrategy``: known model IDs produce expected dicts; unknown
  model IDs produce empty dict; regression snapshot against known model IDs
- ``ProviderDefaultsStrategy``: each known provider returns expected defaults
- ``ProviderFieldStrategy``: Anthropic, Cohere, Google, OpenRouter field parsing
- ``CompositeCapabilityInference``: tiered merge priority
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from magic_llm.engine.discovery.capabilities import (
    CapabilityInferenceStrategy,
    CompositeCapabilityInference,
    ModelNameRegexStrategy,
    ProviderDefaultsStrategy,
    ProviderFieldStrategy,
)
from magic_llm.engine.discovery.capabilities.strategies import (
    VISION_PATTERNS,
    EMBEDDING_PATTERNS,
    FUNCTION_CALLING_PATTERNS,
)
from magic_llm.model.discovery import ModelCapabilities


# =============================================================================
# Protocol Conformance
# =============================================================================

class TestStrategyProtocol:
    """All concrete strategies implement ``infer()`` with correct signature."""

    @pytest.mark.parametrize("strategy_cls", [
        ModelNameRegexStrategy,
        ProviderDefaultsStrategy,
        ProviderFieldStrategy,
        CompositeCapabilityInference,
    ])
    def test_concrete_strategy_has_infer(self, strategy_cls):
        st = strategy_cls() if strategy_cls is not CompositeCapabilityInference else strategy_cls([])
        assert hasattr(st, "infer")
        assert callable(st.infer)

    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            CapabilityInferenceStrategy()

    def test_infer_returns_dict(self):
        st = ModelNameRegexStrategy()
        result = st.infer("openai", "gpt-4o", {"id": "gpt-4o"})
        assert isinstance(result, dict)


# =============================================================================
# ModelNameRegexStrategy — Regression Snapshot
# =============================================================================

class TestModelNameRegexStrategy:
    """Regex-based capability inference with regression snapshot."""

    # ── Snapshot fixture: known model IDs → expected capability fields ────
    # These represent the canonical behaviour extracted from the old
    # ``OpenAICompatibleAdapter._infer_*`` methods and should be preserved
    # by ``ModelNameRegexStrategy``.

    SNAPSHOT: Dict[str, Dict[str, Any]] = {
        # Vision models — match VISION_PATTERNS (gpt-4o, gpt-4-turbo, gpt-4-vision-preview)
        # Also match FUNCTION_CALLING_PATTERNS (r"gpt-4" and r"gpt-3.5-turbo")
        "gpt-4o": {"vision": True, "function_calling": True},
        "gpt-4o-2024-08-06": {"vision": True, "function_calling": True},
        "gpt-4o-mini": {"vision": True, "function_calling": True},
        "gpt-4-turbo": {"vision": True, "function_calling": True},
        "gpt-4-vision-preview": {"vision": True, "function_calling": True},
        "gpt-4-1106-vision-preview": {"vision": True, "function_calling": True},
        # Chat models (no vision, but match function_calling via r"gpt-4" or r"gpt-3.5-turbo")
        "gpt-4": {"function_calling": True},
        "gpt-4-32k": {"function_calling": True},
        "gpt-3.5-turbo": {"function_calling": True},
        "gpt-3.5-turbo-0125": {"function_calling": True},
        # Embedding models
        "text-embedding-3-small": {"embedding": True, "chat": False},
        "text-embedding-3-large": {"embedding": True, "chat": False},
        "text-embedding-ada-002": {"embedding": True, "chat": False},
        # Function-calling models (already covered by gpt-4/gpt-3.5-turbo matches)
        "gpt-4-0613": {"function_calling": True},
        "gpt-3.5-turbo-0613": {"function_calling": True},
        # Claude 3 — matches VISION_PATTERNS via r"claude-3" AND FUNCTION_CALLING via r"claude"
        "claude-3-5-sonnet-20241022": {"vision": True, "function_calling": True},
        "claude-3-haiku-20240307": {"vision": True, "function_calling": True},
        # Gemini — matches VISION_PATTERNS via r"gemini" (no function_calling match)
        "models/gemini-1.5-pro": {"vision": True},
        "gemini-1.5-flash": {"vision": True},
        # DeepSeek
        "deepseek-chat": {},
        "deepseek-coder": {},
        # Llama models
        "Meta-Llama-3.1-8B-Instruct": {},
        "Meta-Llama-3.1-70B-Instruct": {},
        # Mistral
        "mistral-large-latest": {},
        "mistral-small-latest": {},
        # Grok
        "grok-2": {},
        "grok-2-mini": {},
    }

    def test_known_model_ids_match_snapshot(self):
        """Every known model ID produces the expected capability overrides."""
        strategy = ModelNameRegexStrategy()
        for model_id, expected in self.SNAPSHOT.items():
            result = strategy.infer("openai", model_id, {"id": model_id})
            assert result == expected, (
                f"ModelNameRegexStrategy('{model_id}'): "
                f"expected {expected}, got {result}"
            )

    def test_unknown_model_id_returns_empty(self):
        """Unknown/random model IDs produce no overrides."""
        strategy = ModelNameRegexStrategy()
        for model_id in ["completely-unknown-model", "", "custom-llm-v1"]:
            result = strategy.infer("openai", model_id, {"id": model_id})
            assert result == {}, f"Unknown model '{model_id}' produced {result}"

    def test_all_patterns_match_at_least_one_model(self):
        """Every pattern in the tables matches at least one known model."""
        strategy = ModelNameRegexStrategy()
        import re
        for pat_name, patterns in [
            ("VISION", VISION_PATTERNS),
            ("EMBEDDING", EMBEDDING_PATTERNS),
            ("FUNCTION_CALLING", FUNCTION_CALLING_PATTERNS),
        ]:
            for pattern in patterns:
                matches = any(re.search(pattern, mid, re.IGNORECASE) for mid in self.SNAPSHOT)
                assert matches, (
                    f"{pat_name} pattern '{pattern}' matches NO model in snapshot"
                )


# =============================================================================
# ProviderDefaultsStrategy
# =============================================================================

class TestProviderDefaultsStrategy:
    def test_openai_defaults(self):
        st = ProviderDefaultsStrategy()
        result = st.infer("openai", "", {})
        assert result == {"chat": True, "streaming": True, "function_calling": True}

    def test_anthropic_defaults(self):
        st = ProviderDefaultsStrategy()
        result = st.infer("anthropic", "", {})
        assert result == {"chat": True, "streaming": True, "function_calling": True}

    def test_google_defaults(self):
        st = ProviderDefaultsStrategy()
        result = st.infer("google", "", {})
        assert result == {"chat": True, "streaming": True}

    def test_cohere_defaults(self):
        st = ProviderDefaultsStrategy()
        result = st.infer("cohere", "", {})
        assert result == {"chat": True, "streaming": True}

    def test_azure_defaults(self):
        st = ProviderDefaultsStrategy()
        result = st.infer("azure", "", {})
        assert result == {"chat": True, "streaming": True}

    def test_unknown_provider_returns_empty(self):
        st = ProviderDefaultsStrategy()
        result = st.infer("definitely-not-real", "", {})
        assert result == {}

    def test_all_providers_defined(self):
        """All registered engines have defaults (except azure-speech which has no adapter)."""
        from magic_llm.engine.discovery import list_supported_engines
        st = ProviderDefaultsStrategy()
        for engine in list_supported_engines():
            result = st.infer(engine, "", {})
            assert isinstance(result, dict), f"Provider '{engine}' returned {type(result)}"


# =============================================================================
# ProviderFieldStrategy
# =============================================================================

class TestProviderFieldStrategy:
    def test_anthropic_vision_and_reasoning(self):
        st = ProviderFieldStrategy()
        result = st.infer("anthropic", "claude-3-5-sonnet", {
            "capabilities": {
                "image_input": {"supported": True},
                "thinking": {"supported": False},
            },
        })
        assert result["vision"] is True
        assert result["reasoning"] is False

    def test_anthropic_no_capabilities(self):
        st = ProviderFieldStrategy()
        result = st.infer("anthropic", "claude-3-opus", {})
        # Lambdas return False for missing nested keys (dict.get default), so
        # the strategy returns the fields it tried but with falsy values.
        # This is expected behaviour — the composite merge honours Tier 1 fields
        # even when False, preventing Tier 2/3 from incorrectly setting True.
        assert result.get("vision") is False
        assert result.get("reasoning") is False

    def test_cohere_chat_and_function_calling(self):
        st = ProviderFieldStrategy()
        result = st.infer("cohere", "command-r", {
            "endpoints": ["chat"],
            "features": ["tool_use"],
        })
        assert result["chat"] is True
        assert result["function_calling"] is True
        # embedding lambda runs but returns False (not in endpoints)
        assert result["embedding"] is False

    def test_cohere_embedding(self):
        st = ProviderFieldStrategy()
        result = st.infer("cohere", "embed-english-v3.0", {
            "endpoints": ["embed"],
            "features": [],
        })
        assert result["embedding"] is True
        # chat lambda runs but returns False (not in endpoints);
        # function_calling lambda runs but returns False (not in features)
        assert result["chat"] is False
        assert result["function_calling"] is False

    def test_google_chat_using_supported_generation_methods(self):
        st = ProviderFieldStrategy()
        result = st.infer("google", "gemini-1.5-pro", {
            "supportedGenerationMethods": ["generateContent", "countTokens"],
        })
        assert result["chat"] is True

    def test_google_embedding(self):
        st = ProviderFieldStrategy()
        result = st.infer("google", "text-embedding-004", {
            "supportedGenerationMethods": ["embedContent"],
        })
        assert result["embedding"] is True

    def test_openrouter_vision_modality(self):
        st = ProviderFieldStrategy()
        result = st.infer("openrouter", "openai/gpt-4o", {
            "architecture": {
                "modality": {"input": ["text", "image"], "output": ["text"]},
            },
        })
        assert result["vision"] is True

    def test_openrouter_audio_modality(self):
        st = ProviderFieldStrategy()
        result = st.infer("openrouter", "openai/gpt-4o-audio", {
            "architecture": {
                "modality": {"input": ["text", "audio"], "output": ["text"]},
            },
        })
        assert result["audio_input"] is True

    def test_unknown_provider_returns_empty(self):
        st = ProviderFieldStrategy()
        result = st.infer("definitely-not-real", "", {})
        assert result == {}


# =============================================================================
# CompositeCapabilityInference
# =============================================================================

class TestCompositeCapabilityInference:
    def test_tier1_overrides_tier2(self):
        """Highest-priority strategy (first in list) is final override."""
        tier2 = MagicMock(spec=CapabilityInferenceStrategy)
        tier2.infer.return_value = {"vision": True, "chat": True}
        tier1 = MagicMock(spec=CapabilityInferenceStrategy)
        tier1.infer.return_value = {"vision": False}  # override

        composite = CompositeCapabilityInference([tier1, tier2])
        result = composite.infer("test", "test-model", {})
        assert result.vision is False   # Tier 1 wins
        assert result.chat is True       # Tier 2 baseline

    def test_tier2_overrides_tier3(self):
        """Middle strategy overrides baseline."""
        tier3 = MagicMock(spec=CapabilityInferenceStrategy)
        tier3.infer.return_value = {"chat": True}
        tier2 = MagicMock(spec=CapabilityInferenceStrategy)
        tier2.infer.return_value = {"chat": False}
        tier1 = MagicMock(spec=CapabilityInferenceStrategy)
        tier1.infer.return_value = {}  # no override

        composite = CompositeCapabilityInference([tier1, tier2, tier3])
        result = composite.infer("test", "test-model", {})
        assert result.chat is False  # Tier 2 wins over Tier 3

    def test_empty_strategies_list(self):
        """No strategies → all ModelCapabilities defaults."""
        composite = CompositeCapabilityInference([])
        result = composite.infer("test", "test-model", {})
        assert result.chat is True   # default
        assert result.embedding is False  # default

    def test_strategies_that_return_nothing(self):
        """Strategies that return empty dict don't override."""
        tier1 = MagicMock(spec=CapabilityInferenceStrategy)
        tier1.infer.return_value = {}
        tier2 = MagicMock(spec=CapabilityInferenceStrategy)
        tier2.infer.return_value = {}

        composite = CompositeCapabilityInference([tier1, tier2])
        result = composite.infer("test", "test-model", {})
        assert result.chat is True  # default, not overridden

    def test_three_tier_realistic_merge(self):
        """Realistic: ProviderDefaults (T3) + ModelNameRegex (T2) + ProviderField (T1)."""
        composite = CompositeCapabilityInference([
            ProviderFieldStrategy(),      # T1: API fields
            ModelNameRegexStrategy(),      # T2: regex on name
            ProviderDefaultsStrategy(),    # T3: provider baseline
        ])
        # Anthropic Claude with vision capabilities
        result = composite.infer("anthropic", "claude-3-5-sonnet-20241022", {
            "capabilities": {
                "image_input": {"supported": True},
                "thinking": {"supported": False},
            },
        })
        assert result.chat is True       # ProviderDefaults for anthropic
        assert result.streaming is True   # ProviderDefaults for anthropic
        assert result.vision is True      # ProviderFieldStrategy overrides regex
        assert result.reasoning is False  # ProviderFieldStrategy
        assert result.function_calling is True  # ModelNameRegex + ProviderDefaults


# =============================================================================
# Pydantic Model Verification (Dead Field Removal)
# =============================================================================

class TestModelVerification:
    """Verify removed fields are truly gone from the Pydantic models."""

    def test_normalized_model_without_dead_fields_succeeds(self):
        from magic_llm.model.discovery import NormalizedDiscoveredModel
        m = NormalizedDiscoveredModel(external_id="test", provider="openai")
        assert m.external_id == "test"

    def test_normalized_model_rejects_removed_fields(self):
        """Spec: removed fields MUST raise ValidationError (strict rejection)."""
        from pydantic import ValidationError
        from magic_llm.model.discovery import NormalizedDiscoveredModel
        with pytest.raises(ValidationError):
            NormalizedDiscoveredModel(
                external_id="test",
                provider="openai",
                lifecycle_status="active",  # removed — MUST reject
            )
        with pytest.raises(ValidationError):
            NormalizedDiscoveredModel(
                external_id="test",
                provider="openai",
                requires_enablement=False,  # removed — MUST reject
            )

    def test_capabilities_without_fine_tune_succeeds(self):
        from magic_llm.model.discovery import ModelCapabilities
        c = ModelCapabilities(chat=True, vision=True)
        assert c.chat is True
        assert c.vision is True

    def test_capabilities_rejects_removed_fields(self):
        """Spec: removed capability fields MUST raise ValidationError."""
        from pydantic import ValidationError
        from magic_llm.model.discovery import ModelCapabilities
        with pytest.raises(ValidationError):
            ModelCapabilities(fine_tune=True)  # removed — MUST reject
        with pytest.raises(ValidationError):
            ModelCapabilities(code=True)  # removed — MUST reject

    def test_capabilities_fine_tune_not_in_fields(self):
        from magic_llm.model.discovery import ModelCapabilities
        assert "fine_tune" not in ModelCapabilities.model_fields
        assert "code" not in ModelCapabilities.model_fields

    def test_normalized_model_dead_fields_not_in_fields(self):
        from magic_llm.model.discovery import NormalizedDiscoveredModel
        assert "lifecycle_status" not in NormalizedDiscoveredModel.model_fields
        assert "requires_enablement" not in NormalizedDiscoveredModel.model_fields
