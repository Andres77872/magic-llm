"""Unit tests for TokenAliasResolver — centralized alias resolution.

Tests cover:
- Default alias chains for all three fields
- Nested path resolution (dot notation)
- Custom alias override (complete replacement)
- Partial alias prefix (appends remaining defaults)
- Heuristic sentinel dispatch
- No match returns None
- Edge cases: empty raw model, missing keys, None values
"""

from typing import Dict, Any, Optional

import pytest

from magic_llm.engine.discovery.alias_resolver import TokenAliasResolver


# =========================================================================
# Default Chain Resolution
# =========================================================================

class TestDefaultChains:
    """Default alias chains resolve all three fields correctly."""

    def test_context_window_resolved(self):
        """Default chain resolves context_window from 'context_window' field."""
        raw = {"context_window": 128000, "context_length": 32000}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 128000  # First alias wins

    def test_context_window_falls_back_to_context_length(self):
        """Default chain falls back to context_length when context_window missing."""
        raw = {"context_length": 64000}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 64000

    def test_context_window_falls_back_to_max_context(self):
        """Default chain falls back to max_context."""
        raw = {"max_context": 32000}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 32000

    def test_context_window_nested_limits_context(self):
        """Default chain resolves nested limits.context."""
        raw = {"limits": {"context": 128000}}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 128000

    def test_context_window_falls_back_to_max_input_tokens(self):
        """Default chain uses max_input_tokens as input-limit proxy."""
        raw = {"max_input_tokens": 200000}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 200000

    def test_context_window_falls_back_to_input_token_limit(self):
        """Default chain uses inputTokenLimit as Google-style fallback."""
        raw = {"inputTokenLimit": 1000000}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 1000000

    def test_context_window_falls_back_to_max_prompt_tokens(self):
        """Default chain uses max_prompt_tokens as rare fallback."""
        raw = {"max_prompt_tokens": 16000}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 16000

    def test_max_input_tokens_resolved(self):
        """Default chain resolves max_input_tokens."""
        raw = {"max_input_tokens": 200000}
        val = TokenAliasResolver.resolve(raw, "max_input_tokens")
        assert val == 200000

    def test_max_input_tokens_falls_back_to_input_token_limit(self):
        """Default chain falls back to inputTokenLimit."""
        raw = {"inputTokenLimit": 1000000}
        val = TokenAliasResolver.resolve(raw, "max_input_tokens")
        assert val == 1000000

    def test_max_output_tokens_resolved(self):
        """Default chain resolves max_output_tokens."""
        raw = {"max_output_tokens": 8192, "max_tokens": 4096}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens")
        assert val == 8192  # First alias wins

    def test_max_output_tokens_falls_back_to_max_tokens(self):
        """Default chain falls back to max_tokens."""
        raw = {"max_tokens": 4096}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens")
        assert val == 4096

    def test_max_output_tokens_falls_back_to_max_completion_tokens(self):
        """Default chain falls back to max_completion_tokens."""
        raw = {"max_completion_tokens": 16384}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens")
        assert val == 16384

    def test_max_output_tokens_falls_back_to_top_provider_nested(self):
        """Default chain resolves nested top_provider.max_completion_tokens."""
        raw = {"top_provider": {"max_completion_tokens": 16384}}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens")
        assert val == 16384

    def test_max_output_tokens_falls_back_to_output_token_limit(self):
        """Default chain falls back to outputTokenLimit."""
        raw = {"outputTokenLimit": 8192}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens")
        assert val == 8192

    def test_priority_order_context_window_wins(self):
        """context_window wins over all other aliases."""
        raw = {
            "context_window": 128000,
            "context_length": 64000,
            "max_context": 32000,
            "max_input_tokens": 200000,
        }
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 128000

    def test_priority_order_context_length_wins(self):
        """context_length wins when context_window is missing."""
        raw = {
            "context_length": 64000,
            "max_context": 32000,
            "max_input_tokens": 200000,
        }
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val == 64000


# =========================================================================
# Nested Path Resolution
# =========================================================================

class TestNestedPath:
    """Nested paths resolve correctly."""

    def test_limits_context(self):
        """limits.context resolves from nested dict."""
        raw = {"limits": {"context": 128000}}
        val = TokenAliasResolver.resolve(raw, "context_window",
                                          aliases=["limits.context"])
        assert val == 128000

    def test_top_provider_max_completion_tokens(self):
        """top_provider.max_completion_tokens resolves from nested dict."""
        raw = {"top_provider": {"max_completion_tokens": 16384}}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens",
                                          aliases=["top_provider.max_completion_tokens"])
        assert val == 16384

    def test_nested_path_missing_key_returns_none(self):
        """Missing intermediate key returns None without error."""
        raw = {"limits": {}}
        val = TokenAliasResolver._get_nested(raw, "limits.context")
        assert val is None

    def test_nested_path_missing_top_key(self):
        """Missing top-level key returns None."""
        raw = {}
        val = TokenAliasResolver._get_nested(raw, "limits.context")
        assert val is None

    def test_nested_path_non_dict_intermediate(self):
        """Non-dict intermediate returns None without crash."""
        raw = {"limits": "not-a-dict"}
        val = TokenAliasResolver._get_nested(raw, "limits.context")
        assert val is None

    def test_nested_path_deeply_nested(self):
        """Arbitrary depth traversal works."""
        raw = {"a": {"b": {"c": {"d": 42}}}}
        val = TokenAliasResolver._get_nested(raw, "a.b.c.d")
        assert val == 42


# =========================================================================
# Custom Alias Override
# =========================================================================

class TestCustomAliasOverride:
    """Custom alias overrides work correctly."""

    def test_full_custom_chain(self):
        """Complete custom alias chain replaces default."""
        raw = {"foo": 64000}
        val = TokenAliasResolver.resolve(raw, "context_window",
                                          aliases=["foo"])
        assert val == 64000

    def test_custom_chain_ignores_defaults(self):
        """Custom chain does not fall back to defaults when aliases given."""
        raw = {"context_window": 128000, "foo": 64000}
        val = TokenAliasResolver.resolve(raw, "context_window",
                                          aliases=["foo"])
        assert val == 64000  # Only probes "foo", not "context_window"

    def test_custom_chain_all_none(self):
        """Custom chain with only non-matching aliases still falls back to defaults."""
        raw = {"context_window": 128000}
        val = TokenAliasResolver.resolve(raw, "context_window",
                                          aliases=["foo", "bar"])
        # "foo" → None, "bar" → None, then default chain probes context_window → 128000
        assert val == 128000

    def test_google_style_alias(self):
        """Google-style: inputTokenLimit only for context_window."""
        raw = {"inputTokenLimit": 1000000, "context_window": 128000}
        val = TokenAliasResolver.resolve(raw, "context_window",
                                          aliases=["inputTokenLimit"])
        assert val == 1000000  # Custom prefix wins


# =========================================================================
# Partial Alias Prefix (Appends Remaining Defaults)
# =========================================================================

class TestPartialAliasPrefix:
    """Partial alias prefixes append remaining default entries."""

    def test_openrouter_style_prefix(self):
        """OpenRouter-style: context_length before context_window, rest from defaults."""
        raw = {"context_length": 64000, "context_window": 128000}
        # Custom prefix: context_length first, then context_window
        aliases = ["context_length", "context_window"]
        val = TokenAliasResolver.resolve(raw, "context_window", aliases=aliases)
        assert val == 64000  # context_length wins

    def test_openrouter_no_context_length(self):
        """OpenRouter-style: context_window fallback works."""
        raw = {"context_window": 64000}
        aliases = ["context_length", "context_window"]
        val = TokenAliasResolver.resolve(raw, "context_window", aliases=aliases)
        assert val == 64000  # context_window is second alias

    def test_azure_foundry_style_prefix(self):
        """Azure Foundry-style: limits.context first, then defaults."""
        raw = {"limits": {"context": 128000}, "context_window": 64000}
        aliases = ["limits.context", "context_window", "context_length", "max_context"]
        val = TokenAliasResolver.resolve(raw, "context_window", aliases=aliases)
        assert val == 128000  # limits.context wins

    def test_azure_foundry_no_nested(self):
        """Azure Foundry-style: falls through nested to top-level."""
        raw = {"context_window": 64000}
        aliases = ["limits.context", "context_window", "context_length", "max_context"]
        val = TokenAliasResolver.resolve(raw, "context_window", aliases=aliases)
        assert val == 64000  # context_window is second alias

    def test_partial_prefix_appends_missing_defaults(self):
        """Partial prefix appends defaults not in the custom list."""
        raw = {"max_input_tokens": 200000}  # Not in prefix, IS in default
        aliases = ["context_window"]
        val = TokenAliasResolver.resolve(raw, "context_window", aliases=aliases)
        # context_window → None, then default chain: context_length → max_context → limits.context → max_input_tokens
        assert val == 200000  # max_input_tokens from default chain

    def test_partial_prefix_google_alias(self):
        """Google-style: inputTokenLimit prefix, rest from defaults."""
        raw = {"inputTokenLimit": 1000000}
        aliases = ["inputTokenLimit"]
        val = TokenAliasResolver.resolve(raw, "context_window", aliases=aliases)
        assert val == 1000000

    def test_build_chain_appends_defaults(self):
        """_build_chain correctly appends remaining defaults."""
        chain = TokenAliasResolver._build_chain("context_window",
                                                 aliases=["a", "b"])
        assert chain[:2] == ["a", "b"]
        # Remaining entries from default
        defaults = TokenAliasResolver.DEFAULT_ALIASES["context_window"]
        for d in defaults:
            assert d in chain, f"{d} should be in chain"
        assert chain[0] == "a"
        assert chain[1] == "b"
        # a and b are not in defaults, so all defaults should be appended
        assert len(chain) == 2 + len(defaults)


# =========================================================================
# Heuristic Sentinel Dispatch
# =========================================================================

class TestHeuristicDispatch:
    """Heuristic sentinels dispatch to registered callables."""

    @staticmethod
    def _mock_heuristic(raw_model):
        model_id = raw_model.get("id", "")
        if "claude" in model_id:
            return 200000
        return None

    def test_heuristic_sentinel_dispatches(self):
        """Heuristic sentinel calls the registered callable."""
        raw = {"id": "claude-3-5-sonnet"}
        val = TokenAliasResolver.resolve(
            raw, "context_window",
            aliases=["__heuristic_claude__"],
            heuristic_registry={"claude": self._mock_heuristic},
        )
        assert val == 200000

    def test_heuristic_sentinel_no_match(self):
        """Heuristic returns None when callable returns None."""
        raw = {"id": "gpt-4"}
        val = TokenAliasResolver.resolve(
            raw, "context_window",
            aliases=["__heuristic_claude__"],
            heuristic_registry={"claude": self._mock_heuristic},
        )
        assert val is None

    def test_heuristic_sentinel_unregistered(self):
        """Unregistered heuristic sentinel returns None."""
        raw = {"id": "claude-3-5-sonnet"}
        val = TokenAliasResolver.resolve(
            raw, "context_window",
            aliases=["__heuristic_unknown__"],
            heuristic_registry={},
        )
        assert val is None

    def test_heuristic_after_alias_chain(self):
        """Heuristic fires only when alias chain returns None."""
        raw = {"id": "claude-3-5-sonnet", "context_window": 50000}
        val = TokenAliasResolver.resolve(
            raw, "context_window",
            aliases=["context_window", "__heuristic_claude__"],
            heuristic_registry={"claude": self._mock_heuristic},
        )
        assert val == 50000  # Alias wins, heuristic not invoked


# =========================================================================
# No Match / Edge Cases
# =========================================================================

class TestNoMatch:
    """Returns None when no alias matches."""

    def test_no_match_returns_none(self):
        """Empty raw model returns None."""
        raw = {}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val is None

    def test_no_match_for_max_input_tokens(self):
        """Missing max_input_tokens field returns None."""
        raw = {"id": "some-model"}
        val = TokenAliasResolver.resolve(raw, "max_input_tokens")
        assert val is None

    def test_no_match_for_max_output_tokens(self):
        """Missing all output fields returns None."""
        raw = {"id": "some-model"}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens")
        assert val is None

    def test_none_values_are_skipped(self):
        """Fields with None values are treated as missing."""
        raw = {"context_window": None, "context_length": None}
        val = TokenAliasResolver.resolve(raw, "context_window")
        assert val is None

    def test_zero_values_are_returned(self):
        """Zero int values ARE returned (not falsy but counted)."""
        raw = {"max_output_tokens": 0}
        val = TokenAliasResolver.resolve(raw, "max_output_tokens")
        assert val == 0

    def test_empty_alias_list_builds_empty_chain(self):
        """Empty alias list with no defaults (unknown field) returns None."""
        chain = TokenAliasResolver._build_chain("nonexistent_field", aliases=[])
        assert chain == []

    def test_none_aliases_uses_defaults(self):
        """None alias list uses default chain."""
        raw = {"context_window": 128000}
        val = TokenAliasResolver.resolve(raw, "context_window", aliases=None)
        assert val == 128000

    def test_unknown_field_type_uses_empty_defaults(self):
        """Unknown field type returns empty defaults and None."""
        raw = {"foo": 42}
        val = TokenAliasResolver.resolve(raw, "nonexistent_field")
        assert val is None

    def test_invalid_field_type_no_crash(self):
        """Invalid field type does not crash."""
        raw = {}
        val = TokenAliasResolver.resolve(raw, "")
        assert val is None


# =========================================================================
# Heuristic Hooks — Unit Tests
# =========================================================================

class TestClaudeNameHeuristic:
    """Claude-name heuristic in isolation."""

    def test_claude_3_5_sonnet(self):
        from magic_llm.engine.discovery.anthropic_discovery import (
            AnthropicDiscoveryAdapter,
        )
        raw = {"id": "claude-3-5-sonnet-20241022"}
        val = AnthropicDiscoveryAdapter._claude_name_heuristic(raw)
        assert val == 200000

    def test_claude_3_opus(self):
        from magic_llm.engine.discovery.anthropic_discovery import (
            AnthropicDiscoveryAdapter,
        )
        raw = {"id": "claude-3-opus-20240229"}
        val = AnthropicDiscoveryAdapter._claude_name_heuristic(raw)
        assert val == 200000

    def test_claude_3_dot_5_sonnet(self):
        from magic_llm.engine.discovery.anthropic_discovery import (
            AnthropicDiscoveryAdapter,
        )
        raw = {"id": "claude-3.5-sonnet"}
        val = AnthropicDiscoveryAdapter._claude_name_heuristic(raw)
        assert val == 200000

    def test_claude_2(self):
        from magic_llm.engine.discovery.anthropic_discovery import (
            AnthropicDiscoveryAdapter,
        )
        raw = {"id": "claude-2.1"}
        val = AnthropicDiscoveryAdapter._claude_name_heuristic(raw)
        assert val == 100000

    def test_claude_4_no_heuristic_match(self):
        """Claude 4.x models don't match Claude-3/Claude-2 heuristic."""
        from magic_llm.engine.discovery.anthropic_discovery import (
            AnthropicDiscoveryAdapter,
        )
        raw = {"id": "claude-opus-4-20250514"}
        val = AnthropicDiscoveryAdapter._claude_name_heuristic(raw)
        assert val is None  # Alias chain will pick up max_input_tokens

    def test_non_claude_model(self):
        from magic_llm.engine.discovery.anthropic_discovery import (
            AnthropicDiscoveryAdapter,
        )
        raw = {"id": "gpt-4o"}
        val = AnthropicDiscoveryAdapter._claude_name_heuristic(raw)
        assert val is None


class TestLlamaNameHeuristic:
    """Llama-name heuristic in isolation."""

    def test_llama_3_70b(self):
        from magic_llm.engine.discovery.sambanova_discovery import (
            SambaNovaDiscoveryAdapter,
        )
        raw = {"id": "Meta-Llama-3.1-70B-Instruct"}
        val = SambaNovaDiscoveryAdapter._llama_name_heuristic(raw)
        assert val == 128000

    def test_llama_3_8b(self):
        from magic_llm.engine.discovery.sambanova_discovery import (
            SambaNovaDiscoveryAdapter,
        )
        raw = {"id": "Meta-Llama-3.1-8B-Instruct"}
        val = SambaNovaDiscoveryAdapter._llama_name_heuristic(raw)
        assert val == 8192

    def test_llama_3_405b(self):
        from magic_llm.engine.discovery.sambanova_discovery import (
            SambaNovaDiscoveryAdapter,
        )
        raw = {"id": "Llama-3.1-405B-Instruct"}
        val = SambaNovaDiscoveryAdapter._llama_name_heuristic(raw)
        assert val == 128000

    def test_llama_2(self):
        """Llama 2 does NOT match Llama-3 heuristic."""
        from magic_llm.engine.discovery.sambanova_discovery import (
            SambaNovaDiscoveryAdapter,
        )
        raw = {"id": "Llama-2-70b"}
        val = SambaNovaDiscoveryAdapter._llama_name_heuristic(raw)
        assert val is None  # Only matches llama-3/llama3 patterns

    def test_non_llama_model(self):
        from magic_llm.engine.discovery.sambanova_discovery import (
            SambaNovaDiscoveryAdapter,
        )
        raw = {"id": "gpt-4o"}
        val = SambaNovaDiscoveryAdapter._llama_name_heuristic(raw)
        assert val is None


class TestContextWindowMapFallback:
    """CONTEXT_WINDOW_MAP fallback in isolation."""

    def test_gpt_4o_matches(self):
        from magic_llm.engine.discovery.openai_compatible.base import (
            OpenAICompatibleAdapter,
        )
        raw = {"id": "gpt-4o-2024-08-06"}
        val = OpenAICompatibleAdapter._context_window_map_fallback(raw)
        assert val is not None
        assert isinstance(val, int)
        assert val > 0

    def test_custom_model_no_match(self):
        from magic_llm.engine.discovery.openai_compatible.base import (
            OpenAICompatibleAdapter,
        )
        raw = {"id": "custom-model-v1"}
        val = OpenAICompatibleAdapter._context_window_map_fallback(raw)
        assert val is None

    def test_no_id_field(self):
        from magic_llm.engine.discovery.openai_compatible.base import (
            OpenAICompatibleAdapter,
        )
        raw = {}
        val = OpenAICompatibleAdapter._context_window_map_fallback(raw)
        assert val is None
