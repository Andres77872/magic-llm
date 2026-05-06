"""Centralized token field alias resolver for provider discovery.

Resolves a semantic field type (context_window, max_input_tokens,
max_output_tokens) from a raw provider API dict using an ordered
alias chain. Supports nested paths via dot notation and heuristic
sentinel dispatch.

Usage::

    from magic_llm.engine.discovery.alias_resolver import TokenAliasResolver

    # Use default alias chain
    val = TokenAliasResolver.resolve(raw_model, "context_window")

    # Use custom alias prefix (appends remaining defaults)
    val = TokenAliasResolver.resolve(raw_model, "context_window",
                                      aliases=["inputTokenLimit"])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class TokenAliasResolver:
    """Centralized resolver for provider token field aliases.

    Resolves a semantic field type from raw provider data using an
    ordered alias chain.  Supports:

    * **Flat keys**: ``raw_model.get("context_window")``
    * **Nested paths** (dot notation): ``raw_model.get("limits", {}).get("context")``
    * **Heuristic sentinels** (``__heuristic_*__``): dispatches to a
      registry of heuristic callables

    Default alias chains encode the priority defined by the unified
    contract: explicit context fields → input-limit fallbacks →
    heuristic → ``None``.
    """

    # ── Default alias chains ──────────────────────────────────────────
    #
    # Ordered by priority — first non-None value wins.
    # Each entry is either:
    #   * str — flat key lookup via raw_model.get(key)
    #   * str with dots — nested path via safe traversal
    #   * str starting with "__heuristic__" — heuristic sentinel

    DEFAULT_ALIASES: Dict[str, List[str]] = {
        # Explicit context fields first, then input-limit fallbacks
        "context_window": [
            "context_window",          # explicit combined window (OpenAI, Groq, etc.)
            "context_length",          # alternate explicit field (OpenRouter, Together, Cohere)
            "max_context",             # Azure OpenAI alternate
            "limits.context",          # Azure Foundry nested path
            "max_input_tokens",        # input-limit proxy (Anthropic, hybrid providers)
            "inputTokenLimit",         # Google camelCase input limit
            "max_prompt_tokens",       # alternate input-limit field (rare)
        ],
        "max_input_tokens": [
            "max_input_tokens",        # canonical field (Anthropic, OpenAI-compatible)
            "inputTokenLimit",         # Google camelCase
            "max_prompt_tokens",       # alternate field (rare)
        ],
        "max_output_tokens": [
            "max_output_tokens",               # canonical field
            "max_tokens",                       # Anthropic, legacy
            "max_completion_tokens",            # Groq, OpenAI-compatible
            "top_provider.max_completion_tokens",  # OpenRouter-style nested
            "outputTokenLimit",                 # Google camelCase
        ],
    }

    # ── Heuristic sentinel registry ───────────────────────────────────

    HEURISTIC_SENTINEL_PREFIX = "__heuristic_"

    @classmethod
    def resolve(
        cls,
        raw_model: Dict[str, Any],
        field_type: str,
        aliases: Optional[List[str]] = None,
        heuristic_registry: Optional[Dict[str, Callable[[Dict[str, Any]], Optional[int]]]] = None,
    ) -> Optional[int]:
        """Resolve a token field from raw provider data.

        Args:
            raw_model: Raw model dict from provider API response.
            field_type: One of ``"context_window"``, ``"max_input_tokens"``,
                ``"max_output_tokens"``.
            aliases: Optional override alias chain.  If ``None``, uses
                ``DEFAULT_ALIASES[field_type]``.  If a partial list,
                remaining entries from the default chain are appended.
            heuristic_registry: Optional dict mapping sentinel names
                (without the ``__heuristic__`` prefix) to callables.
                The callable receives ``raw_model`` and returns
                ``Optional[int]``.

        Returns:
            First non-``None`` int value found by probing aliases in
            order, or ``None`` if no alias matches.
        """
        chain = cls._build_chain(field_type, aliases)
        for entry in chain:
            val = cls._probe_alias(raw_model, entry, heuristic_registry or {})
            if val is not None:
                return val
        return None

    @classmethod
    def _build_chain(
        cls,
        field_type: str,
        aliases: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the effective alias chain.

        If *aliases* is ``None``, returns the full default chain.
        If *aliases* is a list, appends default entries not already
        present, preserving custom order for the prefix.
        """
        defaults = cls.DEFAULT_ALIASES.get(field_type, [])
        if aliases is None:
            return defaults
        # Append remaining default entries not already in the custom list
        seen = set(aliases)
        return aliases + [a for a in defaults if a not in seen]

    @classmethod
    def _probe_alias(
        cls,
        raw_model: Dict[str, Any],
        entry: str,
        heuristic_registry: Dict[str, Callable[[Dict[str, Any]], Optional[int]]],
    ) -> Optional[int]:
        """Probe a single alias entry against *raw_model*."""
        # Heuristic sentinel dispatch
        if entry.startswith(cls.HEURISTIC_SENTINEL_PREFIX):
            raw_name = entry[len(cls.HEURISTIC_SENTINEL_PREFIX):]
            # Strip trailing "__" if present (sentinel format __heuristic_NAME__)
            sentinel_name = raw_name.rstrip("_")
            hook = heuristic_registry.get(sentinel_name)
            if hook is not None:
                return hook(raw_model)
            return None

        # Nested path (dot notation like "limits.context")
        if "." in entry:
            return cls._get_nested(raw_model, entry)

        # Flat key lookup
        return raw_model.get(entry)

    @staticmethod
    def _get_nested(raw_model: Dict[str, Any], path: str) -> Optional[int]:
        """Safely traverse a dot-notation path.

        ``"limits.context"`` → ``raw_model.get("limits", {}).get("context")``
        ``"top_provider.max_completion_tokens"`` →
            ``raw_model.get("top_provider", {}).get("max_completion_tokens")``

        Returns:
            The value at the end of the path, or ``None`` if any
            intermediate key is missing or not a dict.
        """
        current: Any = raw_model
        for key in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current  # May be None or any type — caller validates int
