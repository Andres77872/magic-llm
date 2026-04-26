"""SubagentBundle: Container for subagent tools for agent loop injection.

magic-llm owns ALL bundle architecture. This module provides:
- SubagentBundle: Dataclass collecting schemas, callables, manifests

Collected by load_subagents() after registration complete.
Provides tool specs for provider adapter consumption.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.registry import SubagentRegistry
from magic_llm.agent.binder import Binder

logger = logging.getLogger(__name__)


@dataclass
class SubagentBundle:
    """Container for subagent tools ready for agent loop injection.

    Collected by load_subagents() after registration complete.
    Provides tool specs for provider adapter consumption.

    Attributes:
        tool_schemas: List of OpenAI-compatible tool definitions.
        tool_callables: Dict of raw callables (NOT wrapped — wrapping in TaskExecutor).
        manifests: List of SubagentManifest for observability.
        registered_count: Number of successfully registered subagents.
    """

    # Required: tool definitions for LLM
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)

    # Raw callables for registration (NOT wrapped)
    tool_callables: dict[str, Callable] = field(default_factory=dict)

    # Manifests for observability/debugging
    manifests: list[SubagentManifest] = field(default_factory=list)

    # Metadata for debugging/observability
    registered_count: int = 0

    @classmethod
    def from_registry(cls, registry: SubagentRegistry) -> 'SubagentBundle':
        """Build bundle from all registered subagents.

        Flow:
        1. List all manifests from registry
        2. For each enabled manifest:
           - Get callable from registry
           - Validate signature via Binder
           - Add schema, callable, and manifest to bundle

        NOTE: Safeguards NOT applied here. Registration in TaskExecutor
        handles depth tracking, timeout, semaphore, and normalization.

        Args:
            registry: SubagentRegistry instance.

        Returns:
            SubagentBundle with schemas, callables, and manifests.
        """
        manifests = registry.list_manifests()

        schemas = []
        callables = {}
        manifest_list = []

        for manifest in manifests:
            if not manifest.enabled:
                logger.debug(
                    "Skipping disabled subagent '%s'",
                    manifest.id
                )
                continue

            callable = registry.get_callable(manifest.id)
            if callable is None:
                logger.warning(
                    "No callable registered for manifest '%s' — skipping",
                    manifest.id
                )
                continue

            # Validate via Binder (signature check only)
            try:
                validated_manifest, validated_callable = Binder.join(manifest, callable)

                schemas.append(validated_manifest.tool_schema)
                callables[validated_manifest.id] = validated_callable
                manifest_list.append(validated_manifest)

                logger.debug(
                    "Added subagent '%s' to bundle",
                    validated_manifest.id
                )

            except Exception as e:
                logger.error(
                    "Failed to validate subagent '%s': %s",
                    manifest.id,
                    e
                )
                continue

        logger.info(
            "Built SubagentBundle with %d subagents",
            len(schemas)
        )

        return cls(
            tool_schemas=schemas,
            tool_callables=callables,
            manifests=manifest_list,
            registered_count=len(schemas)
        )

    def to_tool_specs(self) -> Dict[str, dict]:
        """Provide tool specs for provider adapter consumption.

        Returns dict indexed by manifest ID for easy lookup.

        Returns:
            Dict of {id: tool_schema} for each registered subagent.
        """
        specs = {}
        for manifest in self.manifests:
            if manifest.enabled:
                specs[manifest.id] = manifest.tool_schema
        return specs

    # Backward compatibility: alias for tool_functions
    @property
    def tool_functions(self) -> dict[str, Callable]:
        """Alias for tool_callables (backward compatibility).

        NOTE: These are RAW callables, not wrapped with safeguards.
        Registration via MagicLLM.register_task() applies wrapping.
        """
        return self.tool_callables