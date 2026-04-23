"""Instance-scoped subagent registry.

magic-llm provides SubagentRegistry for instance-scoped registration.
NO GLOBAL MUTABLE STATE. Each MagicLLM instance has its own registry.

This module provides:
- SubagentRegistry: Instance-scoped manifest and callable storage
- RegistryBackend: Protocol for forward-compatibility with v2 backends

Key architectural change from magic-agents:
- Previous: Global _GLOBAL_REGISTRY singleton (shared mutable state)
- New: Instance-scoped SubagentRegistry per MagicLLM client
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Protocol

from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.errors import DuplicateSubagentError

logger = logging.getLogger(__name__)


class RegistryBackend(Protocol):
    """Protocol for subagent registry backend.

    v1: SubagentRegistry (instance-scoped, YAML manifests)
    v2: DatabaseRegistry or APIRegistry can implement same protocol.

    This ensures Binder and Bundle are decoupled from registry implementation.
    """

    def register_manifest(self, manifest: SubagentManifest) -> None:
        """Register a manifest.

        Args:
            manifest: SubagentManifest to register.

        Raises:
            DuplicateSubagentError: If agent_id already registered.
        """
        ...

    def register_callable(self, agent_id: str, callable: Callable) -> None:
        """Register callable from decorator/code registry.

        Args:
            agent_id: Subagent ID.
            callable: Async callable for execution.
        """
        ...

    def get_manifest(self, agent_id: str) -> Optional[SubagentManifest]:
        """Resolve manifest by ID.

        Args:
            agent_id: Subagent ID to lookup.

        Returns:
            SubagentManifest if found, None otherwise.
        """
        ...

    def get_callable(self, agent_id: str) -> Optional[Callable]:
        """Get runtime callable from code registry.

        Args:
            agent_id: Subagent ID to lookup.

        Returns:
            Callable if registered, None otherwise.
        """
        ...

    def list_manifests(self) -> list[SubagentManifest]:
        """List all registered manifests.

        Returns:
            List of all SubagentManifest instances.
        """
        ...

    def is_initialized(self) -> bool:
        """Check if registry has been initialized.

        Returns:
            True if registry is ready for use.
        """
        ...

    def mark_initialized(self) -> None:
        """Mark registry as initialized."""
        ...

    def clear(self) -> None:
        """Clear registry (for testing/reset)."""
        ...


class SubagentRegistry:
    """Instance-scoped registry for subagent manifests and callables.

    NO GLOBAL STATE. Each MagicLLM instance has its own registry.

    Attributes:
        _manifests: Dict of agent_id -> SubagentManifest.
        _callables: Dict of agent_id -> Callable.
        _initialized: Whether registry is ready for use.

    Example:
        >>> registry = SubagentRegistry()
        >>> manifest = SubagentManifest(id="search", ...)
        >>> registry.register_manifest(manifest)
        >>> registry.register_callable("search", my_async_func)
        >>> bound_manifest, bound_callable = Binder.join(manifest, callable)
    """

    def __init__(self):
        """Initialize empty registry (instance-scoped)."""
        self._manifests: Dict[str, SubagentManifest] = {}
        self._callables: Dict[str, Callable] = {}
        self._initialized: bool = False

    def register_manifest(self, manifest: SubagentManifest) -> None:
        """Register a manifest from loader.

        Args:
            manifest: SubagentManifest to register.

        Raises:
            DuplicateSubagentError: If agent_id already registered.
        """
        if manifest.id in self._manifests:
            existing = self._manifests[manifest.id]
            raise DuplicateSubagentError(
                agent_id=manifest.id,
                existing_source=str(existing.source_file),
                new_source=str(manifest.source_file)
            )

        self._manifests[manifest.id] = manifest
        logger.debug(
            "Registered subagent manifest '%s' (version %s)",
            manifest.id,
            manifest.version
        )

    def register_callable(self, agent_id: str, callable: Callable) -> None:
        """Register callable from code registry dict.

        Args:
            agent_id: Subagent ID.
            callable: Async callable for execution.
        """
        self._callables[agent_id] = callable
        logger.debug(
            "Registered callable for subagent '%s'",
            agent_id
        )

    def get_manifest(self, agent_id: str) -> Optional[SubagentManifest]:
        """Resolve manifest by ID.

        Args:
            agent_id: Subagent ID to lookup.

        Returns:
            SubagentManifest if found, None otherwise.
        """
        return self._manifests.get(agent_id)

    def get_callable(self, agent_id: str) -> Optional[Callable]:
        """Get runtime callable from code registry.

        Args:
            agent_id: Subagent ID to lookup.

        Returns:
            Callable if registered, None otherwise.
        """
        return self._callables.get(agent_id)

    def list_manifests(self) -> list[SubagentManifest]:
        """List all registered manifests.

        Returns:
            List of all SubagentManifest instances.
        """
        return list(self._manifests.values())

    def list_callable_ids(self) -> list[str]:
        """List all registered callable IDs.

        Returns:
            List of agent_ids with registered callables.
        """
        return list(self._callables.keys())

    def get_registered_ids(self) -> list[str]:
        """List all registered manifest IDs.

        Returns:
            List of registered agent_ids.
        """
        return list(self._manifests.keys())

    def is_initialized(self) -> bool:
        """Check if registry has been initialized.

        Returns:
            True if registry is ready for use.
        """
        return self._initialized

    def mark_initialized(self) -> None:
        """Mark registry as initialized."""
        self._initialized = True
        logger.debug("Registry marked as initialized")

    def clear(self) -> None:
        """Clear registry (for testing/reset).

        Removes all manifests, callables, and resets initialized flag.
        """
        self._manifests.clear()
        self._callables.clear()
        self._initialized = False
        logger.debug("Registry cleared")