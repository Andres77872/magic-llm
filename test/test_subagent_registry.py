"""Unit tests for SubagentRegistry — instance-scoped registry.

Tests cover:
- Instance isolation (NO global state)
- Manifest and callable registration
- Duplicate detection
- Lookup methods
- Lifecycle methods (is_initialized, mark_initialized, clear)
"""
import pytest

from magic_llm.agent.registry import SubagentRegistry, RegistryBackend
from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.errors import DuplicateSubagentError


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_manifest(
    id: str = "test_agent",
    name: str = "Test Agent",
    description: str = "A test agent",
    version: str = "1.0.0",
    input_schema: dict = None,
    timeout_seconds: int = 30,
    max_concurrency: int = 5,
    max_depth: int = 3,
    source_file: str = None,
) -> SubagentManifest:
    """Create a valid SubagentManifest."""
    if input_schema is None:
        input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    return SubagentManifest(
        id=id,
        name=name,
        description=description,
        version=version,
        input_schema=input_schema,
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
        max_depth=max_depth,
        source_file=source_file,
    )


# ─── Instance Isolation Tests ───────────────────────────────────────────────


class TestSubagentRegistryIsolation:
    """Instance isolation — NO global state."""

    def test_two_instances_have_separate_manifests(self):
        """Two SubagentRegistry instances have separate manifests."""
        registry_a = SubagentRegistry()
        registry_b = SubagentRegistry()

        manifest_a = _make_manifest(id="agent_a")
        manifest_b = _make_manifest(id="agent_b")

        registry_a.register_manifest(manifest_a)
        registry_b.register_manifest(manifest_b)

        # Registry A has only agent_a
        assert registry_a.get_manifest("agent_a") == manifest_a
        assert registry_a.get_manifest("agent_b") is None

        # Registry B has only agent_b
        assert registry_b.get_manifest("agent_b") == manifest_b
        assert registry_b.get_manifest("agent_a") is None

    def test_two_instances_have_separate_callables(self):
        """Two SubagentRegistry instances have separate callables."""
        registry_a = SubagentRegistry()
        registry_b = SubagentRegistry()

        async def func_a(query: str) -> str:
            return "a"

        async def func_b(query: str) -> str:
            return "b"

        registry_a.register_callable("agent_a", func_a)
        registry_b.register_callable("agent_b", func_b)

        assert registry_a.get_callable("agent_a") == func_a
        assert registry_a.get_callable("agent_b") is None

        assert registry_b.get_callable("agent_b") == func_b
        assert registry_b.get_callable("agent_a") is None

    def test_initialized_state_per_instance(self):
        """is_initialized() is per-instance."""
        registry_a = SubagentRegistry()
        registry_b = SubagentRegistry()

        registry_a.mark_initialized()

        assert registry_a.is_initialized() is True
        assert registry_b.is_initialized() is False

    def test_clear_is_per_instance(self):
        """clear() only affects one instance."""
        registry_a = SubagentRegistry()
        registry_b = SubagentRegistry()

        manifest_a = _make_manifest(id="agent_a")
        registry_a.register_manifest(manifest_a)
        registry_a.mark_initialized()

        registry_a.clear()

        # Registry A is cleared
        assert registry_a.get_manifest("agent_a") is None
        assert registry_a.is_initialized() is False

        # Registry B unaffected
        assert registry_b.is_initialized() is False


# ─── Manifest Registration Tests ───────────────────────────────────────────────


class TestSubagentRegistryManifests:
    """Manifest registration."""

    def test_register_manifest_stores_by_id(self):
        """register_manifest() stores manifest indexed by id."""
        registry = SubagentRegistry()
        manifest = _make_manifest(id="my_agent")

        registry.register_manifest(manifest)

        stored = registry.get_manifest("my_agent")
        assert stored == manifest
        assert stored.id == "my_agent"

    def test_register_manifest_duplicate_raises_error(self):
        """register_manifest() raises DuplicateSubagentError for duplicates."""
        registry = SubagentRegistry()
        manifest_a = _make_manifest(id="duplicate_id", source_file="/a.yaml")
        manifest_b = _make_manifest(id="duplicate_id", source_file="/b.yaml")

        registry.register_manifest(manifest_a)

        with pytest.raises(DuplicateSubagentError) as exc_info:
            registry.register_manifest(manifest_b)

        assert exc_info.value.agent_id == "duplicate_id"

    def test_list_manifests_returns_all(self):
        """list_manifests() returns all registered manifests."""
        registry = SubagentRegistry()
        manifest_a = _make_manifest(id="agent_a")
        manifest_b = _make_manifest(id="agent_b")

        registry.register_manifest(manifest_a)
        registry.register_manifest(manifest_b)

        manifests = registry.list_manifests()

        assert len(manifests) == 2
        assert {m.id for m in manifests} == {"agent_a", "agent_b"}

    def test_list_manifests_empty_registry(self):
        """list_manifests() returns empty list for empty registry."""
        registry = SubagentRegistry()
        manifests = registry.list_manifests()
        assert manifests == []

    def test_get_manifest_unknown_returns_none(self):
        """get_manifest() returns None for unknown ID."""
        registry = SubagentRegistry()
        assert registry.get_manifest("unknown") is None

    def test_get_registered_ids_returns_keys(self):
        """get_registered_ids() returns list of registered IDs."""
        registry = SubagentRegistry()
        registry.register_manifest(_make_manifest(id="a"))
        registry.register_manifest(_make_manifest(id="b"))

        ids = registry.get_registered_ids()

        assert "a" in ids
        assert "b" in ids
        assert len(ids) == 2


# ─── Callable Registration Tests ───────────────────────────────────────────────


class TestSubagentRegistryCallables:
    """Callable registration."""

    def test_register_callable_stores_by_id(self):
        """register_callable() stores callable indexed by agent_id."""
        registry = SubagentRegistry()

        async def my_func(query: str) -> str:
            return query

        registry.register_callable("my_agent", my_func)

        stored = registry.get_callable("my_agent")
        assert stored == my_func

    def test_register_callable_allows_overwrite(self):
        """register_callable() allows overwriting existing callable."""
        registry = SubagentRegistry()

        async def func_a(query: str) -> str:
            return "a"

        async def func_b(query: str) -> str:
            return "b"

        registry.register_callable("agent", func_a)
        registry.register_callable("agent", func_b)

        assert registry.get_callable("agent") == func_b

    def test_list_callable_ids_returns_keys(self):
        """list_callable_ids() returns list of callable IDs."""
        registry = SubagentRegistry()

        registry.register_callable("a", lambda: "a")
        registry.register_callable("b", lambda: "b")

        ids = registry.list_callable_ids()

        assert "a" in ids
        assert "b" in ids
        assert len(ids) == 2

    def test_get_callable_unknown_returns_none(self):
        """get_callable() returns None for unknown ID."""
        registry = SubagentRegistry()
        assert registry.get_callable("unknown") is None


# ─── Lifecycle Tests ───────────────────────────────────────────────────────


class TestSubagentRegistryLifecycle:
    """Lifecycle methods."""

    def test_new_registry_not_initialized(self):
        """New registry is not initialized."""
        registry = SubagentRegistry()
        assert registry.is_initialized() is False

    def test_mark_initialized_sets_flag(self):
        """mark_initialized() sets initialized flag."""
        registry = SubagentRegistry()
        registry.mark_initialized()
        assert registry.is_initialized() is True

    def test_clear_removes_all_manifests(self):
        """clear() removes all manifests."""
        registry = SubagentRegistry()
        registry.register_manifest(_make_manifest(id="a"))
        registry.register_manifest(_make_manifest(id="b"))

        registry.clear()

        assert registry.get_manifest("a") is None
        assert registry.get_manifest("b") is None
        assert registry.list_manifests() == []

    def test_clear_removes_all_callables(self):
        """clear() removes all callables."""
        registry = SubagentRegistry()
        registry.register_callable("a", lambda: "a")
        registry.register_callable("b", lambda: "b")

        registry.clear()

        assert registry.get_callable("a") is None
        assert registry.get_callable("b") is None
        assert registry.list_callable_ids() == []

    def test_clear_resets_initialized_flag(self):
        """clear() resets initialized flag."""
        registry = SubagentRegistry()
        registry.mark_initialized()
        registry.clear()

        assert registry.is_initialized() is False

    def test_clear_on_empty_registry_is_safe(self):
        """clear() on empty registry is safe."""
        registry = SubagentRegistry()
        registry.clear()

        assert registry.list_manifests() == []
        assert registry.list_callable_ids() == []
        assert registry.is_initialized() is False


# ─── RegistryBackend Protocol Tests ───────────────────────────────────────────


class TestRegistryBackendProtocol:
    """RegistryBackend protocol compliance."""

    def test_subagent_registry_is_registry_backend(self):
        """SubagentRegistry implements RegistryBackend protocol."""
        registry = SubagentRegistry()

        # Protocol methods should exist
        assert hasattr(registry, "register_manifest")
        assert hasattr(registry, "register_callable")
        assert hasattr(registry, "get_manifest")
        assert hasattr(registry, "get_callable")
        assert hasattr(registry, "list_manifests")
        assert hasattr(registry, "is_initialized")
        assert hasattr(registry, "mark_initialized")
        assert hasattr(registry, "clear")

        # Protocol compliance check
        def check_protocol(obj: RegistryBackend) -> None:
            pass

        check_protocol(registry)  # Should not raise TypeError


# ─── Combined Registration Tests ───────────────────────────────────────────────


class TestSubagentRegistryCombined:
    """Combined manifest + callable registration."""

    def test_manifest_and_callable_for_same_id(self):
        """Registry can hold both manifest and callable for same ID."""
        registry = SubagentRegistry()

        manifest = _make_manifest(id="agent")
        async def func(query: str) -> str:
            return query

        registry.register_manifest(manifest)
        registry.register_callable("agent", func)

        assert registry.get_manifest("agent") == manifest
        assert registry.get_callable("agent") == func

    def test_manifest_without_callable_is_valid(self):
        """Manifest can exist without callable (awaiting registration)."""
        registry = SubagentRegistry()

        manifest = _make_manifest(id="agent")
        registry.register_manifest(manifest)

        assert registry.get_manifest("agent") == manifest
        assert registry.get_callable("agent") is None

    def test_callable_without_manifest_is_valid(self):
        """Callable can exist without manifest (manual registration)."""
        registry = SubagentRegistry()

        async def func(query: str) -> str:
            return query

        registry.register_callable("agent", func)

        assert registry.get_manifest("agent") is None
        assert registry.get_callable("agent") == func