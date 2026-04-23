"""Unit tests for SubagentBundle — schema/callable container.

Tests cover:
- from_registry() collection
- Disabled manifest skipping
- to_tool_specs() format
- tool_functions alias
- Empty bundle handling
"""
import pytest

from magic_llm.agent.bundle import SubagentBundle
from magic_llm.agent.registry import SubagentRegistry
from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.binder import Binder


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_manifest(
    id: str = "test_agent",
    name: str = "Test Agent",
    description: str = "A test agent",
    enabled: bool = True,
    input_schema: dict = None,
) -> SubagentManifest:
    """Create a SubagentManifest."""
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
        version="1.0.0",
        input_schema=input_schema,
        enabled=enabled,
    )


def _make_async_func(id: str = "test_agent"):
    """Create a valid async callable."""
    async def func(query: str) -> str:
        return f"{id}: {query}"
    return func


# ─── from_registry() Tests ───────────────────────────────────────────────────


class TestSubagentBundleFromRegistry:
    """SubagentBundle.from_registry() collection."""

    def test_from_registry_creates_bundle(self):
        """from_registry() creates bundle from registry contents."""
        registry = SubagentRegistry()

        manifest = _make_manifest(id="agent")
        func = _make_async_func()

        registry.register_manifest(manifest)
        registry.register_callable("agent", func)

        bundle = SubagentBundle.from_registry(registry)

        assert bundle is not None
        assert bundle.registered_count == 1

    def test_from_registry_collects_schemas(self):
        """from_registry() collects tool schemas from manifests."""
        registry = SubagentRegistry()

        manifest_a = _make_manifest(id="agent_a")
        manifest_b = _make_manifest(id="agent_b")

        registry.register_manifest(manifest_a)
        registry.register_manifest(manifest_b)
        registry.register_callable("agent_a", _make_async_func("agent_a"))
        registry.register_callable("agent_b", _make_async_func("agent_b"))

        bundle = SubagentBundle.from_registry(registry)

        assert len(bundle.tool_schemas) == 2
        assert bundle.tool_schemas[0]["function"]["name"] == "agent_a"
        assert bundle.tool_schemas[1]["function"]["name"] == "agent_b"

    def test_from_registry_collects_callables(self):
        """from_registry() collects callables from registry."""
        registry = SubagentRegistry()

        manifest = _make_manifest(id="agent")
        func = _make_async_func()

        registry.register_manifest(manifest)
        registry.register_callable("agent", func)

        bundle = SubagentBundle.from_registry(registry)

        assert "agent" in bundle.tool_callables
        assert bundle.tool_callables["agent"] == func

    def test_from_registry_collects_manifests(self):
        """from_registry() collects manifests for observability."""
        registry = SubagentRegistry()

        manifest_a = _make_manifest(id="agent_a")
        manifest_b = _make_manifest(id="agent_b")

        registry.register_manifest(manifest_a)
        registry.register_manifest(manifest_b)
        registry.register_callable("agent_a", _make_async_func("agent_a"))
        registry.register_callable("agent_b", _make_async_func("agent_b"))

        bundle = SubagentBundle.from_registry(registry)

        assert len(bundle.manifests) == 2
        assert {m.id for m in bundle.manifests} == {"agent_a", "agent_b"}

    def test_from_registry_skips_manifest_without_callable(self):
        """from_registry() skips manifest if callable not registered."""
        registry = SubagentRegistry()

        manifest_a = _make_manifest(id="agent_a")
        manifest_b = _make_manifest(id="agent_b")

        registry.register_manifest(manifest_a)
        registry.register_manifest(manifest_b)
        # Only register callable for agent_a
        registry.register_callable("agent_a", _make_async_func("agent_a"))

        bundle = SubagentBundle.from_registry(registry)

        assert bundle.registered_count == 1
        assert "agent_a" in bundle.tool_callables
        assert "agent_b" not in bundle.tool_callables

    def test_from_registry_empty_registry(self):
        """from_registry() returns empty bundle for empty registry."""
        registry = SubagentRegistry()

        bundle = SubagentBundle.from_registry(registry)

        assert bundle.registered_count == 0
        assert bundle.tool_schemas == []
        assert bundle.tool_callables == {}
        assert bundle.manifests == []


# ─── Disabled Manifest Tests ───────────────────────────────────────────────────


class TestSubagentBundleDisabled:
    """Disabled manifest handling."""

    def test_from_registry_skips_disabled_manifest(self):
        """from_registry() skips manifest with enabled=False."""
        registry = SubagentRegistry()

        manifest_enabled = _make_manifest(id="enabled", enabled=True)
        manifest_disabled = _make_manifest(id="disabled", enabled=False)

        registry.register_manifest(manifest_enabled)
        registry.register_manifest(manifest_disabled)
        registry.register_callable("enabled", _make_async_func("enabled"))
        registry.register_callable("disabled", _make_async_func("disabled"))

        bundle = SubagentBundle.from_registry(registry)

        assert bundle.registered_count == 1
        assert "enabled" in bundle.tool_callables
        assert "disabled" not in bundle.tool_callables

    def test_to_tool_specs_skips_disabled(self):
        """to_tool_specs() skips disabled manifests."""
        manifest_enabled = _make_manifest(id="enabled", enabled=True)
        manifest_disabled = _make_manifest(id="disabled", enabled=False)

        bundle = SubagentBundle(
            tool_schemas=[manifest_enabled.tool_schema],
            tool_callables={"enabled": _make_async_func("enabled")},
            manifests=[manifest_enabled, manifest_disabled],
            registered_count=1,
        )

        specs = bundle.to_tool_specs()

        assert "enabled" in specs
        assert "disabled" not in specs


# ─── to_tool_specs() Tests ───────────────────────────────────────────────────


class TestSubagentBundleToolSpecs:
    """to_tool_specs() format."""

    def test_to_tool_specs_returns_dict(self):
        """to_tool_specs() returns dict indexed by manifest ID."""
        registry = SubagentRegistry()

        manifest_a = _make_manifest(id="agent_a")
        manifest_b = _make_manifest(id="agent_b")

        registry.register_manifest(manifest_a)
        registry.register_manifest(manifest_b)
        registry.register_callable("agent_a", _make_async_func("agent_a"))
        registry.register_callable("agent_b", _make_async_func("agent_b"))

        bundle = SubagentBundle.from_registry(registry)
        specs = bundle.to_tool_specs()

        assert isinstance(specs, dict)
        assert specs["agent_a"]["type"] == "function"
        assert specs["agent_b"]["type"] == "function"

    def test_to_tool_specs_schema_format(self):
        """to_tool_specs() schemas are OpenAI-compatible."""
        manifest = _make_manifest(
            id="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        bundle = SubagentBundle(
            tool_schemas=[manifest.tool_schema],
            tool_callables={"search": _make_async_func("search")},
            manifests=[manifest],
            registered_count=1,
        )

        specs = bundle.to_tool_specs()

        assert specs["search"]["type"] == "function"
        assert specs["search"]["function"]["name"] == "search"
        assert specs["search"]["function"]["description"] == "Search the web"
        assert specs["search"]["function"]["parameters"]["type"] == "object"

    def test_to_tool_specs_empty_bundle(self):
        """to_tool_specs() returns empty dict for empty bundle."""
        bundle = SubagentBundle()
        specs = bundle.to_tool_specs()

        assert specs == {}


# ─── tool_functions Alias Tests ───────────────────────────────────────────────


class TestSubagentBundleAlias:
    """tool_functions backward compatibility alias."""

    def test_tool_functions_alias(self):
        """tool_functions is alias for tool_callables."""
        registry = SubagentRegistry()

        manifest = _make_manifest(id="agent")
        func = _make_async_func()

        registry.register_manifest(manifest)
        registry.register_callable("agent", func)

        bundle = SubagentBundle.from_registry(registry)

        assert bundle.tool_functions == bundle.tool_callables
        assert "agent" in bundle.tool_functions

    def test_tool_functions_raw_not_wrapped(self):
        """tool_functions contains raw callables (NOT wrapped with safeguards)."""
        bundle = SubagentBundle(
            tool_schemas=[],
            tool_callables={"agent": _make_async_func()},
            manifests=[],
            registered_count=1,
        )

        # These are raw callables
        callable = bundle.tool_functions["agent"]
        assert callable.__name__ == "func"


# ─── Registered Count Tests ───────────────────────────────────────────────────


class TestSubagentBundleCount:
    """registered_count tracking."""

    def test_registered_count_matches_schemas(self):
        """registered_count matches number of tool_schemas."""
        registry = SubagentRegistry()

        for i in range(5):
            manifest = _make_manifest(id=f"agent_{i}")
            registry.register_manifest(manifest)
            registry.register_callable(f"agent_{i}", _make_async_func(f"agent_{i}"))

        bundle = SubagentBundle.from_registry(registry)

        assert bundle.registered_count == 5
        assert len(bundle.tool_schemas) == 5

    def test_registered_count_zero_for_empty(self):
        """registered_count is 0 for empty bundle."""
        bundle = SubagentBundle()

        assert bundle.registered_count == 0


# ─── Dataclass Behavior Tests ───────────────────────────────────────────────


class TestSubagentBundleDataclass:
    """Dataclass behavior."""

    def test_default_factory_lists(self):
        """tool_schemas and manifests use default_factory=list."""
        bundle = SubagentBundle()

        assert bundle.tool_schemas == []
        assert bundle.manifests == []

    def test_default_factory_dicts(self):
        """tool_callables use default_factory=dict."""
        bundle = SubagentBundle()

        assert bundle.tool_callables == {}

    def test_bundle_is_mutable(self):
        """Bundle fields can be modified."""
        bundle = SubagentBundle()

        bundle.tool_schemas.append({"test": "schema"})
        bundle.tool_callables["test"] = lambda: "test"

        assert len(bundle.tool_schemas) == 1
        assert "test" in bundle.tool_callables


# ─── Validation During Bundle Creation Tests ───────────────────────────────────


class TestSubagentBundleValidation:
    """Binder validation during from_registry()."""

    def test_from_registry_validates_via_binder(self):
        """from_registry() validates each manifest+callable via Binder."""
        registry = SubagentRegistry()

        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        async def valid_func(query: str) -> str:
            return query

        registry.register_manifest(manifest)
        registry.register_callable("test_agent", valid_func)

        bundle = SubagentBundle.from_registry(registry)

        assert bundle.registered_count == 1

    def test_from_registry_skips_binder_error(self):
        """from_registry() skips entries that fail Binder validation."""
        registry = SubagentRegistry()

        manifest_valid = _make_manifest(id="valid")
        manifest_invalid = _make_manifest(
            id="invalid",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        async def valid_func(query: str) -> str:
            return query

        # Invalid: missing required 'query' param
        async def invalid_func() -> str:
            return "missing query"

        registry.register_manifest(manifest_valid)
        registry.register_manifest(manifest_invalid)
        registry.register_callable("valid", valid_func)
        registry.register_callable("invalid", invalid_func)

        bundle = SubagentBundle.from_registry(registry)

        # Only valid should be in bundle
        assert bundle.registered_count == 1
        assert "valid" in bundle.tool_callables
        assert "invalid" not in bundle.tool_callables