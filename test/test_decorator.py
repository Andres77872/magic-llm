"""Unit tests for @subagent decorator — explicit registry dict population.

Tests cover:
- Registry dict population (NO global state)
- Warning when registry is None
- _subagent_id attribute tagging
- register_callable manual registration
"""
import pytest
import logging
import warnings

from magic_llm.agent.decorator import subagent, register_callable


# ─── Registry Dict Population Tests ───────────────────────────────────────────


class TestDecoratorRegistryPopulation:
    """Registry dict population — NO global state."""

    def test_decorator_populates_registry_dict(self):
        """@subagent populates user-provided registry dict."""
        code_registry = {}

        @subagent("my_agent", registry=code_registry)
        async def my_func(query: str) -> str:
            return query

        assert "my_agent" in code_registry
        assert code_registry["my_agent"] == my_func

    def test_decorator_returns_original_function(self):
        """@subagent returns the original function unchanged."""
        code_registry = {}

        @subagent("agent_id", registry=code_registry)
        async def my_func(query: str) -> str:
            return query

        # Function name preserved
        assert my_func.__name__ == "my_func"

    def test_decorator_sets_subagent_id_attribute(self):
        """@subagent sets _subagent_id attribute on function."""
        code_registry = {}

        @subagent("agent_id", registry=code_registry)
        async def my_func(query: str) -> str:
            return query

        assert hasattr(my_func, "_subagent_id")
        assert my_func._subagent_id == "agent_id"

    def test_multiple_decorators_same_registry(self):
        """Multiple @subagent decorators can populate same registry."""
        code_registry = {}

        @subagent("agent_a", registry=code_registry)
        async def func_a(query: str) -> str:
            return "a"

        @subagent("agent_b", registry=code_registry)
        async def func_b(query: str) -> str:
            return "b"

        assert len(code_registry) == 2
        assert code_registry["agent_a"] == func_a
        assert code_registry["agent_b"] == func_b

    def test_different_registry_dicts_independent(self):
        """Different registry dicts are independent."""
        registry_a = {}
        registry_b = {}

        @subagent("agent", registry=registry_a)
        async def func_a(query: str) -> str:
            return "a"

        @subagent("agent", registry=registry_b)
        async def func_b(query: str) -> str:
            return "b"

        assert registry_a["agent"] == func_a
        assert registry_b["agent"] == func_b
        # Same key in both dicts, different callables


# ─── Warning When Registry is None Tests ───────────────────────────────────────


class TestDecoratorNoneRegistry:
    """Warning when registry is None."""

    def test_none_registry_logs_warning(self, caplog):
        """@subagent logs warning if registry is None."""
        caplog.set_level(logging.WARNING)

        # No registry dict provided
        @subagent("my_agent")  # registry=None by default
        async def my_func(query: str) -> str:
            return query

        # Should have logged a warning
        assert any("registry" in record.message.lower() for record in caplog.records)
        assert any("my_agent" in record.message for record in caplog.records)

    def test_none_registry_does_not_populate(self):
        """@subagent with None registry doesn't populate anything."""
        @subagent("my_agent")  # registry=None
        async def my_func(query: str) -> str:
            return query

        # No global state should be affected
        # Function should still have _subagent_id
        assert hasattr(my_func, "_subagent_id")
        assert my_func._subagent_id == "my_agent"

    def test_none_registry_function_still_callable(self):
        """Function is still callable even with None registry."""
        @subagent("my_agent")
        async def my_func(query: str) -> str:
            return query

        # Should be able to call the function directly
        # (async, so we need to run it in a test)
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(my_func("test"))
        assert result == "test"


# ─── Manual Registration Tests ───────────────────────────────────────────────


class TestRegisterCallable:
    """register_callable manual registration."""

    def test_register_callable_populates_dict(self):
        """register_callable() populates registry dict."""
        code_registry = {}

        async def my_func(query: str) -> str:
            return query

        register_callable("my_agent", my_func, code_registry)

        assert "my_agent" in code_registry
        assert code_registry["my_agent"] == my_func

    def test_register_callable_sets_attribute(self):
        """register_callable() sets _subagent_id attribute."""
        code_registry = {}

        async def my_func(query: str) -> str:
            return query

        register_callable("agent_id", my_func, code_registry)

        assert hasattr(my_func, "_subagent_id")
        assert my_func._subagent_id == "agent_id"

    def test_register_callable_allows_overwrite(self):
        """register_callable() allows overwriting existing entry."""
        code_registry = {}

        async def func_a(query: str) -> str:
            return "a"

        async def func_b(query: str) -> str:
            return "b"

        register_callable("agent", func_a, code_registry)
        register_callable("agent", func_b, code_registry)

        assert code_registry["agent"] == func_b


# ─── Integration with SubagentRegistry Tests ───────────────────────────────────


class TestDecoratorIntegration:
    """Integration with SubagentRegistry workflow."""

    def test_decorator_then_register_callable_in_registry(self):
        """Workflow: decorator populates dict → dict passed to SubagentRegistry."""
        from magic_llm.agent.registry import SubagentRegistry

        code_registry = {}

        @subagent("research", registry=code_registry)
        async def research_func(query: str) -> str:
            return f"Research: {query}"

        # Create instance-scoped registry
        registry = SubagentRegistry()

        # Register callable from code_registry
        registry.register_callable("research", code_registry["research"])

        # Verify callable is in registry
        callable = registry.get_callable("research")
        assert callable == research_func

    def test_multiple_agents_workflow(self):
        """Full workflow with multiple decorated callables."""
        from magic_llm.agent.registry import SubagentRegistry
        from magic_llm.agent.definitions import SubagentManifest

        code_registry = {}

        @subagent("web_search", registry=code_registry)
        async def web_search(query: str) -> str:
            return f"Results for: {query}"

        @subagent("web_scrape", registry=code_registry)
        async def web_scrape(url: str) -> str:
            return f"Content from: {url}"

        # Verify both in code_registry
        assert len(code_registry) == 2

        # Create SubagentRegistry and register callables
        registry = SubagentRegistry()

        # Register manifests (simulating loader)
        registry.register_manifest(
            SubagentManifest(
                id="web_search",
                name="Web Search",
                description="Search the web",
                version="1.0.0",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        )
        registry.register_manifest(
            SubagentManifest(
                id="web_scrape",
                name="Web Scrape",
                description="Scrape a URL",
                version="1.0.0",
                input_schema={"type": "object", "properties": {"url": {"type": "string"}}},
            )
        )

        # Register callables from code_registry
        for agent_id, callable in code_registry.items():
            registry.register_callable(agent_id, callable)

        # Verify both are registered
        assert registry.get_callable("web_search") == web_search
        assert registry.get_callable("web_scrape") == web_scrape


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestDecoratorEdgeCases:
    """Edge case handling."""

    def test_decorator_with_complex_agent_id(self):
        """@subagent accepts complex IDs (e.g., 'research.web')."""
        code_registry = {}

        @subagent("research.web.search", registry=code_registry)
        async def func(query: str) -> str:
            return query

        assert "research.web.search" in code_registry

    def test_decorator_preserves_async_nature(self):
        """@subagent preserves async function nature."""
        code_registry = {}

        @subagent("agent", registry=code_registry)
        async def async_func(query: str) -> str:
            return query

        import asyncio
        assert asyncio.iscoroutinefunction(async_func)

    def test_decorator_on_sync_function(self):
        """@subagent can decorate sync function (Binder will reject later)."""
        code_registry = {}

        # Sync function — decorator accepts it, Binder will reject later
        @subagent("agent", registry=code_registry)
        def sync_func(query: str) -> str:
            return query

        assert "agent" in code_registry
        assert code_registry["agent"] == sync_func

    def test_decorator_with_classmethod(self):
        """@subagent can decorate classmethod."""
        code_registry = {}

        class MyClass:
            @subagent("my_agent", registry=code_registry)
            @classmethod
            async def method(cls, query: str) -> str:
                return query

        assert "my_agent" in code_registry

    def test_decorator_with_staticmethod(self):
        """@subagent can decorate staticmethod."""
        code_registry = {}

        class MyClass:
            @subagent("my_agent", registry=code_registry)
            @staticmethod
            async def method(query: str) -> str:
                return query

        assert "my_agent" in code_registry

    def test_decorator_id_matches_manifest_pattern(self):
        """Decorator IDs should match manifest pattern ^[a-z0-9._-]+$."""
        code_registry = {}

        # Valid IDs
        @subagent("valid_id", registry=code_registry)
        async def func1(query: str) -> str:
            return query

        @subagent("valid.id.with.dots", registry=code_registry)
        async def func2(query: str) -> str:
            return query

        @subagent("valid-id-with-dashes", registry=code_registry)
        async def func3(query: str) -> str:
            return query

        assert len(code_registry) == 3