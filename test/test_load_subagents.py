"""Integration tests for load_subagents() — full pipeline.

Tests cover:
- Full pipeline: manifest_dir → discovery → registration → bundle
- MagicLLM.reset_depths() integration
- Feature flag disabled behavior
- Instance-scoped registry per MagicLLM instance
- Integration with register_task()
"""
import tempfile
import yaml
from pathlib import Path
from textwrap import dedent

import pytest

from magic_llm import MagicLLM
from magic_llm.agent.config import enable_subagents, disable_subagents, is_subagents_enabled
from magic_llm.agent.registry import SubagentRegistry
from magic_llm.agent.loader import ManifestLoader
from magic_llm.agent.bundle import SubagentBundle
from magic_llm.agent.decorator import subagent
from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.task_executor import reset_depths, _get_depth, _increment_depth


# ─── Helpers ────────────────────────────────────────────────────────────────


def _write_yaml(path: Path, content: str) -> None:
    """Write YAML content to file."""
    path.write_text(dedent(content))


def _make_valid_yaml(
    id: str = "test.agent",
    name: str = "Test Agent",
    description: str = "A test subagent",
    input_schema: dict = None,
) -> str:
    """Generate valid YAML manifest content."""
    if input_schema is None:
        input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    return yaml.dump({
        "apiVersion": "magic-agents/v1",
        "kind": "TaskSubagent",
        "id": id,
        "name": name,
        "description": description,
        "version": "1.0.0",
        "input_schema": input_schema,
        "timeout_seconds": 30,
        "max_concurrency": 5,
        "max_depth": 3,
    })


# ─── Feature Flag Tests ───────────────────────────────────────────────────────


class TestLoadSubagentsFeatureFlag:
    """Feature flag disabled behavior."""

    def setup_method(self):
        """Reset feature flag before each test."""
        disable_subagents()

    def teardown_method(self):
        """Disable after each test."""
        disable_subagents()

    @pytest.mark.asyncio
    async def test_disabled_returns_empty_bundle(self):
        """load_subagents() returns empty bundle when feature disabled."""
        client = MagicLLM(engine="openai", model="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(dir_path / "test.agent.yaml", _make_valid_yaml())

            bundle = await client.load_subagents(dir_path)

            assert bundle.registered_count == 0
            assert bundle.tool_schemas == []

    @pytest.mark.asyncio
    async def test_enabled_loads_manifests(self):
        """load_subagents() loads manifests when feature enabled."""
        enable_subagents()

        client = MagicLLM(engine="openai", model="gpt-4")

        code_registry = {}

        @subagent("test.agent", registry=code_registry)
        async def test_func(query: str) -> str:
            return query

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(dir_path / "test.agent.yaml", _make_valid_yaml(id="test.agent"))

            bundle = await client.load_subagents(dir_path, code_registry)

            assert bundle.registered_count == 1

    def test_feature_flag_functions(self):
        """enable_subagents() and disable_subagents() toggle flag."""
        assert is_subagents_enabled() is False

        enable_subagents()
        assert is_subagents_enabled() is True

        disable_subagents()
        assert is_subagents_enabled() is False


# ─── Full Pipeline Tests ───────────────────────────────────────────────────────


class TestLoadSubagentsPipeline:
    """Full pipeline: discovery → registration → bundle."""

    def setup_method(self):
        """Enable subagents before each test."""
        enable_subagents()

    def teardown_method(self):
        """Disable after each test."""
        disable_subagents()

    @pytest.mark.asyncio
    async def test_pipeline_discovery_to_bundle(self):
        """load_subagents() orchestrates discovery → registration → bundle."""
        client = MagicLLM(engine="openai", model="gpt-4")

        code_registry = {}

        @subagent("research.web", registry=code_registry)
        async def research_web(query: str) -> str:
            return f"Results for: {query}"

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(
                dir_path / "research.agent.yaml",
                _make_valid_yaml(id="research.web", name="Web Research")
            )

            bundle = await client.load_subagents(dir_path, code_registry)

            # Verify pipeline results
            assert bundle.registered_count == 1
            assert "research.web" in bundle.tool_callables
            assert bundle.tool_schemas[0]["function"]["name"] == "research.web"

    @pytest.mark.asyncio
    async def test_pipeline_multiple_subagents(self):
        """load_subagents() handles multiple subagents."""
        client = MagicLLM(engine="openai", model="gpt-4")

        code_registry = {}

        @subagent("search", registry=code_registry)
        async def search(query: str) -> str:
            return f"Search: {query}"

        @subagent("analyze", registry=code_registry)
        async def analyze(query: str) -> str:
            return f"Analyze: {query}"

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(dir_path / "search.agent.yaml", _make_valid_yaml(id="search"))
            _write_yaml(dir_path / "analyze.agent.yaml", _make_valid_yaml(id="analyze"))

            bundle = await client.load_subagents(dir_path, code_registry)

            assert bundle.registered_count == 2
            assert "search" in bundle.tool_callables
            assert "analyze" in bundle.tool_callables

    @pytest.mark.asyncio
    async def test_pipeline_without_code_registry(self):
        """load_subagents() handles empty code_registry — no bundle entries."""
        client = MagicLLM(engine="openai", model="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(dir_path / "test.agent.yaml", _make_valid_yaml())

            # No code_registry provided — manifests discovered but no callables
            bundle = await client.load_subagents(dir_path)

            # Manifests discovered but NOT in bundle (no callable)
            assert bundle.registered_count == 0
            assert bundle.tool_schemas == []
            assert bundle.manifests == []
            # Registry still has manifests internally
            assert client._subagent_registry is not None
            assert len(client._subagent_registry.list_manifests()) == 1

    @pytest.mark.asyncio
    async def test_pipeline_creates_instance_registry(self):
        """load_subagents() creates instance-scoped registry on client."""
        enable_subagents()

        client = MagicLLM(engine="openai", model="gpt-4")

        code_registry = {}

        @subagent("test", registry=code_registry)
        async def test_func(query: str) -> str:
            return query

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(dir_path / "test.agent.yaml", _make_valid_yaml(id="test"))

            await client.load_subagents(dir_path, code_registry)

            # Verify _subagent_registry exists on client
            assert client._subagent_registry is not None
            assert isinstance(client._subagent_registry, SubagentRegistry)
            assert client._subagent_registry.is_initialized()


# ─── Instance-Scoped Registry Tests ───────────────────────────────────────────


class TestLoadSubagentsInstanceScope:
    """Instance-scoped registry per MagicLLM instance."""

    def setup_method(self):
        """Enable subagents before each test."""
        enable_subagents()

    def teardown_method(self):
        """Disable after each test."""
        disable_subagents()

    @pytest.mark.asyncio
    async def test_two_clients_have_separate_registries(self):
        """Two MagicLLM instances have separate callables and registered tasks."""
        # Each client uses its own directory with its own manifest
        with tempfile.TemporaryDirectory() as tmpdir_a:
            with tempfile.TemporaryDirectory() as tmpdir_b:
                dir_a = Path(tmpdir_a)
                dir_b = Path(tmpdir_b)

                registry_a = {}

                @subagent("agent_a", registry=registry_a)
                async def func_a(query: str) -> str:
                    return "a"

                registry_b = {}

                @subagent("agent_b", registry=registry_b)
                async def func_b(query: str) -> str:
                    return "b"

                _write_yaml(dir_a / "a.agent.yaml", _make_valid_yaml(id="agent_a"))
                _write_yaml(dir_b / "b.agent.yaml", _make_valid_yaml(id="agent_b"))

                client_a = MagicLLM(engine="openai", model="gpt-4")
                client_b = MagicLLM(engine="openai", model="gpt-4")

                await client_a.load_subagents(dir_a, registry_a)
                await client_b.load_subagents(dir_b, registry_b)

                # Each client has own registry with different manifests
                assert client_a._subagent_registry.get_manifest("agent_a") is not None
                assert client_a._subagent_registry.get_manifest("agent_b") is None

                assert client_b._subagent_registry.get_manifest("agent_b") is not None
                assert client_b._subagent_registry.get_manifest("agent_a") is None

                # Each client has registered different tasks in TaskExecutor
                assert "agent_a" in client_a._task_executor._task_registry
                assert "agent_a" not in client_b._task_executor._task_registry

                assert "agent_b" in client_b._task_executor._task_registry
                assert "agent_b" not in client_a._task_executor._task_registry


# ─── reset_depths Tests ───────────────────────────────────────────────────────


class TestMagicLLMResetDepths:
    """MagicLLM.reset_depths() integration."""

    def setup_method(self):
        """Reset depths before each test."""
        reset_depths()

    def teardown_method(self):
        """Reset after each test."""
        reset_depths()

    def test_reset_depths_method_exists(self):
        """MagicLLM has reset_depths() method."""
        client = MagicLLM(engine="openai", model="gpt-4")

        assert hasattr(client, "reset_depths")
        assert callable(client.reset_depths)

    def test_reset_depths_clears_counters(self):
        """reset_depths() clears depth counters."""
        client = MagicLLM(engine="openai", model="gpt-4")

        # Increment depth
        _increment_depth("test_task")
        assert _get_depth("test_task") == 1

        # Reset via client method
        client.reset_depths()

        assert _get_depth("test_task") == 0


# ─── register_task Integration Tests ───────────────────────────────────────────


class TestLoadSubagentsRegisterTask:
    """Integration with register_task()."""

    def setup_method(self):
        """Enable subagents before each test."""
        enable_subagents()

    def teardown_method(self):
        """Disable after each test."""
        disable_subagents()

    @pytest.mark.asyncio
    async def test_load_subagents_registers_with_task_executor(self):
        """load_subagents() registers tasks with internal TaskExecutor."""
        client = MagicLLM(engine="openai", model="gpt-4")

        code_registry = {}

        @subagent("test", registry=code_registry)
        async def test_func(query: str) -> str:
            return query

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(dir_path / "test.agent.yaml", _make_valid_yaml(id="test"))

            await client.load_subagents(dir_path, code_registry)

            # Verify TaskExecutor was created and task registered
            assert client._task_executor is not None
            assert "test" in client._task_executor._task_registry

    @pytest.mark.asyncio
    async def test_registered_task_manifest_stored(self):
        """TaskExecutor stores converted TaskManifest."""
        client = MagicLLM(engine="openai", model="gpt-4")

        code_registry = {}

        @subagent("test", registry=code_registry)
        async def test_func(query: str) -> str:
            return query

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(
                dir_path / "test.agent.yaml",
                _make_valid_yaml(id="test")
            )

            await client.load_subagents(dir_path, code_registry)

            stored_manifest = client._task_executor.get_task_manifest("test")
            assert stored_manifest is not None
            assert stored_manifest.id == "test"
            assert stored_manifest.timeout_seconds == 30
            assert stored_manifest.max_concurrency == 5
            assert stored_manifest.max_depth == 3


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestLoadSubagentsEdgeCases:
    """Edge case handling."""

    def setup_method(self):
        """Enable subagents before each test."""
        enable_subagents()

    def teardown_method(self):
        """Disable after each test."""
        disable_subagents()

    @pytest.mark.asyncio
    async def test_empty_directory_returns_empty_bundle(self):
        """load_subagents() returns empty bundle for empty directory."""
        client = MagicLLM(engine="openai", model="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = await client.load_subagents(Path(tmpdir))

            assert bundle.registered_count == 0

    @pytest.mark.asyncio
    async def test_nonexistent_directory_returns_empty_bundle(self):
        """load_subagents() returns empty bundle for nonexistent directory."""
        client = MagicLLM(engine="openai", model="gpt-4")

        bundle = await client.load_subagents(Path("/nonexistent/path"))

        assert bundle.registered_count == 0

    @pytest.mark.asyncio
    async def test_manifest_without_callable_logs_warning(self, caplog):
        """load_subagents() logs warning for manifest without callable."""
        import logging
        caplog.set_level(logging.WARNING)

        client = MagicLLM(engine="openai", model="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            _write_yaml(dir_path / "orphan.agent.yaml", _make_valid_yaml(id="orphan"))

            # No callable for 'orphan' in code_registry
            await client.load_subagents(dir_path)

            # Should have logged warning about missing callable
            assert any("orphan" in record.message for record in caplog.records)


# ─── _to_task_manifest Tests ───────────────────────────────────────────────


class TestToTaskManifest:
    """SubagentManifest → TaskManifest conversion."""

    def test_to_task_manifest_excludes_yaml_fields(self):
        """_to_task_manifest() excludes YAML-specific fields."""
        enable_subagents()

        client = MagicLLM(engine="openai", model="gpt-4")

        subagent_manifest = SubagentManifest(
            id="test",
            name="Test",
            description="Test",
            version="1.0.0",
            input_schema={"type": "object"},
            apiVersion="magic-agents/v1",
            kind="TaskSubagent",
            source_file=Path("/test.yaml"),
            timeout_seconds=60,
            max_concurrency=10,
            max_depth=5,
        )

        task_manifest = client._to_task_manifest(subagent_manifest)

        # TaskManifest should NOT have YAML fields
        assert hasattr(task_manifest, "id")
        assert hasattr(task_manifest, "timeout_seconds")
        assert not hasattr(task_manifest, "apiVersion")
        assert not hasattr(task_manifest, "kind")
        assert not hasattr(task_manifest, "version")
        assert not hasattr(task_manifest, "source_file")

    def test_to_task_manifest_preserves_policy_fields(self):
        """_to_task_manifest() preserves execution policy fields."""
        enable_subagents()

        client = MagicLLM(engine="openai", model="gpt-4")

        subagent_manifest = SubagentManifest(
            id="test",
            name="Test",
            description="Test",
            version="1.0.0",
            input_schema={"type": "object"},
            timeout_seconds=120,
            max_concurrency=15,
            max_depth=8,
        )

        task_manifest = client._to_task_manifest(subagent_manifest)

        assert task_manifest.timeout_seconds == 120
        assert task_manifest.max_concurrency == 15
        assert task_manifest.max_depth == 8