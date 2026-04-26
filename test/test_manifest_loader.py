"""Unit tests for ManifestLoader — YAML discovery and validation.

Tests cover:
- Directory scanning for *.agent.yaml files
- YAML parsing and validation
- Duplicate ID detection
- Malformed YAML handling
- Empty directory handling
- Explicit file list loading
"""
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from magic_llm.agent.loader import ManifestLoader, ManifestLoadError
from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.errors import DuplicateSubagentError


# ─── Helpers ────────────────────────────────────────────────────────────────


def _write_yaml(path: Path, content: str) -> None:
    """Write YAML content to file."""
    path.write_text(dedent(content))


def _make_valid_yaml(
    id: str = "test.agent",
    name: str = "Test Agent",
    description: str = "A test subagent",
    version: str = "1.0.0",
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
        "version": version,
        "input_schema": input_schema,
        "timeout_seconds": 30,
        "max_concurrency": 5,
        "max_depth": 3,
    })


# ─── Directory Discovery Tests ───────────────────────────────────────────────


class TestManifestLoaderDiscovery:
    """Directory scanning for *.agent.yaml files."""

    @pytest.mark.asyncio
    async def test_load_all_scans_directory(self):
        """load_all() scans directory for *.agent.yaml files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Write two valid manifests
            _write_yaml(
                dir_path / "research.agent.yaml",
                _make_valid_yaml(id="research", name="Research Agent")
            )
            _write_yaml(
                dir_path / "search.agent.yaml",
                _make_valid_yaml(id="search", name="Search Agent")
            )

            loader = ManifestLoader(dir_path)
            manifests = await loader.load_all()

            assert len(manifests) == 2
            assert {m.id for m in manifests} == {"research", "search"}

    @pytest.mark.asyncio
    async def test_load_all_nonexistent_directory_returns_empty(self):
        """load_all() returns empty list if directory doesn't exist."""
        loader = ManifestLoader(Path("/nonexistent/path"))
        manifests = await loader.load_all()

        assert manifests == []
        assert loader.get_loaded_ids() == {}

    @pytest.mark.asyncio
    async def test_load_all_empty_directory_returns_empty(self):
        """load_all() returns empty list if directory has no YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            loader = ManifestLoader(dir_path)
            manifests = await loader.load_all()

            assert manifests == []
            assert loader.get_loaded_ids() == {}

    @pytest.mark.asyncio
    async def test_load_all_skips_non_yaml_files(self):
        """load_all() only scans *.agent.yaml, not other files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Write valid manifest
            _write_yaml(
                dir_path / "valid.agent.yaml",
                _make_valid_yaml(id="valid")
            )

            # Write non-YAML files that should be ignored
            (dir_path / "readme.txt").write_text("documentation")
            (dir_path / "config.json").write_text("{}")
            (dir_path / "other.yaml").write_text("key: value")

            loader = ManifestLoader(dir_path)
            manifests = await loader.load_all()

            assert len(manifests) == 1
            assert manifests[0].id == "valid"


# ─── YAML Parsing Tests ───────────────────────────────────────────────────────


class TestManifestLoaderParsing:
    """YAML parsing and validation."""

    @pytest.mark.asyncio
    async def test_load_file_parses_valid_yaml(self):
        """_load_file() parses valid YAML into SubagentManifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.agent.yaml"
            _write_yaml(yaml_path, _make_valid_yaml(id="test"))

            loader = ManifestLoader(Path(tmpdir))
            manifest = await loader._load_file(yaml_path)

            assert manifest.id == "test"
            assert manifest.name == "Test Agent"
            assert manifest.description == "A test subagent"
            assert manifest.version == "1.0.0"
            assert manifest.source_file == yaml_path

    @pytest.mark.asyncio
    async def test_load_file_sets_source_file_attribute(self):
        """_load_file() sets source_file path for observability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.agent.yaml"
            _write_yaml(yaml_path, _make_valid_yaml())

            loader = ManifestLoader(Path(tmpdir))
            manifest = await loader._load_file(yaml_path)

            assert manifest.source_file == yaml_path

    @pytest.mark.asyncio
    async def test_load_file_validates_id_pattern(self):
        """_load_file() validates ID matches ^[a-z0-9._-]+$ pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "invalid.agent.yaml"
            invalid_content = _make_valid_yaml(id="Invalid-ID-with-uppercase")

            _write_yaml(yaml_path, invalid_content)

            loader = ManifestLoader(Path(tmpdir))

            with pytest.raises(ManifestLoadError) as exc_info:
                await loader._load_file(yaml_path)

            assert "pattern" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_load_file_validates_version_pattern(self):
        """_load_file() validates version matches semver pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "invalid.agent.yaml"
            invalid_content = _make_valid_yaml(version="invalid-version")

            _write_yaml(yaml_path, invalid_content)

            loader = ManifestLoader(Path(tmpdir))

            with pytest.raises(ManifestLoadError) as exc_info:
                await loader._load_file(yaml_path)

            assert exc_info.value.file_path == yaml_path

    @pytest.mark.asyncio
    async def test_load_file_validates_required_fields(self):
        """_load_file() validates required fields (id, name, description, version, input_schema)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "missing.agent.yaml"
            # Missing required field 'name'
            missing_content = yaml.dump({
                "apiVersion": "magic-agents/v1",
                "kind": "TaskSubagent",
                "id": "test",
                "description": "test",
                "version": "1.0.0",
                "input_schema": {"type": "object"},
            })

            _write_yaml(yaml_path, missing_content)

            loader = ManifestLoader(Path(tmpdir))

            with pytest.raises(ManifestLoadError):
                await loader._load_file(yaml_path)


# ─── Duplicate Detection Tests ───────────────────────────────────────────────


class TestManifestLoaderDuplicates:
    """Duplicate ID detection."""

    @pytest.mark.asyncio
    async def test_duplicate_id_raises_error(self):
        """load_all() raises DuplicateSubagentError for duplicate IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Write two manifests with same ID
            _write_yaml(
                dir_path / "first.agent.yaml",
                _make_valid_yaml(id="duplicate_id")
            )
            _write_yaml(
                dir_path / "second.agent.yaml",
                _make_valid_yaml(id="duplicate_id")
            )

            loader = ManifestLoader(dir_path)

            with pytest.raises(DuplicateSubagentError) as exc_info:
                await loader.load_all()

            assert exc_info.value.agent_id == "duplicate_id"
            assert "duplicate_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_duplicate_detection_tracks_loaded_ids(self):
        """Loader tracks loaded IDs for duplicate detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            _write_yaml(
                dir_path / "a.agent.yaml",
                _make_valid_yaml(id="agent_a")
            )
            _write_yaml(
                dir_path / "b.agent.yaml",
                _make_valid_yaml(id="agent_b")
            )

            loader = ManifestLoader(dir_path)
            await loader.load_all()

            loaded_ids = loader.get_loaded_ids()
            assert loaded_ids["agent_a"] == dir_path / "a.agent.yaml"
            assert loaded_ids["agent_b"] == dir_path / "b.agent.yaml"


# ─── Malformed YAML Tests ─────────────────────────────────────────────────────


class TestManifestLoaderMalformedYAML:
    """Malformed YAML handling."""

    @pytest.mark.asyncio
    async def test_malformed_yaml_raises_manifest_load_error(self):
        """_load_file() raises ManifestLoadError for malformed YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "malformed.agent.yaml"
            yaml_path.write_text("invalid: yaml: content: [")

            loader = ManifestLoader(Path(tmpdir))

            with pytest.raises(ManifestLoadError) as exc_info:
                await loader._load_file(yaml_path)

            assert exc_info.value.file_path == yaml_path
            assert "YAML" in exc_info.value.error_details or "parsing" in exc_info.value.error_details.lower()

    @pytest.mark.asyncio
    async def test_empty_yaml_raises_manifest_load_error(self):
        """_load_file() raises ManifestLoadError for empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "empty.agent.yaml"
            yaml_path.write_text("")

            loader = ManifestLoader(Path(tmpdir))

            with pytest.raises(ManifestLoadError) as exc_info:
                await loader._load_file(yaml_path)

            assert exc_info.value.file_path == yaml_path
            assert "Empty" in exc_info.value.error_details

    @pytest.mark.asyncio
    async def test_load_all_skips_malformed_files(self):
        """load_all() logs error and skips malformed files, continues with others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Write valid and malformed manifests
            _write_yaml(
                dir_path / "valid.agent.yaml",
                _make_valid_yaml(id="valid")
            )
            (dir_path / "malformed.agent.yaml").write_text("broken: yaml: [")

            loader = ManifestLoader(dir_path)
            manifests = await loader.load_all()

            # Should still load the valid one
            assert len(manifests) == 1
            assert manifests[0].id == "valid"


# ─── Explicit File List Tests ───────────────────────────────────────────────


class TestManifestLoaderExplicitFiles:
    """load_from_files() for explicit file list."""

    @pytest.mark.asyncio
    async def test_load_from_files_loads_explicit_list(self):
        """load_from_files() loads specific files without directory scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Write files in different locations
            file_a = dir_path / "a.agent.yaml"
            file_b = dir_path / "b.agent.yaml"

            _write_yaml(file_a, _make_valid_yaml(id="a"))
            _write_yaml(file_b, _make_valid_yaml(id="b"))

            loader = ManifestLoader(Path("/any/path"))  # Path doesn't matter
            manifests = await loader.load_from_files([file_a, file_b])

            assert len(manifests) == 2
            assert {m.id for m in manifests} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_load_from_files_duplicate_raises_error(self):
        """load_from_files() raises DuplicateSubagentError for duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            file_a = dir_path / "a.agent.yaml"
            file_b = dir_path / "b.agent.yaml"

            _write_yaml(file_a, _make_valid_yaml(id="duplicate_id"))
            _write_yaml(file_b, _make_valid_yaml(id="duplicate_id"))

            loader = ManifestLoader(Path("/any/path"))

            with pytest.raises(DuplicateSubagentError):
                await loader.load_from_files([file_a, file_b])


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestManifestLoaderEdgeCases:
    """Edge case handling."""

    @pytest.mark.asyncio
    async def test_validates_policy_bounds(self):
        """Policy fields validated against bounds (timeout 1-600, concurrency 1-20, depth 1-10)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "bounds.agent.yaml"
            invalid_content = yaml.dump({
                "apiVersion": "magic-agents/v1",
                "kind": "TaskSubagent",
                "id": "test",
                "name": "Test",
                "description": "Test",
                "version": "1.0.0",
                "input_schema": {"type": "object"},
                "timeout_seconds": 700,  # Exceeds max 600
            })

            _write_yaml(yaml_path, invalid_content)

            loader = ManifestLoader(Path(tmpdir))

            with pytest.raises(ManifestLoadError):
                await loader._load_file(yaml_path)

    @pytest.mark.asyncio
    async def test_instruction_file_converted_to_path(self):
        """instruction_file string is converted to Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.agent.yaml"
            content = yaml.dump({
                "apiVersion": "magic-agents/v1",
                "kind": "TaskSubagent",
                "id": "test",
                "name": "Test",
                "description": "Test",
                "version": "1.0.0",
                "input_schema": {"type": "object"},
                "instruction_file": "instructions.md",  # String, not Path
            })

            _write_yaml(yaml_path, content)

            loader = ManifestLoader(Path(tmpdir))
            manifest = await loader._load_file(yaml_path)

            assert manifest.instruction_file == Path("instructions.md")

    @pytest.mark.asyncio
    async def test_disabled_manifest_loaded_but_not_executed(self):
        """enabled=False manifest is loaded but marked as disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "disabled.agent.yaml"
            content = yaml.dump({
                "apiVersion": "magic-agents/v1",
                "kind": "TaskSubagent",
                "id": "disabled",
                "name": "Disabled",
                "description": "Disabled agent",
                "version": "1.0.0",
                "input_schema": {"type": "object"},
                "enabled": False,
            })

            _write_yaml(yaml_path, content)

            loader = ManifestLoader(Path(tmpdir))
            manifests = await loader.load_all()

            assert len(manifests) == 1
            assert manifests[0].enabled is False