"""Manifest loader for YAML files.

Loads *.agent.yaml files, validates with Pydantic, detects duplicates.
magic-llm is repo-agnostic — accepts explicit directory paths.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from magic_llm.agent.definitions import SubagentManifest

logger = logging.getLogger(__name__)


class ManifestLoadError(Exception):
    """Error loading manifest file.

    Attributes:
        file_path: Path to the problematic YAML file.
        error_details: Detailed error message.
    """

    def __init__(
        self,
        file_path: Path,
        error_details: str,
    ):
        self.file_path = file_path
        self.error_details = error_details
        message = f"Failed to load manifest from {file_path}: {error_details}"
        super().__init__(message)


class ManifestLoader:
    """Load and validate YAML manifest files.

    Scans explicit directory for *.agent.yaml files.
    Validates each with Pydantic SubagentManifest model.
    Detects duplicate IDs (hard error).

    magic-llm stays agnostic of filesystem conventions — directory passed by caller.

    Example YAML:
        apiVersion: magic-agents/v1
        kind: TaskSubagent
        id: research.web
        name: Web Research
        description: Search and summarize web content
        version: 1.0.0
        input_schema:
          type: object
          properties:
            query: {type: string}
          required: [query]
        timeout_seconds: 30
        max_concurrency: 5
        max_depth: 3

    Attributes:
        manifest_dir: Directory containing *.agent.yaml files.
        _loaded_ids: Dict tracking IDs for duplicate detection.
    """

    def __init__(self, manifest_dir: Path):
        """Initialize loader with directory path.

        Args:
            manifest_dir: Directory containing *.agent.yaml files.
        """
        self.manifest_dir = manifest_dir
        self._loaded_ids: dict[str, Path] = {}  # Track IDs for duplicate detection

    async def load_all(self) -> list[SubagentManifest]:
        """Load all manifest files from directory.

        Returns:
            List of validated SubagentManifest instances.

        Raises:
            DuplicateSubagentError: If duplicate IDs found (from definitions.py).
        """
        manifests = []

        if not self.manifest_dir.exists():
            logger.debug(
                "Manifest directory %s does not exist — no subagents loaded",
                self.manifest_dir
            )
            return manifests

        yaml_files = list(self.manifest_dir.glob("*.agent.yaml"))

        if not yaml_files:
            logger.debug(
                "No *.agent.yaml files found in %s",
                self.manifest_dir
            )
            return manifests

        logger.info(
            "Loading %d manifest files from %s",
            len(yaml_files),
            self.manifest_dir
        )

        for yaml_file in yaml_files:
            try:
                manifest = await self._load_file(yaml_file)

                # Duplicate detection - import here to avoid circular dependency
                from magic_llm.agent.errors import DuplicateSubagentError

                if manifest.id in self._loaded_ids:
                    raise DuplicateSubagentError(
                        agent_id=manifest.id,
                        existing_source=str(self._loaded_ids[manifest.id]),
                        new_source=str(yaml_file)
                    )

                self._loaded_ids[manifest.id] = yaml_file
                manifests.append(manifest)

            except Exception as e:
                # Check if it's a hard error (DuplicateSubagentError)
                from magic_llm.agent.errors import DuplicateSubagentError
                if isinstance(e, DuplicateSubagentError):
                    raise  # Re-raise - this is a hard error
                # Otherwise log and skip
                logger.error(
                    "Failed to load manifest %s: %s",
                    yaml_file,
                    e
                )
                continue

        logger.info(
            "Successfully loaded %d subagent manifests",
            len(manifests)
        )

        return manifests

    async def _load_file(self, yaml_file: Path) -> SubagentManifest:
        """Load and validate a single YAML file.

        Args:
            yaml_file: Path to YAML manifest.

        Returns:
            Validated SubagentManifest.

        Raises:
            ManifestLoadError: If YAML parsing or validation fails.
        """
        try:
            content = yaml_file.read_text()
            data = yaml.safe_load(content)

            if data is None:
                raise ManifestLoadError(
                    file_path=yaml_file,
                    error_details="Empty YAML file"
                )

            # Validate with Pydantic
            manifest = SubagentManifest(
                **data,
                source_file=yaml_file  # Track source for error messages
            )

            logger.debug(
                "Loaded manifest '%s' v%s from %s",
                manifest.id,
                manifest.version,
                yaml_file
            )

            return manifest

        except yaml.YAMLError as e:
            raise ManifestLoadError(
                file_path=yaml_file,
                error_details=f"YAML parsing error: {e}"
            )
        except Exception as e:
            # Check for validation errors from Pydantic
            raise ManifestLoadError(
                file_path=yaml_file,
                error_details=str(e)
            )

    def get_loaded_ids(self) -> dict[str, Path]:
        """Get map of loaded IDs to source files.

        Returns:
            Dict of agent_id -> source file path.
        """
        return self._loaded_ids.copy()

    async def load_from_files(self, manifest_files: list[Path]) -> list[SubagentManifest]:
        """Load manifests from explicit file list (no directory scanning).

        Args:
            manifest_files: List of YAML file paths to load.

        Returns:
            List of validated SubagentManifest instances.

        Raises:
            DuplicateSubagentError: If duplicate IDs found.
            ManifestLoadError: If any file fails to load.
        """
        manifests = []

        for yaml_file in manifest_files:
            manifest = await self._load_file(yaml_file)

            # Duplicate detection
            from magic_llm.agent.errors import DuplicateSubagentError

            if manifest.id in self._loaded_ids:
                raise DuplicateSubagentError(
                    agent_id=manifest.id,
                    existing_source=str(self._loaded_ids[manifest.id]),
                    new_source=str(yaml_file)
                )

            self._loaded_ids[manifest.id] = yaml_file
            manifests.append(manifest)

        logger.info(
            "Loaded %d manifest files from explicit list",
            len(manifests)
        )

        return manifests