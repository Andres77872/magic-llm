"""Subagent definitions: YAML manifest model and protocols.

magic-llm owns ALL definition architecture. This module provides:
- SubagentManifest: YAML manifest model with identity, schema, and policies
- BoundSubagent: Protocol for bound subagent exposed as tool

This is the canonical definition layer, repo-agnostic (no filesystem conventions).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol

from pydantic import BaseModel, Field, field_validator


class SubagentManifest(BaseModel):
    """YAML manifest for task-backed subagent registration.

    This is the machine source of truth for identity, schema, and policies.
    magic-llm owns ALL definition architecture.

    Loaded from YAML files matching pattern: *.agent.yaml

    Attributes:
        apiVersion: YAML versioning field (magic-agents/v1).
        kind: Resource type field (TaskSubagent).
        id: Unique identifier matching pattern ^[a-z0-9._-]+$.
        name: Human-readable name for display/logging.
        description: When-to-use summary for routing/delegation.
        version: Semantic version (pattern ^\\d+\\.\\d+\\.\\d+$).
        input_schema: JSON Schema for input validation.
        output_schema: Optional output schema.
        timeout_seconds: Per-task timeout (default: 30s, range: 1-600).
        max_concurrency: Concurrent instances allowed (default: 5, range: 1-20).
        max_depth: Recursion depth limit (default: 3, range: 1-10).
        model_override: Optional model override (e.g., "gpt-4.1-mini").
        instruction_file: Optional markdown reference (supportive).
        enabled: Whether subagent is active (default: True).
        source_file: Source file path (set by loader, not in YAML).
    """

    # Identity
    apiVersion: Literal["magic-agents/v1"] = "magic-agents/v1"
    kind: Literal["TaskSubagent"] = "TaskSubagent"
    id: str = Field(..., pattern=r'^[a-z0-9._-]+$')  # Stable registry ID
    name: str  # Human-readable name
    description: str  # When-to-use summary for routing/delegation
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')  # Semver

    # Schema
    input_schema: dict[str, Any]  # JSON Schema for input validation
    output_schema: Optional[dict[str, Any]] = None  # Optional output schema

    # Execution Policy
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_concurrency: int = Field(default=5, ge=1, le=20)
    max_depth: int = Field(default=3, ge=1, le=10)

    # Optional Overrides
    model_override: Optional[str] = None  # e.g., "gpt-4.1-mini"
    instruction_file: Optional[Path] = None  # Markdown reference (supportive)

    # State
    enabled: bool = True

    # Source tracking (set by loader, not in YAML)
    source_file: Optional[Path] = None

    @property
    def tool_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible tool schema.

        Tool name = manifest id (stable registry key).

        Returns:
            Dict with type='function' and function definition.
        """
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.description,
                "parameters": self.input_schema
            }
        }

    @field_validator('instruction_file', mode='before')
    @classmethod
    def validate_instruction_file(cls, v: Any) -> Optional[Path]:
        """Convert string to Path if provided.

        Args:
            v: Input value (string, Path, or None).

        Returns:
            Path instance or None.
        """
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v


class BoundSubagent(Protocol):
    """Protocol for bound subagent exposed as tool.

    Follows FetchToolCallable pattern: tool_schema + tool_callable.
    Used by bundle and registry for tool injection.
    """

    @property
    def tool_schema(self) -> dict[str, Any]:
        """OpenAI-compatible tool schema."""
        ...

    @property
    def tool_callable(self) -> Callable:
        """Async callable for execution."""
        ...