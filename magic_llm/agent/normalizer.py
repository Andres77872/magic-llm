"""Result normalizer: Convert raw output to plain-text Markdown.

Main return surface is Markdown summary, not raw JSON.
Part of the task/subagent runtime output contract in magic-llm.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from magic_llm.agent.types import TaskError, TaskManifest, TaskResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ResultNormalizer:
    """Converts raw callable output to plain-text Markdown summary.

    Main return surface is Markdown, not JSON.
    Parent LLM receives concise summary for context injection.

    This class is part of the magic-llm runtime contract, moved from
    magic-agents to consolidate ALL task/subagent runtime behavior.
    """

    MAX_SUMMARY_LENGTH: int = 5000  # Prevent token blowup

    @classmethod
    def normalize(
        cls,
        raw_output: Any,
        manifest: TaskManifest,
        task_id: str,
        status: str = "ok",
        error: TaskError | None = None,
    ) -> TaskResult:
        """Normalize callable output to TaskResult.

        Converts output to plain-text Markdown summary.
        Handles str, dict, and other output types.

        Args:
            raw_output: Output from decorated callable.
            manifest: TaskManifest for context (id, name, description).
            task_id: Unique task invocation ID.
            status: Result status ("ok", "failed", "timeout", "cancelled").
            error: Optional TaskError if status != "ok".

        Returns:
            TaskResult with Markdown summary.
        """
        if error is not None or status != "ok":
            # Error case: use provided error message
            summary = cls._format_error_summary(manifest, error, status)
        else:
            # Success case: convert output to Markdown
            content = cls._convert_to_markdown(raw_output)
            summary = cls._truncate_summary(content)

        return TaskResult(
            task_id=task_id,
            task_type=manifest.id,
            status=status,
            summary=summary,
            error=error,
        )

    @classmethod
    def _convert_to_markdown(cls, raw_output: Any) -> str:
        """Convert raw output to Markdown string.

        Args:
            raw_output: Output from callable.

        Returns:
            Markdown-formatted string.
        """
        if isinstance(raw_output, str):
            # Already a string - return directly
            return raw_output

        if isinstance(raw_output, dict):
            # Structured output → format as Markdown bullet list
            return cls._dict_to_markdown(raw_output)

        if isinstance(raw_output, list):
            # List output → format as numbered list
            return cls._list_to_markdown(raw_output)

        # Other types → convert to string
        return str(raw_output)

    @classmethod
    def _dict_to_markdown(cls, data: dict) -> str:
        """Convert dict to Markdown bullet list.

        Args:
            data: Dict with key-value pairs.

        Returns:
            Markdown bullet list string.
        """
        lines = []
        for key, value in data.items():
            if isinstance(value, list):
                # Nested list
                items = ', '.join(str(v) for v in value)
                lines.append(f"- **{key}**: {items}")
            elif isinstance(value, dict):
                # Nested dict (recursively format)
                nested = cls._dict_to_markdown(value)
                lines.append(f"- **{key}**:")
                for nested_line in nested.split('\n'):
                    lines.append(f"  {nested_line}")
            else:
                lines.append(f"- **{key}**: {value}")

        return '\n'.join(lines)

    @classmethod
    def _list_to_markdown(cls, data: list) -> str:
        """Convert list to Markdown numbered list.

        Args:
            data: List of items.

        Returns:
            Markdown numbered list string.
        """
        lines = []
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                # Dict item → format inline
                formatted = cls._dict_to_markdown(item)
                lines.append(f"{i}.")
                for subline in formatted.split('\n'):
                    lines.append(f"   {subline}")
            elif isinstance(item, list):
                # Nested list
                nested = cls._list_to_markdown(item)
                lines.append(f"{i}.")
                for subline in nested.split('\n'):
                    lines.append(f"   {subline}")
            else:
                lines.append(f"{i}. {item}")

        return '\n'.join(lines)

    @classmethod
    def _truncate_summary(cls, content: str) -> str:
        """Truncate content to MAX_SUMMARY_LENGTH.

        Adds truncation marker if content exceeds limit.

        Args:
            content: Full content string.

        Returns:
            Truncated string with marker if needed.
        """
        if len(content) <= cls.MAX_SUMMARY_LENGTH:
            return content

        truncated = content[:cls.MAX_SUMMARY_LENGTH]
        return truncated + "... [output truncated, see full logs]"

    @classmethod
    def _format_error_summary(
        cls,
        manifest: TaskManifest,
        error: TaskError | None,
        status: str,
    ) -> str:
        """Format error summary as Markdown.

        Args:
            manifest: TaskManifest for context.
            error: TaskError with details.
            status: Result status.

        Returns:
            Markdown error summary.
        """
        if error is None:
            return f"## Task {status.capitalize()}\n\nTask '{manifest.name}' ended with status: {status}"

        error_sections = {
            "failed": "Task Failed",
            "timeout": "Task Timeout",
            "cancelled": "Task Cancelled",
        }

        heading = error_sections.get(status, "Task Error")

        summary_lines = [
            f"## {heading}",
            "",
            f"Task '{manifest.name}' could not complete.",
            "",
            f"**Error Type**: {error.error_type}",
            f"**Message**: {error.message}",
            f"**Retry Recommended**: {'Yes' if error.retryable else 'No'}",
        ]

        return '\n'.join(summary_lines)