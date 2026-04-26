"""Unit tests for ResultNormalizer — direct runtime contract tests.

This module tests ResultNormalizer behavior directly, not via TaskExecutor.
ResultNormalizer is part of the magic-llm runtime contract and should be
tested where it lives, not in magic-agents.
"""
import pytest

from magic_llm.agent import TaskManifest, TaskError, ResultNormalizer


def _make_manifest(
    id: str = "test.agent",
    name: str = "Test Agent",
    description: str = "Test",
    input_schema: dict = None,
) -> TaskManifest:
    """Helper to create a minimal TaskManifest for tests."""
    return TaskManifest(
        id=id,
        name=name,
        description=description,
        input_schema=input_schema or {"type": "object"},
    )


class TestResultNormalizerSuccessCases:
    """Tests for successful output normalization."""

    def test_normalize_string_output(self):
        """String output appears directly in summary."""
        manifest = _make_manifest()

        result = ResultNormalizer.normalize(
            raw_output="# Research Results\n\nFound 3 sources.",
            manifest=manifest,
            task_id="abc123",
        )

        assert result.task_id == "abc123"
        assert result.task_type == "test.agent"
        assert result.status == "ok"
        assert result.summary == "# Research Results\n\nFound 3 sources."
        assert result.error is None

    def test_normalize_dict_output(self):
        """Dict output converted to Markdown bullet list."""
        manifest = _make_manifest()
        raw_output = {
            "sources": ["source1", "source2", "source3"],
            "summary": "Key findings",
        }

        result = ResultNormalizer.normalize(
            raw_output=raw_output,
            manifest=manifest,
            task_id="test123",
        )

        assert result.status == "ok"
        assert "**sources**" in result.summary
        assert "**summary**" in result.summary

    def test_normalize_list_output(self):
        """List output converted to Markdown numbered list."""
        manifest = _make_manifest()
        raw_output = ["Item 1", "Item 2", "Item 3"]

        result = ResultNormalizer.normalize(
            raw_output=raw_output,
            manifest=manifest,
            task_id="list123",
        )

        assert result.status == "ok"
        assert "1. Item 1" in result.summary
        assert "2. Item 2" in result.summary
        assert "3. Item 3" in result.summary

    def test_normalize_int_output(self):
        """Int output converted to string."""
        manifest = _make_manifest()

        result = ResultNormalizer.normalize(
            raw_output=42,
            manifest=manifest,
            task_id="int123",
        )

        assert result.status == "ok"
        assert "42" in result.summary

    def test_normalize_none_output(self):
        """None output converted to 'None' string."""
        manifest = _make_manifest()

        result = ResultNormalizer.normalize(
            raw_output=None,
            manifest=manifest,
            task_id="none123",
        )

        assert result.status == "ok"
        assert "None" in result.summary


class TestResultNormalizerTruncation:
    """Tests for summary truncation at MAX_SUMMARY_LENGTH."""

    def test_truncate_long_output(self):
        """Output exceeding MAX_SUMMARY_LENGTH is truncated."""
        manifest = _make_manifest()
        long_text = "x" * 6000  # Exceeds MAX_SUMMARY_LENGTH (5000)

        result = ResultNormalizer.normalize(
            raw_output=long_text,
            manifest=manifest,
            task_id="long123",
        )

        # MAX_SUMMARY_LENGTH = 5000, plus truncation marker
        assert len(result.summary) <= 5050
        assert "[output truncated" in result.summary

    def test_output_within_limit_not_truncated(self):
        """Output within MAX_SUMMARY_LENGTH is not truncated."""
        manifest = _make_manifest()
        short_text = "x" * 100

        result = ResultNormalizer.normalize(
            raw_output=short_text,
            manifest=manifest,
            task_id="short123",
        )

        assert result.summary == short_text
        assert "[output truncated" not in result.summary


class TestResultNormalizerErrorCases:
    """Tests for error output normalization."""

    def test_normalize_with_error(self):
        """Error produces formatted Markdown error summary."""
        manifest = _make_manifest()
        error = TaskError(
            error_type=TaskError.VALIDATION,
            message="Missing required field",
            retryable=False,
        )

        result = ResultNormalizer.normalize(
            raw_output="ignored",
            manifest=manifest,
            task_id="err123",
            status="failed",
            error=error,
        )

        assert result.status == "failed"
        assert "## Task Failed" in result.summary
        assert "Missing required field" in result.summary
        assert "**Error Type**" in result.summary

    def test_normalize_timeout_error(self):
        """Timeout error produces timeout-specific summary."""
        manifest = _make_manifest()
        error = TaskError(
            error_type=TaskError.TIMEOUT,
            message="Execution exceeded 30s",
            retryable=True,
        )

        result = ResultNormalizer.normalize(
            raw_output="",
            manifest=manifest,
            task_id="timeout123",
            status="timeout",
            error=error,
        )

        assert result.status == "timeout"
        assert "## Task Timeout" in result.summary
        assert "**Retry Recommended**: Yes" in result.summary

    def test_normalize_depth_limit_error(self):
        """Depth limit error produces cancelled summary."""
        manifest = _make_manifest()
        error = TaskError(
            error_type=TaskError.DEPTH_LIMIT,
            message="Depth 5 exceeded max_depth 3",
            retryable=False,
        )

        result = ResultNormalizer.normalize(
            raw_output="",
            manifest=manifest,
            task_id="depth123",
            status="cancelled",
            error=error,
        )

        assert result.status == "cancelled"
        assert "## Task Cancelled" in result.summary
        assert "**Retry Recommended**: No" in result.summary

    def test_error_without_status_change(self):
        """Providing error with status='ok' still shows error format."""
        manifest = _make_manifest()
        error = TaskError(
            error_type=TaskError.EXECUTION,
            message="Something went wrong",
            retryable=True,
        )

        # Even if status is 'ok', error forces error formatting
        result = ResultNormalizer.normalize(
            raw_output="output",
            manifest=manifest,
            task_id="conflict123",
            status="ok",
            error=error,
        )

        # Error takes precedence - should show error format
        assert result.status == "ok"  # Status preserved
        assert result.error is not None
        assert "## Task Error" in result.summary


class TestResultNormalizerMarkdownFormatting:
    """Tests for internal Markdown formatting helpers."""

    def test_dict_to_markdown_nested(self):
        """Nested dict is formatted with indentation."""
        data = {"outer": {"inner": "value"}}

        markdown = ResultNormalizer._dict_to_markdown(data)

        assert "**outer**:" in markdown
        assert "**inner**: value" in markdown

    def test_dict_to_markdown_with_list_values(self):
        """Dict with list values formats list inline."""
        data = {"items": ["a", "b", "c"], "count": 3}

        markdown = ResultNormalizer._dict_to_markdown(data)

        assert "**items**: a, b, c" in markdown
        assert "**count**: 3" in markdown

    def test_list_to_markdown_nested(self):
        """Nested list of dicts formatted with indentation."""
        data = [
            {"name": "Item 1", "count": 5},
            {"name": "Item 2", "count": 10},
        ]

        markdown = ResultNormalizer._list_to_markdown(data)

        assert "1." in markdown
        assert "**name**" in markdown
        assert "**count**" in markdown

    def test_list_to_markdown_nested_lists(self):
        """Nested lists within list items."""
        data = [["a", "b"], ["c", "d"]]

        markdown = ResultNormalizer._list_to_markdown(data)

        # Each item is numbered, nested items are indented
        assert "1." in markdown
        assert "2." in markdown


class TestResultNormalizerEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_empty_string_output(self):
        """Empty string is valid output."""
        manifest = _make_manifest()

        result = ResultNormalizer.normalize(
            raw_output="",
            manifest=manifest,
            task_id="empty123",
        )

        assert result.status == "ok"
        assert result.summary == ""

    def test_empty_dict_output(self):
        """Empty dict produces empty bullet list."""
        manifest = _make_manifest()

        result = ResultNormalizer.normalize(
            raw_output={},
            manifest=manifest,
            task_id="emptydict123",
        )

        assert result.status == "ok"
        assert result.summary == ""

    def test_empty_list_output(self):
        """Empty list produces empty numbered list."""
        manifest = _make_manifest()

        result = ResultNormalizer.normalize(
            raw_output=[],
            manifest=manifest,
            task_id="emptylist123",
        )

        assert result.status == "ok"
        assert result.summary == ""

    def test_unicode_output(self):
        """Unicode characters preserved in summary."""
        manifest = _make_manifest()
        unicode_text = "Results: café, ñandú, 日本語"

        result = ResultNormalizer.normalize(
            raw_output=unicode_text,
            manifest=manifest,
            task_id="unicode123",
        )

        assert result.status == "ok"
        assert "café" in result.summary
        assert "ñandú" in result.summary
        assert "日本語" in result.summary

    def test_boolean_output(self):
        """Boolean output converted to string."""
        manifest = _make_manifest()

        result_true = ResultNormalizer.normalize(
            raw_output=True,
            manifest=manifest,
            task_id="bool_true",
        )
        result_false = ResultNormalizer.normalize(
            raw_output=False,
            manifest=manifest,
            task_id="bool_false",
        )

        assert result_true.status == "ok"
        assert "True" in result_true.summary
        assert result_false.status == "ok"
        assert "False" in result_false.summary