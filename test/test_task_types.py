"""Unit tests for task runtime types — TaskManifest, TaskResult, TaskError.

These types are part of the magic-llm runtime contract and should be tested
where they live, not in magic-agents. Tests cover validation, serialization,
and contract behavior.
"""
import json
import pytest

from magic_llm.agent import TaskManifest, TaskResult, TaskError


class TestTaskManifest:
    """Tests for TaskManifest Pydantic model — runtime subset of SubagentManifest."""

    def test_valid_manifest_defaults(self):
        """Valid manifest with minimal fields uses defaults."""
        manifest = TaskManifest(
            id="test.agent",
            name="Test Agent",
            description="A test subagent",
            input_schema={"type": "object"},
        )

        assert manifest.id == "test.agent"
        assert manifest.name == "Test Agent"
        assert manifest.timeout_seconds == 30  # Default
        assert manifest.max_concurrency == 5  # Default
        assert manifest.max_depth == 3  # Default

    def test_valid_manifest_with_custom_policy(self):
        """Manifest with custom policy fields."""
        manifest = TaskManifest(
            id="slow.agent",
            name="Slow Agent",
            description="A slow agent",
            input_schema={"type": "object"},
            timeout_seconds=120,
            max_concurrency=1,
            max_depth=1,
        )

        assert manifest.timeout_seconds == 120
        assert manifest.max_concurrency == 1
        assert manifest.max_depth == 1

    def test_invalid_id_pattern_uppercase(self):
        """Invalid ID pattern with uppercase raises ValidationError."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TaskManifest(
                id="Invalid-ID",  # Contains uppercase
                name="Test",
                description="Test",
                input_schema={"type": "object"},
            )

    def test_invalid_id_pattern_spaces(self):
        """Invalid ID pattern with spaces raises ValidationError."""
        with pytest.raises(Exception):
            TaskManifest(
                id="test agent",  # Contains space
                name="Test",
                description="Test",
                input_schema={"type": "object"},
            )

    def test_valid_id_patterns(self):
        """Valid ID patterns accepted."""
        valid_ids = [
            "simple",
            "with.dots",
            "with-hyphens",
            "with_underscores",
            "mixed.dots-hyphens_underscores",
            "numbers123",
        ]

        for id in valid_ids:
            manifest = TaskManifest(
                id=id,
                name="Test",
                description="Test",
                input_schema={"type": "object"},
            )
            assert manifest.id == id

    def test_timeout_bounds(self):
        """Timeout must be between 1 and 600."""
        # Valid bounds
        manifest_min = TaskManifest(
            id="test",
            name="Test",
            description="Test",
            input_schema={"type": "object"},
            timeout_seconds=1,
        )
        assert manifest_min.timeout_seconds == 1

        manifest_max = TaskManifest(
            id="test",
            name="Test",
            description="Test",
            input_schema={"type": "object"},
            timeout_seconds=600,
        )
        assert manifest_max.timeout_seconds == 600

        # Invalid bounds
        with pytest.raises(Exception):
            TaskManifest(
                id="test",
                name="Test",
                description="Test",
                input_schema={"type": "object"},
                timeout_seconds=0,  # Below min
            )

        with pytest.raises(Exception):
            TaskManifest(
                id="test",
                name="Test",
                description="Test",
                input_schema={"type": "object"},
                timeout_seconds=601,  # Above max
            )

    def test_concurrency_bounds(self):
        """Concurrency must be between 1 and 20."""
        # Valid bounds
        manifest_min = TaskManifest(
            id="test",
            name="Test",
            description="Test",
            input_schema={"type": "object"},
            max_concurrency=1,
        )
        assert manifest_min.max_concurrency == 1

        manifest_max = TaskManifest(
            id="test",
            name="Test",
            description="Test",
            input_schema={"type": "object"},
            max_concurrency=20,
        )
        assert manifest_max.max_concurrency == 20

        # Invalid bounds
        with pytest.raises(Exception):
            TaskManifest(
                id="test",
                name="Test",
                description="Test",
                input_schema={"type": "object"},
                max_concurrency=0,
            )

        with pytest.raises(Exception):
            TaskManifest(
                id="test",
                name="Test",
                description="Test",
                input_schema={"type": "object"},
                max_concurrency=21,
            )

    def test_depth_bounds(self):
        """Depth must be between 1 and 10."""
        # Valid bounds
        manifest_min = TaskManifest(
            id="test",
            name="Test",
            description="Test",
            input_schema={"type": "object"},
            max_depth=1,
        )
        assert manifest_min.max_depth == 1

        manifest_max = TaskManifest(
            id="test",
            name="Test",
            description="Test",
            input_schema={"type": "object"},
            max_depth=10,
        )
        assert manifest_max.max_depth == 10

        # Invalid bounds
        with pytest.raises(Exception):
            TaskManifest(
                id="test",
                name="Test",
                description="Test",
                input_schema={"type": "object"},
                max_depth=0,
            )

        with pytest.raises(Exception):
            TaskManifest(
                id="test",
                name="Test",
                description="Test",
                input_schema={"type": "object"},
                max_depth=11,
            )


class TestTaskError:
    """Tests for TaskError Pydantic model — structured error representation."""

    def test_error_classification_constants(self):
        """Error type constants are defined."""
        assert TaskError.VALIDATION == "ValidationError"
        assert TaskError.TIMEOUT == "TimeoutError"
        assert TaskError.EXECUTION == "ExecutionError"
        assert TaskError.DEPTH_LIMIT == "DepthLimitError"

    def test_validation_error(self):
        """ValidationError with retryable=False."""
        error = TaskError(
            error_type=TaskError.VALIDATION,
            message="Invalid input schema",
            retryable=False,
        )

        assert error.error_type == "ValidationError"
        assert error.message == "Invalid input schema"
        assert error.retryable is False

    def test_timeout_error(self):
        """TimeoutError with retryable=True."""
        error = TaskError(
            error_type=TaskError.TIMEOUT,
            message="Task exceeded 30s",
            retryable=True,
        )

        assert error.error_type == "TimeoutError"
        assert error.retryable is True

    def test_execution_error(self):
        """ExecutionError with retryable=True."""
        error = TaskError(
            error_type=TaskError.EXECUTION,
            message="Runtime error in callable",
            retryable=True,
        )

        assert error.error_type == "ExecutionError"

    def test_depth_limit_error(self):
        """DepthLimitError with retryable=False."""
        error = TaskError(
            error_type=TaskError.DEPTH_LIMIT,
            message="Depth exceeded",
            retryable=False,
        )

        assert error.error_type == "DepthLimitError"

    def test_custom_error_type(self):
        """Custom error type allowed (not restricted to constants)."""
        error = TaskError(
            error_type="CustomError",
            message="Something custom",
            retryable=True,
        )

        assert error.error_type == "CustomError"

    def test_default_retryable(self):
        """Default retryable is False."""
        error = TaskError(
            error_type=TaskError.EXECUTION,
            message="No retryable specified",
        )

        assert error.retryable is False


class TestTaskResult:
    """Tests for TaskResult Pydantic model — structured output envelope."""

    def test_success_result(self):
        """Successful result with no error."""
        result = TaskResult(
            task_id="abc123",
            task_type="test.agent",
            status="ok",
            summary="# Results\n\nFound 3 sources.",
        )

        assert result.task_id == "abc123"
        assert result.task_type == "test.agent"
        assert result.status == "ok"
        assert result.summary == "# Results\n\nFound 3 sources."
        assert result.error is None

    def test_failed_result_with_error(self):
        """Failed result with TaskError."""
        error = TaskError(
            error_type=TaskError.EXECUTION,
            message="Something went wrong",
            retryable=True,
        )

        result = TaskResult(
            task_id="abc123",
            task_type="test.agent",
            status="failed",
            summary="## Task Failed",
            error=error,
        )

        assert result.status == "failed"
        assert result.error.error_type == TaskError.EXECUTION
        assert result.error.retryable is True

    def test_timeout_result(self):
        """Timeout result."""
        error = TaskError(
            error_type=TaskError.TIMEOUT,
            message="Timeout after 30s",
            retryable=True,
        )

        result = TaskResult(
            task_id="abc123",
            task_type="test.agent",
            status="timeout",
            summary="## Task Timeout",
            error=error,
        )

        assert result.status == "timeout"
        assert result.error.error_type == TaskError.TIMEOUT

    def test_cancelled_result(self):
        """Cancelled result (depth limit exceeded)."""
        error = TaskError(
            error_type=TaskError.DEPTH_LIMIT,
            message="Depth exceeded",
            retryable=False,
        )

        result = TaskResult(
            task_id="abc123",
            task_type="test.agent",
            status="cancelled",
            summary="## Task Cancelled",
            error=error,
        )

        assert result.status == "cancelled"
        assert result.error.error_type == TaskError.DEPTH_LIMIT

    def test_status_literal_validation(self):
        """Status must be one of the literal values."""
        valid_statuses = ["ok", "failed", "timeout", "cancelled"]

        for status in valid_statuses:
            result = TaskResult(
                task_id="test",
                task_type="test.agent",
                status=status,
                summary=f"Status: {status}",
            )
            assert result.status == status

        # Invalid status
        with pytest.raises(Exception):
            TaskResult(
                task_id="test",
                task_type="test.agent",
                status="invalid",  # Not in literal
                summary="Invalid status",
            )

    def test_to_tool_result_json(self):
        """JSON serialization for ToolResult.content injection."""
        result = TaskResult(
            task_id="abc123",
            task_type="test.agent",
            status="ok",
            summary="# Test Summary",
        )

        json_str = result.to_tool_result_json()

        # Parse to verify structure
        parsed = json.loads(json_str)
        assert parsed["task_id"] == "abc123"
        assert parsed["status"] == "ok"
        assert parsed["summary"] == "# Test Summary"
        assert "error" not in parsed  # Excluded when None

    def test_to_tool_result_json_with_error(self):
        """JSON serialization includes error when present."""
        error = TaskError(
            error_type=TaskError.EXECUTION,
            message="Failed",
            retryable=True,
        )

        result = TaskResult(
            task_id="abc123",
            task_type="test.agent",
            status="failed",
            summary="## Task Failed",
            error=error,
        )

        json_str = result.to_tool_result_json()
        parsed = json.loads(json_str)

        assert parsed["status"] == "failed"
        assert parsed["error"]["error_type"] == "ExecutionError"
        assert parsed["error"]["message"] == "Failed"
        assert parsed["error"]["retryable"] is True

    def_to_tool_result_json_excludes_none_error = (
        "JSON serialization excludes error field when None."
    )

    def test_model_dump_json_format(self):
        """model_dump_json returns valid JSON."""
        result = TaskResult(
            task_id="abc123",
            task_type="test.agent",
            status="ok",
            summary="# Test",
        )

        # Should be valid JSON
        json_str = result.model_dump_json(exclude_none=True)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestTaskResultTaskErrorIntegration:
    """Tests for TaskResult + TaskError interaction."""

    def result_preserves_error_reference(self):
        """Result preserves exact error reference."""
        error = TaskError(
            error_type=TaskError.VALIDATION,
            message="Invalid",
            retryable=False,
        )

        result = TaskResult(
            task_id="test",
            task_type="test.agent",
            status="failed",
            summary="Error",
            error=error,
        )

        assert result.error is error
        assert result.error.message == "Invalid"

    def test_error_serialization_in_result(self):
        """Error is serialized as nested object in result JSON."""
        error = TaskError(
            error_type=TaskError.TIMEOUT,
            message="Timeout",
            retryable=True,
        )

        result = TaskResult(
            task_id="test",
            task_type="test.agent",
            status="timeout",
            summary="Timeout",
            error=error,
        )

        json_str = result.to_tool_result_json()
        parsed = json.loads(json_str)

        # Error is nested object
        assert isinstance(parsed["error"], dict)
        assert parsed["error"]["error_type"] == "TimeoutError"