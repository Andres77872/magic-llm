"""Tests for ToolExecutor max_content_size enforcement — Task 1.1.

Tests cover:
- Content under limit passes through unchanged
- Content over limit gets truncated with [TRUNCATED] suffix
- Per-tool name override takes precedence over global limit
"""

import json

import pytest

from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


class TestMaxContentSize:
    """Task 1.1: max_content_size enforcement in ToolExecutor."""

    def test_content_under_limit_passes_through(self):
        """Content <= limit passes unchanged, is_error=False.

        NOTE: _serialize_output uses json.dumps which wraps strings in quotes,
        so for a string return value, the content is the JSON string.
        """
        executor = ToolExecutor(max_content_size=50)

        def small_tool():
            return {"status": "ok", "value": "hello"}  # dict, JSON-serialized ~30 chars

        executor.register("small", small_tool)
        result = executor.execute(_make_call("small"))

        assert result.is_error is False
        assert "[TRUNCATED]" not in result.content
        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"

    def test_max_content_size_truncates_oversized(self):
        """Content > limit gets truncated with [TRUNCATED] suffix, is_error=True."""
        executor = ToolExecutor(max_content_size=20)

        def large_tool():
            return {"data": "x" * 100}  # JSON-serialized will be > 20 chars

        executor.register("large", large_tool)
        result = executor.execute(_make_call("large"))

        assert result.is_error is True
        assert result.content.endswith("[TRUNCATED]")
        # The total content should be 20 + len("[TRUNCATED]") = 31
        assert len(result.content) == 20 + len("[TRUNCATED]")

    def test_max_content_size_per_tool_override(self):
        """Per-tool override takes precedence over global limit.

        generate_image has per-tool max of 2000, so 3000-char string gets truncated to 2000.
        """
        executor = ToolExecutor(
            max_content_size=50000,
            max_content_sizes={"generate_image": 20},  # Small for test
        )

        def image_tool():
            return {"url": "/images/img.webp", "data": "x" * 100}

        executor.register("generate_image", image_tool)
        result = executor.execute(_make_call("generate_image"))

        # With json.dumps, the dict is ~130 chars. With max_content_size=20, truncated.
        assert result.is_error is True
        assert result.content.endswith("[TRUNCATED]")

    def test_per_tool_override_does_not_affect_other_tools(self):
        """Per-tool override only affects the named tool, not others."""
        executor = ToolExecutor(
            max_content_size=20,  # Small global limit to force truncation
            max_content_sizes={"generate_image": 500},  # Generous per-tool limit
        )

        def image_tool():
            return {"url": "/images/img.webp", "width": 1024, "height": 1024}
            # JSON: ~65 chars — under 500 per-tool limit

        def browse_tool():
            return {"results": [{"url": "https://example.com"}] * 20}
            # JSON: much larger than 20 global limit

        executor.register("generate_image", image_tool)
        executor.register("search", browse_tool)

        image_result = executor.execute(_make_call("generate_image"))
        browse_result = executor.execute(_make_call("search"))

        # Image gen: uses per-tool override (500), so ~65 chars passes
        assert image_result.is_error is False
        assert "[TRUNCATED]" not in image_result.content
        parsed = json.loads(image_result.content)
        assert "/images/img.webp" in parsed["url"]

        # Browsing: uses global (20), so much larger content gets truncated
        assert browse_result.is_error is True
        assert browse_result.content.endswith("[TRUNCATED]")

    def test_max_content_size_async_execution(self):
        """Async execution also enforces max_content_size."""
        executor = ToolExecutor(max_content_size=20)

        async def large_tool():
            return {"data": "x" * 100}

        executor.register("large", large_tool)

        import asyncio
        result = asyncio.run(executor.execute_async(_make_call("large")))

        assert result.is_error is True
        assert result.content.endswith("[TRUNCATED]")
        assert len(result.content) == 20 + len("[TRUNCATED]")

    def test_max_content_size_default_50000(self):
        """Default max_content_size is 50000, so normal output passes."""
        executor = ToolExecutor()

        def normal_tool():
            return {"status": "ok"}

        executor.register("normal", normal_tool)
        result = executor.execute(_make_call("normal"))

        assert result.is_error is False
        assert result.content == json.dumps({"status": "ok"})
