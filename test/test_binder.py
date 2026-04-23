"""Unit tests for Binder — manifest + callable joiner.

Tests cover:
- Signature validation success
- Signature mismatch error
- Missing params detection
- Extra required params detection
- Non-async callable rejection
- Type hint compatibility warnings
"""
import pytest
import logging

from magic_llm.agent.binder import Binder
from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.errors import BinderValidationError, SubagentValidationError


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_manifest(
    id: str = "test_agent",
    input_schema: dict = None,
) -> SubagentManifest:
    """Create a SubagentManifest with given input_schema."""
    if input_schema is None:
        input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    return SubagentManifest(
        id=id,
        name="Test Agent",
        description="A test agent",
        version="1.0.0",
        input_schema=input_schema,
    )


# ─── Signature Validation Success Tests ───────────────────────────────────────


class TestBinderJoinSuccess:
    """Binder.join() signature validation success."""

    def test_join_returns_tuple(self):
        """Binder.join() returns (manifest, callable) tuple."""
        manifest = _make_manifest(id="agent")

        async def valid_func(query: str) -> str:
            return query

        result = Binder.join(manifest, valid_func)

        assert result[0] == manifest
        assert result[1] == valid_func

    def test_join_single_required_param(self):
        """join() succeeds with single required param matching schema."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        async def func(query: str) -> str:
            return query

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest
        assert bound_callable == func

    def test_join_multiple_required_params(self):
        """join() succeeds with multiple required params matching schema."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "depth": {"type": "integer"},
                },
                "required": ["query", "depth"],
            }
        )

        async def func(query: str, depth: int) -> dict:
            return {"query": query, "depth": depth}

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest.id == "test_agent"

    def test_join_optional_params_allowed(self):
        """join() allows callable to have optional params (with defaults)."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        # Callable has required param + optional param with default
        async def func(query: str, optional: str = "default") -> str:
            return query

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest

    def test_join_schema_with_no_required(self):
        """join() succeeds when schema has no required fields."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"optional": {"type": "string"}},
                "required": [],  # No required fields
            }
        )

        async def func() -> str:
            return "no params needed"

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest


# ─── Signature Mismatch Tests ───────────────────────────────────────────────


class TestBinderSignatureMismatch:
    """Binder.join() signature mismatch errors."""

    def test_missing_required_param_raises_error(self):
        """join() raises BinderValidationError for missing required param."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        # Callable missing required 'query' param
        async def func() -> str:
            return "missing query"

        with pytest.raises(BinderValidationError) as exc_info:
            Binder.join(manifest, func)

        assert exc_info.value.manifest_id == "test_agent"
        assert "query" in exc_info.value.missing_params

    def test_missing_multiple_required_params(self):
        """join() raises error for multiple missing required params."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "depth": {"type": "integer"},
                },
                "required": ["query", "depth"],
            }
        )

        async def func() -> str:
            return "missing all"

        with pytest.raises(BinderValidationError) as exc_info:
            Binder.join(manifest, func)

        assert exc_info.value.manifest_id == "test_agent"
        assert "query" in exc_info.value.missing_params
        assert "depth" in exc_info.value.missing_params

    def test_partial_missing_params(self):
        """join() raises error when some required params missing."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "depth": {"type": "integer"},
                },
                "required": ["query", "depth"],
            }
        )

        # Callable has 'query' but missing 'depth'
        async def func(query: str) -> str:
            return query

        with pytest.raises(BinderValidationError) as exc_info:
            Binder.join(manifest, func)

        assert exc_info.value.manifest_id == "test_agent"
        assert exc_info.value.missing_params == {"depth"}
        assert "query" not in exc_info.value.missing_params

    def test_error_contains_expected_schema(self):
        """BinderValidationError contains expected_schema for debugging."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        async def func() -> str:
            return "missing"

        with pytest.raises(BinderValidationError) as exc_info:
            Binder.join(manifest, func)

        assert exc_info.value.expected_schema == manifest.input_schema


# ─── Extra Required Params Tests ───────────────────────────────────────────────


class TestBinderExtraRequiredParams:
    """Callable with extra required params not in schema."""

    def test_extra_required_param_raises_error(self):
        """join() raises SubagentValidationError for extra required param."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        # Callable has extra required param 'extra_param'
        async def func(query: str, extra_param: str) -> str:
            return query

        with pytest.raises(SubagentValidationError) as exc_info:
            Binder.join(manifest, func)

        assert exc_info.value.agent_id == "test_agent"
        assert exc_info.value.validation_type == "signature"
        assert "extra_param" in str(exc_info.value)

    def test_extra_param_with_default_allowed(self):
        """join() allows extra param if it has a default value."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        # Callable has extra param with default — should be allowed
        async def func(query: str, extra_param: str = "default") -> str:
            return query

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest


# ─── Non-Async Callable Tests ───────────────────────────────────────────────


class TestBinderNonAsyncCallable:
    """Non-async callable rejection."""

    def test_non_async_raises_error(self):
        """join() raises SubagentValidationError for non-async callable."""
        manifest = _make_manifest()

        # Sync function, not async
        def sync_func(query: str) -> str:
            return query

        with pytest.raises(SubagentValidationError) as exc_info:
            Binder.join(manifest, sync_func)

        assert exc_info.value.agent_id == "test_agent"
        assert exc_info.value.validation_type == "signature"
        assert "async" in str(exc_info.value).lower()

    def test_lambda_raises_error(self):
        """join() raises error for lambda (non-async)."""
        manifest = _make_manifest()

        with pytest.raises(SubagentValidationError) as exc_info:
            Binder.join(manifest, lambda query: query)

        assert "async" in str(exc_info.value).lower()


# ─── Type Hint Compatibility Tests ─────────────────────────────────────────────


class TestBinderTypeHints:
    """Type hint compatibility warnings."""

    def test_type_hint_mismatch_logs_warning(self, caplog):
        """join() logs warning for type hint mismatch (doesn't raise)."""
        caplog.set_level(logging.WARNING)

        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        # Callable has int type hint but schema says string
        async def func(query: int) -> str:  # Wrong type hint!
            return str(query)

        # Should succeed (warning only, not error)
        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest
        # Should have logged a warning about type mismatch
        assert any("mismatch" in record.message.lower() for record in caplog.records)

    def test_no_type_hints_succeeds(self):
        """join() succeeds when callable has no type hints."""
        manifest = _make_manifest()

        # No type hints on callable
        async def func(query):
            return query

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest

    def test_correct_type_hints_no_warning(self, caplog):
        """join() succeeds silently when type hints match schema."""
        caplog.set_level(logging.WARNING)

        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        async def func(query: str) -> str:  # Correct type hint
            return query

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest
        # No warning about type mismatch
        assert not any("mismatch" in record.message.lower() for record in caplog.records)


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestBinderEdgeCases:
    """Edge case handling."""

    def test_callable_with_self_param(self):
        """join() ignores 'self' param for bound methods."""
        manifest = _make_manifest()

        class MyClass:
            async def method(self, query: str) -> str:
                return query

        obj = MyClass()
        bound_manifest, bound_callable = Binder.join(manifest, obj.method)

        assert bound_manifest == manifest

    def test_callable_with_kwargs(self):
        """join() ignores 'kwargs' param."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        async def func(query: str, **kwargs) -> str:
            return query

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest

    def test_callable_with_args(self):
        """join() ignores 'args' param."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )

        async def func(query: str, *args) -> str:
            return query

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest

    def test_empty_schema_no_required(self):
        """join() succeeds with empty input_schema (no properties)."""
        manifest = _make_manifest(
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            }
        )

        async def func() -> str:
            return "ok"

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest

    def test_return_type_hint_not_checked(self):
        """join() doesn't validate return type hint."""
        manifest = _make_manifest()

        # Return type is dict but callable returns str — not checked
        async def func(query: str) -> dict:
            return {"result": query}  # Returns dict as declared

        bound_manifest, bound_callable = Binder.join(manifest, func)

        assert bound_manifest == manifest