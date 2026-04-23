"""Binder: Join manifest with callable for magic-llm registration.

Validates callable signature matches manifest input_schema.
Returns manifest + callable tuple for MagicLLM.register_task().
"""
from __future__ import annotations

import inspect
import logging
from typing import Callable, Tuple, get_type_hints

from magic_llm.agent.definitions import SubagentManifest
from magic_llm.agent.errors import SubagentValidationError, BinderValidationError

logger = logging.getLogger(__name__)


class Binder:
    """Join manifest + callable → tuple for magic-llm registration.

    Validates callable signature matches manifest input_schema.
    Returns (manifest, callable) tuple for MagicLLM.register_task().

    NOTE: Semaphore and safeguards are handled by magic-llm's TaskExecutor.
    This binder only validates and returns the components for registration.

    Example:
        >>> manifest = SubagentManifest(id="search", input_schema={"query": {...}})
        >>> async def search(query: str) -> dict: ...
        >>> bound_manifest, bound_callable = Binder.join(manifest, search)
        >>> # Pass to MagicLLM.register_task()
    """

    @staticmethod
    def join(
        manifest: SubagentManifest,
        callable: Callable
    ) -> Tuple[SubagentManifest, Callable]:
        """Join manifest with callable for magic-llm registration.

        Validates signature and returns tuple for registration.

        Args:
            manifest: SubagentManifest with schema and policy.
            callable: Decorated async function.

        Returns:
            Tuple of (manifest, callable) for MagicLLM.register_task().

        Raises:
            SubagentValidationError: If callable is not async.
            BinderValidationError: If signature mismatches schema.
        """
        # Validate callable is async
        if not inspect.iscoroutinefunction(callable):
            raise SubagentValidationError(
                agent_id=manifest.id,
                message=f"Callable must be async function, got {type(callable).__name__}",
                validation_type="signature"
            )

        # Validate signature matches input_schema
        Binder._validate_signature(manifest, callable)

        logger.debug(
            "Bound subagent '%s' with callable %s for magic-llm registration",
            manifest.id,
            callable.__name__
        )

        # Return tuple for magic-llm registration
        # magic-llm's TaskExecutor handles semaphore, depth, timeout, etc.
        return manifest, callable

    @staticmethod
    def _validate_signature(
        manifest: SubagentManifest,
        callable: Callable
    ) -> None:
        """Validate callable signature matches input_schema.

        Checks that:
        - Required schema properties are callable parameters
        - Callable doesn't have extra required params
        - Type hints are compatible (if available)

        Args:
            manifest: SubagentManifest with input_schema.
            callable: Async function to validate.

        Raises:
            BinderValidationError: If mismatch found.
        """
        # Extract callable parameters
        sig = inspect.signature(callable)
        params = sig.parameters

        # Get schema required properties
        schema_props = manifest.input_schema.get("properties", {})
        schema_required = set(manifest.input_schema.get("required", []))

        # Get callable params (excluding 'self' and internal params)
        callable_params = set()
        for name, param in params.items():
            if name in ('self', 'kwargs', 'args'):
                continue
            callable_params.add(name)

        # Check: All schema required properties must be callable params
        missing_params = schema_required - callable_params
        if missing_params:
            raise BinderValidationError(
                manifest_id=manifest.id,
                missing_params=missing_params,
                expected_schema=manifest.input_schema
            )

        # Check: Callable shouldn't have extra required params (no default)
        # that aren't in schema
        extra_required = set()
        for name, param in params.items():
            if name in ('self', 'kwargs', 'args'):
                continue
            if param.default is inspect.Parameter.empty:
                # This param is required by callable
                if name not in schema_props and name not in schema_required:
                    extra_required.add(name)

        if extra_required:
            raise SubagentValidationError(
                agent_id=manifest.id,
                message=f"Callable has extra required params not in schema: {extra_required}",
                validation_type="signature"
            )

        # Type hint validation (optional, warn only)
        try:
            hints = get_type_hints(callable)
            Binder._check_type_compatibility(manifest, hints)
        except Exception:
            # Type hints not available, skip
            pass

    @staticmethod
    def _check_type_compatibility(
        manifest: SubagentManifest,
        hints: dict
    ) -> None:
        """Check type hints are compatible with schema types.

        Logs warnings for mismatches, doesn't raise errors.

        Args:
            manifest: SubagentManifest with input_schema.
            hints: Type hints from callable.
        """
        schema_props = manifest.input_schema.get("properties", {})

        for param_name, hint_type in hints.items():
            if param_name in ('return', 'self', 'kwargs', 'args'):
                continue

            schema_prop = schema_props.get(param_name, {})
            schema_type = schema_prop.get("type")

            # Map Python types to JSON Schema types
            py_to_json = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }

            expected_json_type = py_to_json.get(hint_type)
            if expected_json_type and schema_type:
                if expected_json_type != schema_type:
                    logger.warning(
                        "Type hint mismatch for '%s.%s': "
                        "callable has %s (maps to %s), schema has %s",
                        manifest.id,
                        param_name,
                        hint_type.__name__,
                        expected_json_type,
                        schema_type
                    )