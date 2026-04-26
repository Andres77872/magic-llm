"""Subagent error taxonomy for magic-llm.

magic-llm owns ALL error types for subagent registration, validation,
and lookup. Runtime execution errors (depth, timeout) use TaskError from types.py.

This module provides:
- DuplicateSubagentError: Duplicate registration error (hard error)
- SubagentValidationError: Validation/binding error
- UnknownSubagentError: Unknown ID lookup error
- ManifestLoadError: YAML loading error (moved from loader.py for clarity)
"""
from typing import Optional


class DuplicateSubagentError(Exception):
    """Duplicate registration error.

    Raised when attempting to register a subagent with an ID that
    already exists in the registry. This is a hard error per spec.

    Attributes:
        agent_id: The duplicate agent ID.
        existing_source: Source file of existing registration.
        new_source: Source file of new registration attempt.
    """

    def __init__(
        self,
        agent_id: str,
        existing_source: Optional[str] = None,
        new_source: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.existing_source = existing_source
        self.new_source = new_source
        message = f"Duplicate registration: agent_id '{agent_id}' already registered"
        if existing_source:
            message += f" (existing: {existing_source})"
        if new_source:
            message += f" (new: {new_source})"
        message += ". Duplicate registration is not allowed."
        super().__init__(message)


class SubagentValidationError(Exception):
    """Validation error for subagent input or binding.

    Raised when:
    - Input validation fails against input_schema
    - Callable signature mismatches manifest schema
    - Required fields are missing

    Attributes:
        agent_id: The subagent ID being validated.
        message: Detailed error message.
        validation_type: Type of validation (input, signature, etc.).
    """

    def __init__(
        self,
        agent_id: str,
        message: str,
        validation_type: str = "input"
    ):
        self.agent_id = agent_id
        self.message = message
        self.validation_type = validation_type
        full_message = f"Validation error for subagent '{agent_id}' ({validation_type}): {message}"
        super().__init__(full_message)


class BinderValidationError(Exception):
    """Binder validation error for callable signature mismatch.

    Raised when Binder detects signature mismatch between callable
    and manifest input_schema.

    Attributes:
        manifest_id: The manifest ID being validated.
        missing_params: Set of missing parameters.
        expected_schema: The expected input_schema.
    """

    def __init__(
        self,
        manifest_id: str,
        missing_params: set[str],
        expected_schema: dict
    ):
        self.manifest_id = manifest_id
        self.missing_params = missing_params
        self.expected_schema = expected_schema
        message = (
            f"Binder validation error for '{manifest_id}': "
            f"callable missing required parameters: {missing_params}. "
            f"Expected schema: {expected_schema}"
        )
        super().__init__(message)


class UnknownSubagentError(Exception):
    """Unknown subagent ID error.

    Raised when attempting to resolve or invoke a subagent
    that is not registered.

    Attributes:
        agent_id: The unknown agent ID.
        registered_ids: List of registered IDs for debugging.
    """

    def __init__(
        self,
        agent_id: str,
        registered_ids: list[str] = []
    ):
        self.agent_id = agent_id
        self.registered_ids = registered_ids
        message = f"Unknown subagent_id '{agent_id}'."
        if registered_ids:
            message += f" Registered: {', '.join(registered_ids)}"
        else:
            message += " No subagents registered."
        super().__init__(message)


class ManifestLoadError(Exception):
    """Error loading manifest file.

    Attributes:
        file_path: Path to the problematic YAML file.
        error_details: Detailed error message.
    """

    def __init__(
        self,
        file_path,
        error_details: str,
    ):
        self.file_path = file_path
        self.error_details = error_details
        message = f"Failed to load manifest from {file_path}: {error_details}"
        super().__init__(message)