"""Task subagents configuration for magic-llm.

magic-llm owns repo-level feature flags and defaults.
Application-level enable/disable is controlled by caller (magic-agents).

Feature flags default to disabled for safe rollout.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Feature flag: Enable subagents at repo-level
# When False, load_subagents() returns empty results
# Default: False for backward compatibility and safe rollout
ENABLE_SUBAGENTS: bool = False


# Feature flag: Enable nested LLM node execution
# When True, tasks with nested_tools config instantiate child AsyncAgentLoop
# When False, tasks with nested_config fall back to legacy callable pattern
# Default: False for backward compatibility and safe rollout
ENABLE_NESTED_LLM_NODES: bool = False


# Maximum global nesting depth (safety cap for unbounded nesting)
# Prevents runaway nesting across different task IDs
# Default: 10 (reasonable for most use cases)
MAX_GLOBAL_DEPTH: int = 10


# Maximum summary length for TaskResult
# Prevents token blowup from long child outputs
MAX_SUMMARY_LENGTH: int = 5000


# Default values for SubagentManifest fields
DEFAULT_TIMEOUT_SECONDS: int = 30
DEFAULT_MAX_CONCURRENCY: int = 5
DEFAULT_MAX_DEPTH: int = 3


def is_subagents_enabled() -> bool:
    """Check if subagents feature is enabled at repo-level.

    This is the magic-llm repo-level flag.
    Application-level control is handled by caller.

    Returns:
        True if ENABLE_SUBAGENTS is True.
    """
    return ENABLE_SUBAGENTS


def enable_subagents() -> None:
    """Enable subagents feature at repo-level.

    Called by magic-llm internals or for testing.
    """
    global ENABLE_SUBAGENTS
    ENABLE_SUBAGENTS = True
    logger.debug("Subagents feature enabled at repo-level")


def disable_subagents() -> None:
    """Disable subagents feature at repo-level.

    Called for rollback or testing.
    """
    global ENABLE_SUBAGENTS
    ENABLE_SUBAGENTS = False
    logger.debug("Subagents feature disabled at repo-level")


def is_nested_llm_nodes_enabled() -> bool:
    """Check if nested LLM node execution is enabled.

    When True, tasks with nested_tools config instantiate child AsyncAgentLoop.
    When False, tasks with nested_config fall back to legacy callable pattern.

    Returns:
        True if ENABLE_NESTED_LLM_NODES is True.
    """
    return ENABLE_NESTED_LLM_NODES


def enable_nested_llm_nodes() -> None:
    """Enable nested LLM node execution at repo-level.

    Called by magic-llm internals or for testing.
    """
    global ENABLE_NESTED_LLM_NODES
    ENABLE_NESTED_LLM_NODES = True
    logger.debug("Nested LLM node execution enabled at repo-level")


def disable_nested_llm_nodes() -> None:
    """Disable nested LLM node execution at repo-level.

    Called for rollback or testing.
    """
    global ENABLE_NESTED_LLM_NODES
    ENABLE_NESTED_LLM_NODES = False
    logger.debug("Nested LLM node execution disabled at repo-level")