"""Decorator for runtime callable binding.

@subagent decorator registers callable in user-provided registry dict.
NO GLOBAL STATE. Registry dict passed explicitly by caller.

magic-llm owns decorator architecture. This follows the explicit-registry
clarification: decorator populates user-provided dict, NOT hidden global.

Example:
    # User creates registry dict
    code_registry = {}

    # Decorator populates it
    @subagent("research.web", registry=code_registry)
    async def research_web(query: str) -> str:
        return execute_agent_loop(...)

    # Later: pass to load_subagents()
    client.load_subagents(manifest_dir, code_registry)
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def subagent(
    agent_id: str,
    registry: Optional[Dict[str, Callable]] = None,
) -> Callable:
    """Decorator for registering callable in explicit registry dict.

    NO GLOBAL STATE. Registry dict passed by user.

    Usage:
        code_registry = {}

        @subagent("research.web", registry=code_registry)
        async def research_web(query: str) -> str:
            # Agent loop execution implementation
            ...

        # Later: pass to load_subagents()
        client.load_subagents(manifest_dir, code_registry)

    Args:
        agent_id: Stable registry ID (must match manifest.id).
        registry: User-provided dict to populate (REQUIRED).

    Returns:
        Decorator function that registers the callable.

    Raises:
        ValueError: If registry is None (defensive programming).
    """
    if registry is None:
        logger.warning(
            "Decorator @subagent('%s') called without registry parameter. "
            "Callables will NOT be registered. "
            "Pass explicit registry dict: @subagent('%s', registry=my_dict)",
            agent_id,
            agent_id
        )

    def decorator(func: Callable) -> Callable:
        if registry is not None:
            registry[agent_id] = func
            logger.debug(
                "Registered callable '%s' in registry dict",
                agent_id
            )
        func._subagent_id = agent_id  # Tag for introspection
        return func

    return decorator


def register_callable(
    agent_id: str,
    callable: Callable,
    registry: Dict[str, Callable],
) -> None:
    """Manually register a callable in registry dict (alternative to decorator).

    Useful for programmatic registration without decorator syntax.

    Args:
        agent_id: Subagent ID.
        callable: Async callable for execution.
        registry: User-provided dict to populate.

    Example:
        >>> code_registry = {}
        >>> async def my_func(query: str) -> str: ...
        >>> register_callable("custom.id", my_func, code_registry)
    """
    registry[agent_id] = callable
    callable._subagent_id = agent_id
    logger.debug(
        "Manually registered callable '%s' in registry dict",
        agent_id
    )