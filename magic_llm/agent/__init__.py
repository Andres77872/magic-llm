# magic_llm.agent — Optional ReAct-style agent loop package.
#
# This package is an OPTIONAL add-on. Importing magic_llm alone MUST NOT
# import any agent modules. Consumers must explicitly import from
# magic_llm.agent to use the agent loop.
#
# No eager imports of submodules — the package is loaded only when
# explicitly imported by the consumer.
#
# Application-specific tools such as browsing are intentionally not exported
# from magic-llm. Consumers compose those tools in their own app/package layer.

# Task/Subagent Runtime Types (for MagicLLM.register_task API)
from magic_llm.agent.types import (
    TaskManifest,
    TaskResult,
    TaskError,
    TaskBudget,
    TaskState,
    AgentBudget,
    AgentState,
)

# Task/Subagent Runtime Components
from magic_llm.agent.task_executor import TaskExecutor, reset_depths

# Global Depth Helpers (for nested LLM node depth tracking)
from magic_llm.agent._loop_shared import (
    GLOBAL_DEPTH,
    PARENT_BUDGET,
    PARENT_STATE,
    PARENT_HOOKS,
    DEPTH,
    get_global_depth,
    increment_global_depth,
    decrement_global_depth,
    reset_global_depth,
)

# Result Normalizer
from magic_llm.agent.normalizer import ResultNormalizer

# Library-owned generic AgentHooks persistence implementation. Service/API side
# effects stay behind injected sinks; magic-llm does not import app modules.
from magic_llm.agent.persistence_hooks import (
    AgentPersistenceHooks,
    AgentPersistenceSink,
)

# ─── Subagent Architecture (magic-llm owns ALL) ──────────────────────────────

# Definitions (YAML manifest model)
from magic_llm.agent.definitions import (
    SubagentManifest,
    BoundSubagent,
)

# Loader (YAML discovery)
from magic_llm.agent.loader import (
    ManifestLoader,
    ManifestLoadError,
)

# Registry (instance-scoped, NO global state)
from magic_llm.agent.registry import (
    SubagentRegistry,
    RegistryBackend,
)

# Binder (manifest + callable joiner)
from magic_llm.agent.binder import Binder

# Bundle (schema/callable container)
from magic_llm.agent.bundle import SubagentBundle

# Decorator (explicit registry dict, NO global state)
from magic_llm.agent.decorator import (
    subagent,
    register_callable,
)

# Config (repo-level flags and defaults)
from magic_llm.agent.config import (
    is_subagents_enabled,
    enable_subagents,
    disable_subagents,
    ENABLE_SUBAGENTS,
    MAX_SUMMARY_LENGTH,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DEPTH,
    # Nested LLM node config
    is_nested_llm_nodes_enabled,
    enable_nested_llm_nodes,
    disable_nested_llm_nodes,
    ENABLE_NESTED_LLM_NODES,
    MAX_GLOBAL_DEPTH,
)

# Errors (registration/validation/lookup errors)
from magic_llm.agent.errors import (
    DuplicateSubagentError,
    SubagentValidationError,
    BinderValidationError,
    UnknownSubagentError,
)

__all__ = [
    # Task/Subagent Runtime Types
    "TaskManifest",
    "TaskResult",
    "TaskError",
    "TaskBudget",
    "TaskState",
    "AgentBudget",
    "AgentState",
    # Task/Subagent Runtime Components
    "TaskExecutor",
    "ResultNormalizer",
    "AgentPersistenceHooks",
    "AgentPersistenceSink",
    "reset_depths",
    # Global Depth Helpers
    "GLOBAL_DEPTH",
    "PARENT_BUDGET",
    "PARENT_STATE",
    "PARENT_HOOKS",
    "DEPTH",
    "get_global_depth",
    "increment_global_depth",
    "decrement_global_depth",
    "reset_global_depth",
    # ─── Subagent Architecture (magic-llm owns ALL) ───
    # Definitions
    "SubagentManifest",
    "BoundSubagent",
    # Loader
    "ManifestLoader",
    "ManifestLoadError",
    # Registry
    "SubagentRegistry",
    "RegistryBackend",
    # Binder
    "Binder",
    # Bundle
    "SubagentBundle",
    # Decorator
    "subagent",
    "register_callable",
    # Config
    "is_subagents_enabled",
    "enable_subagents",
    "disable_subagents",
    "ENABLE_SUBAGENTS",
    "MAX_SUMMARY_LENGTH",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_DEPTH",
    # Nested LLM node config
    "is_nested_llm_nodes_enabled",
    "enable_nested_llm_nodes",
    "disable_nested_llm_nodes",
    "ENABLE_NESTED_LLM_NODES",
    "MAX_GLOBAL_DEPTH",
    # Errors
    "DuplicateSubagentError",
    "SubagentValidationError",
    "BinderValidationError",
    "UnknownSubagentError",
]
