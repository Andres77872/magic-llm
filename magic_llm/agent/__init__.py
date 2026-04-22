# magic_llm.agent — Optional ReAct-style agent loop package.
#
# This package is an OPTIONAL add-on. Importing magic_llm alone MUST NOT
# import any agent modules. Consumers must explicitly import from
# magic_llm.agent to use the agent loop.
#
# No eager imports of submodules — the package is loaded only when
# explicitly imported by the consumer.
#
# Builtin tools (web_search, web_scrape) are available via:
#   from magic_llm.agent.builtin import web_search, web_scrape, get_browsing_tools

# Re-export builtin tools for convenience
# Note: This does NOT eagerly import the builtin module — it only provides
# the import path. Actual import happens when consumer uses these names.
from magic_llm.agent.builtin import (
    web_search,
    web_scrape,
    get_browsing_adapter,
    get_browsing_tools,
    get_browsing_tool_functions,
)

# Task/Subagent Runtime Types (for MagicLLM.register_task API)
from magic_llm.agent.types import (
    TaskManifest,
    TaskResult,
    TaskError,
    TaskBudget,
    TaskState,
)

# Task/Subagent Runtime Components
from magic_llm.agent.task_executor import TaskExecutor
from magic_llm.agent.normalizer import ResultNormalizer

__all__ = [
    # Builtin tools
    "web_search",
    "web_scrape",
    "get_browsing_adapter",
    "get_browsing_tools",
    "get_browsing_tool_functions",
    # Task/Subagent Runtime Types
    "TaskManifest",
    "TaskResult",
    "TaskError",
    "TaskBudget",
    "TaskState",
    # Task/Subagent Runtime Components
    "TaskExecutor",
    "ResultNormalizer",
]
