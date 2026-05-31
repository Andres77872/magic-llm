# Subagents

Subagents expose task-like callables as tools with manifests, concurrency limits, timeouts, and depth controls.

## Critical feature flag

Subagents are disabled by default:

```python
from magic_llm.agent.config import enable_subagents

enable_subagents()
```

If you call `load_subagents()` without enabling subagents, Magic LLM returns an empty bundle.

## Programmatic registration

```python
from magic_llm.agent.types import TaskManifest

manifest = TaskManifest(
    id="research.web",
    name="Web Research",
    description="Research a topic and return a concise summary.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    timeout_seconds=30,
    max_concurrency=5,
    max_depth=3,
)

async def research_web(query: str) -> str:
    return f"Research summary for {query}"

client.register_task(manifest, research_web)
```

Registered tasks can be used by the async agent loop through the internal task executor.

## YAML manifests

`load_subagents()` discovers files matching `*.agent.yaml`.

```yaml
apiVersion: magic-agents/v1
kind: TaskSubagent
id: research.web
name: Web Research
description: Research a topic and return a concise summary.
version: 1.0.0
input_schema:
  type: object
  properties:
    query:
      type: string
  required: [query]
timeout_seconds: 30
max_concurrency: 5
max_depth: 3
enabled: true
```

Load them with an explicit directory and callable registry:

```python
from pathlib import Path
from magic_llm.agent.config import enable_subagents

enable_subagents()

code_registry = {"research.web": research_web}
bundle = await client.load_subagents(Path("subagents"), code_registry=code_registry)
```

## Runtime safeguards

- `timeout_seconds`: per-task timeout.
- `max_concurrency`: parallel instances allowed.
- `max_depth`: recursion depth for this task.
- Global depth cap defaults to `MAX_GLOBAL_DEPTH = 10`.
- `reset_depths()` resets depth counters for a new graph execution.
