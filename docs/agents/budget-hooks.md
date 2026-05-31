# Budgets and hooks

Agent loops can be bounded with `AgentBudget` and observed with `AgentHooks`.

## Budget limits

```python
from magic_llm.agent.types import AgentBudget

budget = AgentBudget(
    max_iterations=8,
    max_input_tokens=20_000,
    max_output_tokens=4_000,
    wall_clock_timeout=30.0,
)

response = client.run_agent(
    user_input="Use tools only if needed.",
    tools=[my_tool],
    budget=budget,
)
```

If a budget is exceeded, the loop raises `AgentBudgetExceeded` and calls `on_budget_exceeded` when hooks are provided.

## Hooks

`AgentHooks` defines observer callbacks:

- `on_iteration_start(iteration, state)`
- `on_llm_response(response, state)`
- `on_tool_start(tool_name, tool_call_id, arguments, state)`
- `on_tool_complete(result, state)`
- `on_loop_complete(final_response, state)`
- `on_budget_exceeded(budget_type, details)`

Example:

```python
class LoggingHooks:
    def on_iteration_start(self, iteration, state):
        print("iteration", iteration)

    def on_tool_start(self, tool_name, tool_call_id, arguments, state):
        print("tool", tool_name, arguments)

response = client.run_agent(
    user_input="Do the task.",
    tools=[my_tool],
    hooks=LoggingHooks(),
)
```

Hooks are observers. Do not mutate loop state inside hooks. Exceptions raised by hooks propagate and terminate the loop.

## Deduplication

```python
response = client.run_agent(
    user_input="Avoid repeating identical work.",
    tools=[my_tool],
    deduplicate=True,
)
```

With deduplication enabled, repeated tool-call fingerprints are skipped.
