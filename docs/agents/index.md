# Agents

Magic LLM includes ReAct-style agent loops on top of the chat API. Agents repeatedly call an LLM, extract tool calls, execute tools, append tool results, and stop when the provider indicates completion or a budget is exceeded.

## Main entry points

```python
client.run_agent(...)
client.run_agent_stream(...)
await client.run_agent_async(...)
async for chunk in client.run_agent_stream_async(...): ...
```

## Minimal callable tool

```python
from magic_llm import MagicLLM

client = MagicLLM(engine="openai", model="gpt-4o-mini", private_key="sk-your-key")

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

response = client.run_agent(
    user_input="Add 17 and 25, then explain the result.",
    tools=[add],
    tool_choice="auto",
    max_iterations=4,
)

print(response.content)
```

## Tool support by provider

Tool calling is supported for OpenAI, Anthropic, Google, and many OpenAI-compatible adapters. Amazon Bedrock, Cohere, and Cloudflare do not support tool calling in Magic LLM v0.1.36.

## Agent loop behavior

- Sync loop executes tools through thread-based parallelism.
- Async loop supports sync and async callables and executes parallel tool calls with `asyncio.gather`.
- `AgentBudget` can limit iterations, input tokens, output tokens, and wall-clock time.
- Lifecycle hooks observe loop progress.
- `deduplicate=True` avoids repeated tool-call fingerprints.
- Agent loop instances are guarded against concurrent `.run()` / `.stream()` calls.

Read next: [tool calling](tool-calling.md), [budgets and hooks](budget-hooks.md), [subagents](subagents.md).
