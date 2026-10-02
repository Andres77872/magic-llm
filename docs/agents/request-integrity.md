# Complete tool results and mandatory request validation

Application-owned metadata catalogs can expose an ordinary callable that returns
selected full definitions. Magic LLM supplies generic loop, output and validation
seams; the host still owns record validation, selection scope and retention. An
embedded agent JSON can contain every definition without any new database,
registry, filesystem reader or injected resolver.

## Validate every request before spending it

All four `MagicLLM.run_agent*` wrappers, `AgentLoop` and `AsyncAgentLoop` accept
`request_guard=context -> None`. This synchronous callback runs before each
provider request, after current conversation/tool assembly and outside best-effort
lifecycle hooks. Exceptions stop the loop; they are not converted into tool error
results. The callback is a control option and never becomes a provider field.

`AgentRequestContext` from `magic_llm.agent.request` exposes `chat`, `messages`,
`tools`, `tool_choice`, `provider`, `model`, `generation_options`, and
`estimated_input_tokens()`. The estimate includes nested tool arguments/results,
opaque replay data and normalized schemas. It is an estimate rather than a
provider tokenizer guarantee. The host supplies a finite model/context policy and
response reserve; `AgentBudget.max_input_tokens` remains a cumulative usage limit.

A guarded chat requires complete context. `ModelChat.require_complete_context()`
prevents `get_messages()` from silently dropping any history or half of a tool
exchange when its configured limit is exceeded. If `max_input_tokens` is absent,
the host guard must enforce its own finite bound. Each loop clones initial chat.

Register a second check for the **actual final mapped provider JSON**:

```python
import json
from magic_llm.exception.ChatException import ChatException

# Application-selected conservative bound, not a provider token claim.
max_request_bytes = 24000

def request_guard(context):
    def final_payload_guard(payload):
        if len(json.dumps(payload).encode("utf-8")) > max_request_bytes:
            raise ChatException("Complete request does not fit", "APP_CONTEXT_LIMIT")
    context.chat.set_provider_payload_guard(final_payload_guard)

response = await client.run_agent_async(
    user_input="Use relevant definitions to answer the request.",
    tools=host_schemas,
    tool_functions=host_functions,
    request_guard=request_guard,
    max_iterations=8,
    builtin_todo_tools=False,
)
```

Supported adapters call `chat.validate_provider_payload(payload)` after mapping,
before debug output or transport: OpenAI chat and Responses, compatible adapters
using the base path, DeepInfra/SambaNova final overrides, native Anthropic and
Google sync/async. Bedrock's baseline payload is checked too, but its package path
still rejects tool requests. No provider/model certification or support for native
hosted Skills is implied. Inspect real request sequences for the chosen endpoint.

Final payload failures raise `RequestValidationError` from
`magic_llm.exception.ChatException`. Its outward text is generic and safe, while
`validation_error`, `__cause__`, `code`, `error_code`, and a copied host `outcome`
retain application error mapping. BaseChat generation/stream wrappers immediately
rethrow it without retry, callback telemetry, fallback or another transport.
Repeated preparation cannot consume credits after a terminal fit failure.

## Return complete structured data

`ToolExecutor.serialize_output(output)` returns the exact **untruncated** ordinary
tool serialization. It uses `json.dumps` (including its Unicode escaping) with a
string fallback for non-JSON objects. Pass a structured object once; passing an
already-serialized JSON string causes normal JSON string encoding.

`executor.content_limit(name)` resolves per-tool overrides over global limits.
`client.tool_content_limit(name, tool_executor_options=...)` resolves the same
run-local options for the client's internally registered executor. Supply
`tool_executor=` when the loop uses an explicit executor. Storage byte bounds and
provider token counts do not replace this serialized-character limit.

```python
from magic_llm.agent.tool_executor import ToolExecutor

executor = ToolExecutor(max_content_size=50000)
executor.require_complete_output("read_definitions")
executor.register("read_definitions", read_definitions)
```

A host-owned callable may also set `_require_complete_output = True` before
registration. This is useful when public wrappers construct the executor.
Protected success and normalized error results return either complete content or
a compact `is_error=True` `ToolOutputLimitError`, never a partial JSON/prompt body
or `[TRUNCATED]` fragment. Reject incompatible tiny limits during host preflight;
a limit too small for the atomic error itself fails terminally. Unprotected tools
retain their prior truncation behavior. Repeated/deduplicated successful calls
still return complete content under the current call ID.

Use `executor.registered_names()` and `client.registered_tool_names()` for
read-only collision checks, including a pre-existing task name that has no visible
schema in the current graph. These views do not expose mutable registries.

## Separate inference from observers

`tool_result_observer=result -> ToolResult` is an optional synchronous transform
on wrappers and loops. It receives a deep copy used only by `on_tool_complete`;
the original canonical result remains intact for model continuation and dedup.
The host may replace bodies with safe counts/IDs while retaining status/error
metadata. Other hook state snapshots can still contain full canonical history;
sanitize those in the host relay as well.

`context.chat.set_observer_projection(projector)` configures callback-only chat
history for BaseChat's configured application callback. The synchronous projector
receives a detached ModelChat copy without control callbacks and must return a
ModelChat. Projection failure suppresses that callback; it never falls back to
canonical history. Without a configured projector, existing callbacks keep their
ordinary chat contract. The loop clone preserves this projection.

Guarded provider debug payloads (including the full-dump environment flag) emit
only counts/sizes rather than message previews or full bodies. This protects
pre-request provider diagnostics, not arbitrary application logging. Hosts must
also separate observer persistence from ephemeral model configuration and filter
stale identified exchanges without breaking unrelated parallel tool pairs.

## Canonical replay

Native Anthropic preparation copies input, preserves system guidance, translates
assistant tool calls to `tool_use`, and pairs native results. Native Google
translates assistant calls to `functionCall` and preserves the entire native
`gemini_parts` array, including signed text, function calls and signature-only
parts. Native parts are kept outside ordinary response/chunk serialization and
replayed once in their original order. Stream chunks accumulate native parts
without merging signed parts; canonical calls retain their separate IDs. This
follows Google's requirement to return signatures in their original parts.
[Google thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).
Stream loops use extracted Google calls to continue even when its native finish
reason is STOP. Anthropic argument deltas are accumulated once, and empty-input
calls remain represented. These fixes are validated with fake transports; no live
provider calls are required by the tests.
