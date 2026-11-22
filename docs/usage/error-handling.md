# Error handling

Magic LLM normalizes provider failures through `ChatException` where possible and supports fallback clients, retry interception, metrics, and callbacks.

## Catch `ChatException`

```python
from magic_llm.exception.ChatException import ChatException

try:
    response = client.llm.generate(chat)
except ChatException as exc:
    print("LLM request failed:", exc)
```

Provider-specific HTTP errors may be wrapped by engine/adapters. Keep logs around provider, model, and request ID if your provider returns one.

## What an `HttpError` keeps

A provider's error body can echo the API key or the prompt, so it is never
kept, and no message is taken from it. The body is classified while it is in
hand, and the error keeps only:

| Attribute | Meaning |
| --- | --- |
| `status_code` | The HTTP status, when there was one |
| `code` | The classification: `PROVIDER_AUTHENTICATION_REJECTED`, `PROVIDER_QUOTA_EXCEEDED`, `PROVIDER_CONTEXT_EXCEEDED`, `PROVIDER_PERMISSION_DENIED`, `PROVIDER_RATE_LIMITED` or `PROVIDER_UNAVAILABLE` |
| `error_type` | A recognized typed kind (`magic_llm.util.provider_errors.ERROR_TYPES`), such as OpenRouter's `context_length_exceeded` or Anthropic's `overloaded_error`. Unrecognized values are dropped |
| `provider_code` | A recognized auth, quota or rate-limit code |
| `retry_after` | A numeric `retry_after` the body gave, in seconds |

`PROVIDER_CONTEXT_EXCEEDED` covers HTTP 413, the typed context errors and a
plain 400 whose body names the context window, so a caller can retry with a
smaller request instead of giving up. A rejected key behind a plain 400
(Google's "API key not valid") is classified as an authentication failure.

With a body, the text is the safe message for the classification and the
status; without one, it is the library's own wording. Errors the library
raises for a broken wire protocol are `ProviderProtocolError` (a
`ValueError`) and keep their wording through retries, as `ChatException`
does. `magic_llm.util.provider_errors.failure_message(exc)` gives the text
that is safe to log or show for any exception.

## OpenRouter

OpenRouter's usage, cache reads and writes (`cached_tokens_write`) and cost
(`usage.provider_extra["cost"]`, with a BYOK key's upstream charge added) are
read from the stream itself. A failure inside an HTTP 200 (a top-level
`error`, or a choice with `finish_reason: "error"`) fails the call with its
status and `error_type`.

## Retries

`generate`, `stream_generate`, `async_generate`, and `async_stream_generate` make up to `retries` attempts (constructor option, default 3), waiting 1 second between attempts. Each failed attempt calls the callback with `meta.status` set to `"ERROR: <message>"`.

How a failure is handled depends on the error:

| Error | Attempts | Fallback | What the caller gets |
| --- | --- | --- | --- |
| Transport failures, HTTP 408, 429, 5xx, and other errors | All `retries` | Yes | `ChatException` with `error_code="STREAM_GENERATION_ERROR"` |
| Rejections no retry can change: `VISION_NOT_SUPPORTED`, `NotImplementedError` (chat on `engine="azure"`) | One | Yes | `ChatException` with `error_code="STREAM_GENERATION_ERROR"` |
| HTTP 4xx other than 408 and 429 | One | No | The `HttpError`, unwrapped, with no error callback |
| `TOOL_WIRING_NOT_IMPLEMENTED`, `RequestValidationError`, private-output rejections | One | No | The original exception, with no error callback |

In the `STREAM_GENERATION_ERROR` cases, the last underlying error is the exception's `__context__`. `async_stream_generate` without a fallback yields one empty chunk whose `finish_reason` is `"error: <message>"` before raising.

With `provider_attempt_control`, retries and fallback follow the control contract instead; see [reliable application integration](../agents/production-integration.md).

## Fallback chain

```python
backup = MagicLLM(engine="openai", model="gpt-4o-mini", private_key="sk-backup")

client = MagicLLM(
    engine="openai",
    model="primary-model",
    private_key="sk-primary",
    fallback=backup,
)

response = client.llm.generate(chat)
```

When the primary client ends a call on an error that allows a fallback (see the table above), the same chat and options go to the fallback client, and its result is returned.

## Callbacks

```python
def on_chunk(chat, content, usage, model_name, meta):
    print(model_name, usage, meta)

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
    callback=on_chunk,
)
```

Callbacks are useful for logging, usage tracking, and UI updates. Keep callback code fast and defensive.

## Debug payload logging

Set these environment variables when you need request payload visibility:

- `MAGIC_LLM_DEBUG_PAYLOAD` — compact payload summary.
- `MAGIC_LLM_DEBUG_PAYLOAD_FULL` — full payload JSON.

Do not enable full payload logging in production if prompts or tool outputs contain sensitive data.
