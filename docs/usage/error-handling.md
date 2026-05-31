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

If the primary client fails, the fallback client can be used transparently.

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
