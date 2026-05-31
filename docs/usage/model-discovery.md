# Model discovery

Use `client.list_models()` and `client.async_list_models()` to fetch normalized model metadata.

```python
from magic_llm import MagicLLM

client = MagicLLM(engine="openai", private_key="sk-your-key")

models = client.list_models()
for model in models:
    print(model.external_id, model.capabilities.chat)
```

Async:

```python
models = await client.async_list_models()
```

## Response model

Discovery returns a list of `NormalizedDiscoveredModel` objects with normalized fields such as:

- `external_id`
- capabilities flags through `model.capabilities`
- optional context/pricing metadata where available

## Supported and unsupported engines

Model discovery is supported for OpenAI, Anthropic, Google, Cohere, and many OpenAI-compatible adapters.

Unsupported engines:

- `amazon`
- `cloudflare`

Calling discovery on unsupported engines raises `NotImplementedError`.

## Failure modes

Discovery can raise provider-specific discovery errors for authentication, rate limits, not found, or provider API failures. Treat discovery as a network call, not a static local registry.
