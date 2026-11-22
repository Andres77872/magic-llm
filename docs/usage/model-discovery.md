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

Discovery returns a list of `NormalizedDiscoveredModel` objects with these fields:

- `external_id`, `provider`, `display_name`, `description`
- `capabilities`: capability flags such as `capabilities.chat`
- `context_window`, `max_input_tokens`, `max_output_tokens`, `pricing` where the provider exposes them
- `raw`: the provider's original record

## Engine support for `client.list_models()`

| Engine | Result |
| --- | --- |
| `openai` (official, no `base_url`) | Lists `https://api.openai.com/v1/models` |
| `openai` + `base_url` | Lists the configured provider's models on the `base_url` origin ([how the URL is chosen](../providers/openai.md#model-discovery)) |
| `openai` with `private_transport=True` | Raises `NotImplementedError` |
| `anthropic` | Lists `https://api.anthropic.com/v1/models` |
| `cohere` | Lists `https://api.cohere.com/v1/models` |
| `google` | Lists `https://generativelanguage.googleapis.com/v1beta/models` |
| `azure` | Raises `ValueError`; the Azure discovery adapters need different credentials ([Azure](../providers/azure.md#model-discovery)) |
| `amazon`, `cloudflare` | Raises `NotImplementedError` |

## Using discovery adapters directly

To list models without a chat client, resolve and call an adapter yourself:

```python
from magic_llm.engine.discovery import get_adapter, resolve_discovery_engine

# Pick the adapter for an OpenAI-compatible base URL.
base_url = "https://api.groq.com/openai/v1"
name = resolve_discovery_engine("openai", base_url)  # "groq"
adapter = get_adapter(name)(api_key="gsk-your-key", base_url=base_url if name == "openai" else None)
models = adapter.discover()            # or: await adapter.async_discover()

# Google AI Studio
models = get_adapter("google")(api_key="google-api-key").discover()
```

`resolve_discovery_engine(engine, base_url)` matches the base URL host, including subdomains, against each adapter's hosts. Unknown hosts return `"openai"`. Pass `base_url` explicitly for those, because the generic adapter otherwise defaults to `api.openai.com`.

| Adapter | Hosts matched | Default listing URL |
| --- | --- | --- |
| `openai` | `api.openai.com` | `https://api.openai.com/v1/models` |
| `groq` | `api.groq.com` | `https://api.groq.com/openai/v1/models` |
| `deepinfra` | `api.deepinfra.com` | `https://api.deepinfra.com/v1/models` |
| `deepseek` | `api.deepseek.com` | `https://api.deepseek.com/v1/models` |
| `mistral` | `api.mistral.ai` | `https://api.mistral.ai/v1/models` |
| `together` | `api.together.xyz`, `api.together.ai` | `https://api.together.xyz/v1/models` |
| `openrouter` | `openrouter.ai` | OpenRouter models API |
| `sambanova` | `api.sambanova.ai` | SambaNova models API |
| `xai` | `api.x.ai` | `https://api.x.ai/v1/models` |
| `cerebras` | `api.cerebras.ai` | `https://api.cerebras.ai/v1/models` |
| `hyperbolic` | `api.hyperbolic.xyz` | `https://api.hyperbolic.xyz/v1/models` |
| `nebius` | `api.studio.nebius.ai`, `api.studio.nebius.com`, `api.tokenfactory.nebius.com` | `https://api.studio.nebius.ai/v1/models` |
| `novita` | `api.novita.ai` | `https://api.novita.ai/v3/openai/models` |
| `parasail` | `api.parasail.io` | `https://api.parasail.io/v1/models` |
| `perplexity` | `api.perplexity.ai` | `https://api.perplexity.ai/v1/models` |
| `anthropic` | none | `https://api.anthropic.com/v1/models` |
| `cohere` | `api.cohere.com`, `api.cohere.ai` | `https://api.cohere.com/v1/models` |
| `google` | none | `https://generativelanguage.googleapis.com/v1beta/models` |
| `azure` | none | `https://{resource_name}.openai.azure.com/openai/models` |
| `azure-foundry` | none | Azure management plane, per subscription and location |

Fireworks has a chat adapter but no discovery adapter, so it resolves to the generic `openai` adapter.

`magic_llm.engine.discovery.list_supported_engines()` returns the registered adapter names.

## Failure modes

Discovery errors are importable from `magic_llm.engine.discovery`:

- `DiscoveryAuthError`: HTTP 401 or 403, or a 400 the provider uses for a rejected key.
- `DiscoveryRateLimitError`: HTTP 429, with `retry_after` when the provider sends it.
- `DiscoveryError`: the provider is unreachable, returns 5xx or another 4xx, or sends a body that is not JSON.

An HTTP 404 returns an empty list. Treat discovery as a network call, not a static local registry.
