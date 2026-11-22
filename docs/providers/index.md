# Providers

Magic LLM has native engines and OpenAI-compatible provider adapters.

## Engine names

| Engine | Provider family | Chat | Tools | Vision input | Embeddings | TTS | STT | `client.list_models()` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai` | Official OpenAI | Yes | Yes | Yes | Yes | Async | Sync + async | Yes |
| `openai` + `base_url` | OpenAI-compatible APIs | Yes | Yes | OpenRouter only | Request sent; provider-dependent | Together sync + async, DeepInfra async | Fireworks async | Yes, if the provider supports [model listing](openai.md#model-discovery) |
| `anthropic` | Anthropic Claude | Yes | Yes | Yes | No | No | No | Yes |
| `google` | Google AI Studio / Gemini | Yes | Yes | Yes | No | Sync + async | No | Yes |
| `amazon` | AWS Bedrock / Polly | Yes | No | No | No | Polly sync only | No | No |
| `cloudflare` | Cloudflare Workers AI | Yes | No | No | No | No | No | No |
| `cohere` | Cohere | Yes | No | No | No | No | No | Yes |
| `azure` | Azure Speech | No | No | N/A | No | Async only | Async only | No ([Azure OpenAI/Foundry adapters](azure.md#model-discovery)) |

"Chat" covers `generate`, `stream_generate`, `async_generate`, and `async_stream_generate`. See [openai.md](openai.md) for the per-host OpenAI-compatible matrix, the Responses API (`endpoint="responses"`), and `private_transport`.

## Unsupported-feature errors

Unsupported features fail before any request is sent, but the errors differ:

| Feature | What you get |
| --- | --- |
| Tools on `amazon`, `cloudflare`, `cohere`, `azure` | `ChatException` with `error_code="TOOL_WIRING_NOT_IMPLEMENTED"`. No retry or fallback. |
| Images on a non-vision adapter | One attempt, no retries. A configured `fallback` handles the request; otherwise `ChatException` with `error_code="STREAM_GENERATION_ERROR"`, whose `__context__` is the `VISION_NOT_SUPPORTED` error. |
| Unsupported TTS/STT method | `ChatException` with `error_code="UNSUPPORTED_MEDIA_OPERATION"`, naming the supported alternative |
| Chat on `azure` | One attempt, no retries. A configured `fallback` handles the request; otherwise `ChatException` with `error_code="STREAM_GENERATION_ERROR"`, whose `__context__` is the `NotImplementedError`. |
| `list_models()` on `amazon`, `cloudflare`, or an `openai` client with `private_transport=True` | `NotImplementedError` |
| `embedding()` / `async_embedding()` on any engine except `openai` | Returns `None` (no error) |

See [retries](../usage/error-handling.md#retries) for how every other chat failure is retried.

## Common constructor shape

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
)
```

Provider-specific arguments are documented in the child pages. Shared engine options are passed to `MagicLLM(...)` as keyword arguments: `retries` (default 3), `headers`, `fallback`, and `callback`.

## OpenAI-compatible URL routing

These providers are selected through `engine="openai"` plus `base_url`: Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, and Together. Any other compatible endpoint uses the generic adapter.

```python
client = MagicLLM(
    engine="openai",
    model="provider-model-name",
    private_key="provider-key",
    base_url="https://provider.example/v1",
)
```

See [openai.md](openai.md) for the important `max_tokens` behavior.

First-class image generation/image output is not part of the current core API. `vision` means image input only.
