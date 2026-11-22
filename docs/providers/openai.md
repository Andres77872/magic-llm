# OpenAI and OpenAI-compatible providers

Use `engine="openai"` for the official OpenAI API and for OpenAI-compatible endpoints.

## Official OpenAI

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
)
```

## OpenAI-compatible endpoints

```python
client = MagicLLM(
    engine="openai",
    model="llama-3.1-8b-instant",
    private_key="gsk-your-key",
    base_url="https://api.groq.com/openai/v1",
)
```

The `base_url` hostname selects a provider adapter. The match is exact, so subdomains and proxies fall back to the generic adapter. An explicit `base_url` always owns routing, including the provider's API prefix. Unrecognized hosts use the generic OpenAI adapter.

| Provider | Matched host | Vision input | TTS | STT | Discovery adapter |
| --- | --- | --- | --- | --- | --- |
| OpenAI | default, or `api.openai.com` (`/v1`) | Yes | Async | Sync + async | `openai` |
| Groq | `api.groq.com` | No | No | No | `groq` |
| SambaNova | `api.sambanova.ai` | No | No | No | `sambanova` |
| OpenRouter | `openrouter.ai` | Yes | No | No | `openrouter` |
| Mistral | `api.mistral.ai` | No | No | No | `mistral` |
| Fireworks | `api.fireworks.ai` | No | No | Async, `whisper-v3` and `whisper-v3-turbo` only | generic `openai` |
| DeepSeek | `api.deepseek.com` | No | No | No | `deepseek` |
| DeepInfra | `api.deepinfra.com`, path `/v1` or below | No | Async | No | `deepinfra` |
| Together | `api.together.xyz` | No | Sync + async | No | `together` |
| Any other host | none | No | Async (see note) | No | By host (xAI, Cerebras, Hyperbolic, Nebius, Novita, Parasail, Perplexity), else generic `openai` |

Every adapter supports chat, streaming, async chat, async streaming, and tool calling. Tools are passed through in the OpenAI format, and the provider decides whether the model accepts them.

"No" in the vision, TTS, or STT columns means the call is rejected before any request is sent; see [unsupported-feature errors](index.md#unsupported-feature-errors).

> **Generic-host TTS:** an unrecognized host is flagged as not supporting async TTS, but the generic adapter's `async_audio_speech()` does not check that flag. It still posts to `{base_url}/audio/speech`. Don't rely on this; the check may be enforced later.

## Critical `max_tokens` behavior

On the Chat Completions path for the official `api.openai.com` endpoint only, Magic LLM transforms request payloads:

- `max_tokens` becomes `max_completion_tokens`.
- If both `max_tokens` and `max_completion_tokens` are present, `max_completion_tokens` wins and `max_tokens` is removed.
- Other OpenAI-compatible providers keep `max_tokens` unchanged.

```python
response = client.llm.generate(chat, max_tokens=512)
```

On official OpenAI this is sent as `max_completion_tokens=512`; on Groq, Mistral, Together, and similar endpoints it remains `max_tokens=512`. With `endpoint="responses"` it is sent as `max_output_tokens=512`.

## Responses API

Pass `endpoint="responses"` to call `/responses` instead of `/chat/completions` (the default is `endpoint="chat_completions"`):

```python
client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
    endpoint="responses",
)
```

`generate`, `stream_generate`, `async_generate`, `async_stream_generate`, and all `run_agent*` methods keep the same inputs and response models. The adapter maps the request:

- Messages become Responses `input` items. Assistant turns replay their stored Responses output items, so encrypted reasoning and tool calls survive agent iterations without provider-side storage.
- `max_tokens`/`max_completion_tokens` become `max_output_tokens`, and `reasoning_effort` becomes `reasoning.effort`. Call-time options override client defaults.
- `response_format` becomes `text.format`, and `verbosity` becomes `text.verbosity`.
- Function tools and `tool_choice` are converted to the Responses shape.
- `store` defaults to `False`, and `reasoning.encrypted_content` is added to `include`.
- Only `n=1` is supported. Other values raise `ValueError`.
- For `gpt-6` models on official OpenAI, `temperature`, `top_p`, and logprob options are dropped unless reasoning effort is `"none"`.

A stream that ends without a terminal Responses event raises `ProviderProtocolError`.

## Private transport for caller-supplied endpoints

`private_transport=True` hardens chat calls to an endpoint and key supplied at request time, for example a user-configured provider:

```python
client = MagicLLM(
    engine="openai",
    model="vendor/model-id",
    private_key=user_supplied_key,
    base_url="https://llm.vendor.example/v1",
    private_transport=True,
)

response = await client.llm.async_generate(chat)
```

With it enabled:

- `base_url` must be public HTTPS on port 443, with no credentials, query, or fragment. Localhost and private, loopback, link-local, or reserved IP addresses are rejected at construction with `HttpError`. DNS answers are also checked at connect time and must all be public.
- The generic OpenAI adapter is always used, even for known hosts, and the model ID is sent as given.
- Only async chat runs: `async_generate`, `async_stream_generate`, and the async agent methods. `generate` and `stream_generate` fail.
- Requests do not follow redirects, ignore proxy environment variables, use no automatic retries at the HTTP layer, and cap responses at 16 MiB. Raw provider error bodies are not surfaced.
- `stream_options` usage reporting is not requested, and Responses calls do not request encrypted reasoning.
- Responses and streaming deltas are checked for credential echoes before they reach callbacks or callers. An optional `private_limits` profile (`magic_llm.util.private_limits.PrivateRequestLimits`) binds the model, budgets and timeout per attempt; it cannot be used with a fallback.
- The value must be a real `bool`. Anything else raises `ValueError`.

Private transport applies to chat only. Embeddings, audio, and `list_models()` use the ordinary HTTP client.

## Model discovery

`client.list_models()` and `await client.async_list_models()` select the listing adapter using the configured provider origin. Known providers can have a listing path different from their chat path; unknown compatible endpoints use their own base URL plus `/models`.

See [model discovery](../usage/model-discovery.md) for the full list of adapters.

## Embeddings

`client.llm.embedding(...)` and `client.llm.async_embedding(...)` post to `{base_url}/embeddings` for every adapter, using the client's `model`. Support is confirmed for OpenAI, DeepInfra, Together, Mistral, and Fireworks; for other hosts it depends on the provider. See [embeddings](../usage/embeddings.md).
