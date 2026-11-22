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
| Any other host | none | No | Async, unverified (see below) | No | By host (xAI, Cerebras, Hyperbolic, Nebius, Novita, Parasail, Perplexity), else generic `openai` |

Every adapter supports chat, streaming, async chat, async streaming, and tool calling. Tools are passed through in the OpenAI format, and the provider decides whether the model accepts them.

"No" in the vision, TTS, or STT columns means the call is rejected before any request is sent; see [unsupported-feature errors](index.md#unsupported-feature-errors).

For hosts without a dedicated adapter, `async_audio_speech()` posts the request to `{base_url}/audio/speech` and returns the response bytes. The adapter reports `supports_tts_async = False` for these hosts because the endpoint is unverified, so check that your provider serves this route before relying on it.

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
- `list_models()` and `async_list_models()` raise `NotImplementedError`, so the caller-supplied key never reaches the listing client. Pass the model ID explicitly.
- Requests do not follow redirects, ignore proxy environment variables, use no automatic retries at the HTTP layer, and cap responses at 16 MiB. Raw provider error bodies are not surfaced.
- `stream_options` usage reporting is not requested, and Responses calls do not request encrypted reasoning.
- Responses and streaming deltas are checked for credential echoes before they reach callbacks or callers. An optional `private_limits` profile (`magic_llm.util.private_limits.PrivateRequestLimits`) binds the model, budgets and timeout per attempt; it cannot be used with a fallback.
- The value must be a real `bool`. Anything else raises `ValueError`.

The guarded transport covers chat only. Embeddings and audio calls use the ordinary HTTP client.

## Model discovery

`client.list_models()` and `await client.async_list_models()` pick the listing adapter from the `base_url` host, and the listing request always goes to that same origin with the client's key:

- A provider with a dedicated discovery adapter lists at the adapter's default URL when it shares the `base_url` origin. This covers providers whose listing path differs from their chat path, such as DeepInfra (`https://api.deepinfra.com/v1/models` for a `.../v1/openai` chat URL).
- A dedicated adapter reached through a different origin, such as a regional subdomain, lists at `{base_url}/models`.
- Hosts without a dedicated adapter, including Fireworks and custom endpoints, list at `{base_url}/models`.

See [model discovery](../usage/model-discovery.md) for the full list of adapters.

## Embeddings

`client.llm.embedding(...)` and `client.llm.async_embedding(...)` post to `{base_url}/embeddings` for every adapter, using the client's `model`. Support is confirmed for OpenAI, DeepInfra, Together, Mistral, and Fireworks; for other hosts it depends on the provider. See [embeddings](../usage/embeddings.md).
