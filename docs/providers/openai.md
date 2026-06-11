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

Known URL patterns route to provider adapters for Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, and Together.

## Critical `max_tokens` behavior

For the official `api.openai.com` endpoint only, Magic LLM transforms request payloads:

- `max_tokens` becomes `max_completion_tokens`.
- If both `max_tokens` and `max_completion_tokens` are present, `max_completion_tokens` wins and `max_tokens` is removed.
- Other OpenAI-compatible providers keep `max_tokens` unchanged.

```python
response = client.llm.generate(chat, max_tokens=512)
```

On official OpenAI this is sent as `max_completion_tokens=512`; on Groq, Mistral, Together, and similar endpoints it remains `max_tokens=512`.

## Capabilities

- Chat: supported.
- Streaming: supported.
- Async chat and streaming: supported.
- Tool calling: supported for most adapters, but provider behavior depends on the endpoint.
- Embeddings: supported by OpenAI and selected compatible providers such as DeepInfra, Together, Mistral, and Fireworks.
- Audio: official OpenAI supports async TTS and sync+async STT; Together supports sync+async TTS; DeepInfra keeps async TTS only; Fireworks supports async STT only for `whisper-v3` and `whisper-v3-turbo`; other compatible adapters fail fast for media unless explicitly verified.
- Model discovery: supported by many adapters; availability depends on the provider.

Unknown OpenAI-compatible endpoints no longer inherit official OpenAI media or vision claims. Provide accurate STT `filename` and `content_type`; multipart uploads use a real `file` part plus form fields.
