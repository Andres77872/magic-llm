# Magic LLM

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Andres77872/magic-llm)

Magic LLM is a Python 3.10+ client library that exposes one `MagicLLM` interface across native and OpenAI-compatible LLM providers. Version `0.1.37` supports chat, streaming, async calls, embeddings, provider-specific audio, model discovery, tool calling, ReAct-style agents, and YAML-backed subagents.

> Magic LLM is a client library. It does **not** ship a CLI, REST server, or `.env` loader. Pass credentials explicitly to `MagicLLM(...)` from your own application configuration.

## Quick links

- [Documentation hub](docs/index.md)
- [Installation](docs/installation.md)
- [5-minute quickstart](docs/quickstart.md)
- [Provider guide](docs/providers/index.md)
- [Chat usage](docs/usage/chat.md)
- [Tool calling and agents](docs/agents/index.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development setup](docs/development/setup.md)
- [Testing](docs/development/testing.md)

## Features

- Unified `MagicLLM(engine=..., model=..., private_key=...)` constructor.
- Public chat methods: `generate`, `stream_generate`, `async_generate`, and `async_stream_generate`.
- Native engines: `openai`, `google`, `cloudflare`, `amazon`, `cohere`, `anthropic`, and `azure`.
- OpenAI-compatible routing by `engine='openai'` plus `base_url` for Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, Together, and similar endpoints.
- Streaming and async support across the core chat surface.
- Unified response models with usage and latency metadata where providers expose it.
- Embeddings, speech-to-text, text-to-speech, model discovery, fallback clients, callbacks, tool calling, ReAct agents, and subagents.
- Vision means image input for chat. First-class image generation/image output is not part of the current core API; user tools named `generate_image` are caller-owned tools.

## Install

Primary install path:

```bash
pip install git+https://github.com/Andres77872/magic-llm.git
```

If the package is published in your environment, this may also work:

```bash
pip install magic-llm
```

The project metadata does not declare a Python floor, but the source uses Python 3.10+ syntax. Use Python 3.10 or newer.

## 5-minute quickstart

```python
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
)

chat = ModelChat(system="You are a concise assistant.")
chat.add_user_message("Explain what a vector embedding is in one paragraph.")

response = client.llm.generate(chat)
print(response.content)
```

Streaming uses the same chat object:

```python
for chunk in client.llm.stream_generate(chat):
    text = chunk.choices[0].delta.content or ""
    print(text, end="", flush=True)
```

Async streaming:

```python
async for chunk in client.llm.async_stream_generate(chat):
    text = chunk.choices[0].delta.content or ""
    print(text, end="", flush=True)
```

## Provider overview

| Provider family | Engine | Notes |
| --- | --- | --- |
| OpenAI | `openai` | Official OpenAI endpoint by default. |
| OpenAI-compatible providers | `openai` + `base_url` | Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, Together, and others are selected by URL matching. |
| Anthropic | `anthropic` | Native Claude API support. |
| Google AI Studio | `google` | Native Gemini request/response formatting. |
| AWS Bedrock | `amazon` | Routes by Bedrock model prefix. Model discovery is unsupported. |
| Cloudflare Workers AI | `cloudflare` | Requires `account_id`. Tool calling and model discovery are unsupported. |
| Cohere | `cohere` | Native Cohere chat formatting. Tool calling is unsupported. |
| Azure | `azure` | Speech-only engine. Chat generation methods raise `NotImplementedError`. |

See [providers/index.md](docs/providers/index.md) and the per-provider pages for credentials and examples.

## OpenAI-compatible providers

Use `engine='openai'` and set `base_url`:

```python
client = MagicLLM(
    engine="openai",
    model="llama-3.1-8b-instant",
    private_key="gsk-your-key",
    base_url="https://api.groq.com/openai/v1",
)
```

Magic LLM detects known provider URLs and applies provider-specific adapters where implemented.

## Key gotchas

- **Official OpenAI token argument:** for `api.openai.com` only, `max_tokens` is transformed into `max_completion_tokens`. If both are supplied, `max_completion_tokens` wins and `max_tokens` is removed. Other OpenAI-compatible endpoints keep `max_tokens` unchanged.
- **Azure is speech-only:** `generate`, `stream_generate`, `async_generate`, and `async_stream_generate` raise `NotImplementedError` for `engine='azure'`.
- **Audio support is provider/method-specific:** unsupported TTS/STT paths fail fast instead of returning `None`; see `docs/usage/audio.md` for the sync/async matrix and STT upload metadata requirements.
- **No core image generation:** Magic LLM supports vision/image input for compatible chat models, not text-to-image output generation.
- **Model discovery gaps:** Amazon and Cloudflare do not support model discovery.
- **Tool-calling gaps:** Amazon Bedrock, Cohere, and Cloudflare do not support tool calling.
- **Subagents are disabled by default:** call `enable_subagents()` before `load_subagents()` or you will get an empty bundle.
- **No `.env` loading:** read secrets from your own secret manager or environment layer, then pass them as constructor arguments.

## Agents and tool calling

```python
from magic_llm import MagicLLM

client = MagicLLM(engine="openai", model="gpt-4o-mini", private_key="sk-your-key")

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

response = client.run_agent(
    user_input="Use the tool to add 17 and 25, then explain the result.",
    tools=[add],
    max_iterations=4,
)

print(response.content)
```

Tool specs may be Python callables, OpenAI-style JSON schemas with `tool_functions`, or Pydantic model classes. See [agents/tool-calling.md](docs/agents/tool-calling.md).

## Model discovery

```python
client = MagicLLM(engine="openai", private_key="sk-your-key")

for model in client.list_models():
    print(model.external_id, model.capabilities.chat)
```

Discovery is provider-dependent. Amazon and Cloudflare are explicitly unsupported. See [usage/model-discovery.md](docs/usage/model-discovery.md).

## Development

```bash
git clone https://github.com/Andres77872/magic-llm.git
cd magic-llm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Test command used by the project:

```bash
pytest test/ -v
```

Do not commit real keys. Integration tests use `MAGIC_LLM_KEYS`, `MAGIC_LLM_AUDIO_FILE`, and `MAGIC_LLM_IMAGE_B64_FILE`; debug payload logging uses `MAGIC_LLM_DEBUG_PAYLOAD` and `MAGIC_LLM_DEBUG_PAYLOAD_FULL`.

More details: [development/setup.md](docs/development/setup.md), [development/testing.md](docs/development/testing.md), and [development/contributing.md](docs/development/contributing.md).
