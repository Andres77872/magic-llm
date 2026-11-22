# Magic LLM

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Andres77872/magic-llm)

Magic LLM is a Python 3.10+ client library that exposes one `MagicLLM` interface across native and OpenAI-compatible LLM providers. It supports chat, streaming, async calls, embeddings, provider-specific audio, model discovery, tool calling, ReAct-style agents, and YAML-backed subagents.

> Magic LLM is a client library. It does **not** ship a CLI, REST server, or `.env` loader. Pass credentials explicitly to `MagicLLM(...)` from your own application configuration.

## Quick links

- [Documentation hub](docs/index.md)
- [Installation](docs/installation.md)
- [5-minute quickstart](docs/quickstart.md)
- [Provider guide](docs/providers/index.md)
- [Chat usage](docs/usage/chat.md)
- [Vision](docs/usage/vision.md)
- [Audio](docs/usage/audio.md)
- [Embeddings](docs/usage/embeddings.md)
- [Model discovery](docs/usage/model-discovery.md)
- [Error handling](docs/usage/error-handling.md)
- [Tool calling and agents](docs/agents/index.md)
- [Agent budgets and hooks](docs/agents/budget-hooks.md)
- [Reliable application integration](docs/agents/production-integration.md)
- [Request validation and complete tool results](docs/agents/request-integrity.md)
- [Prompt fragments](docs/agents/prompt-fragment.md)
- [Subagents](docs/agents/subagents.md)
- [Troubleshooting](docs/troubleshooting.md)
- [FAQ](docs/faq.md)
- [Development setup](docs/development/setup.md)
- [Testing](docs/development/testing.md)

## Features

- Unified `MagicLLM(engine=..., model=..., private_key=...)` constructor.
- Public chat methods: `generate`, `stream_generate`, `async_generate`, and `async_stream_generate`.
- Native engines: `openai`, `google`, `cloudflare`, `amazon`, `cohere`, `anthropic`, and `azure`.
- OpenAI-compatible routing by `engine='openai'` plus `base_url`: dedicated chat adapters are URL-matched for Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, and Together; any other OpenAI-compatible endpoint works through the generic path.
- OpenAI Responses API through `engine='openai', endpoint='responses'`, behind the same chat, streaming, and agent methods. The default is `endpoint='chat_completions'`.
- Streaming and async support across the core chat surface.
- Unified response models with usage and latency metadata where providers expose it.
- Embeddings, speech-to-text, text-to-speech, model discovery, fallback clients, callbacks, tool calling, ReAct agents (`run_agent`, `run_agent_stream`, `run_agent_async`, `run_agent_stream_async`), and YAML-backed subagents.
- Vision means image input for chat. First-class image generation/image output is not part of the current core API; user tools named `generate_image` are caller-owned tools.

## Install

Install from the official Git repository:

```bash
python -m pip install 'magic_llm @ git+https://github.com/Andres77872/magic-llm.git@master'
```

For an editable install, run `python -m pip install -e .` from a local `magic-llm` checkout. No `magic-agents` install is needed. Use `.[notebook]` to add the Jupyter dependency, or install `-r requirements.txt` for an editable checkout with test tools.

Don't install by bare name (`magic-llm` or `magic_llm`) from public PyPI; it is a different project.

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
| OpenAI-compatible providers | `openai` + `base_url` | Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, and Together are selected by URL matching; other endpoints use the generic path. |
| Anthropic | `anthropic` | Native Claude API support. |
| Google AI Studio | `google` | Native Gemini request/response formatting. |
| AWS Bedrock | `amazon` | Routes by Bedrock model prefix. Tool calling and model discovery are unsupported. |
| Cloudflare Workers AI | `cloudflare` | Requires `account_id`. Tool calling and model discovery are unsupported. |
| Cohere | `cohere` | Native Cohere chat formatting. Tool calling is unsupported. |
| Azure | `azure` | Speech-only engine (async TTS/STT). Chat generation is not implemented. |

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

Magic LLM detects known provider URLs and applies provider-specific chat adapters where implemented; unrecognized `base_url` values fall back to the generic OpenAI-compatible path.

Model discovery covers a wider set than chat routing: registered discovery adapters also include xAI, Cerebras, Hyperbolic, Nebius, Novita, Parasail, and Perplexity. `client.list_models()` picks the adapter from the `base_url` host; see [usage/model-discovery.md](docs/usage/model-discovery.md).

For endpoints and keys supplied by your users at runtime, `private_transport=True` restricts chat to async calls over public HTTPS, with no redirects or proxies. See [providers/openai.md](docs/providers/openai.md#private-transport-for-caller-supplied-endpoints).

### OpenAI Responses API

Pass `endpoint="responses"` to send requests to `/responses` instead of `/chat/completions`:

```python
client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
    endpoint="responses",
)
```

`ModelChat`, `generate`/`stream_generate` (sync and async), and `run_agent*` work the same way. Conversation state stays in `ModelChat`, and Responses output items (including encrypted reasoning) are replayed with tool results, so nothing is stored on the provider side between agent iterations.

## Key gotchas

- **Official OpenAI token argument:** on the Chat Completions path for `api.openai.com` only, `max_tokens` is transformed into `max_completion_tokens`. If both are supplied, `max_completion_tokens` wins and `max_tokens` is removed. Other OpenAI-compatible endpoints keep `max_tokens` unchanged. With `endpoint="responses"`, the value is sent as `max_output_tokens`.
- **Azure is speech-only:** `generate`, `stream_generate`, `async_generate`, and `async_stream_generate` fail for `engine='azure'` without retrying. Without a fallback they raise `ChatException` (`STREAM_GENERATION_ERROR`).
- **Retries depend on the error:** transient failures are retried up to `retries` times (default 3). Unsupported images and Azure chat fail after one attempt but still reach a `fallback`, and HTTP 4xx rejections (except 408 and 429) raise immediately. See [retries](docs/usage/error-handling.md#retries).
- **Embeddings are OpenAI-family only:** `embedding()` on any other engine returns `None` instead of raising.
- **Audio support is provider/method-specific:** unsupported TTS/STT paths fail fast instead of returning `None`; see `docs/usage/audio.md` for the sync/async matrix and STT upload metadata requirements.
- **No core image generation:** Magic LLM supports vision/image input for compatible chat models, not text-to-image output generation.
- **Model discovery gaps:** Amazon and Cloudflare do not support model discovery. See [model discovery](docs/usage/model-discovery.md) for provider-specific listing endpoints.
- **Tool-calling gaps:** Amazon Bedrock, Cohere, and Cloudflare do not support tool calling.
- **Subagents are disabled by default:** call `magic_llm.agent.enable_subagents()` before `await client.load_subagents(...)` (it is async), or you will get an empty bundle.
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

`run_agent` is the synchronous entry point. The same loop is also available as `run_agent_stream` (sync streaming), `run_agent_async` (async), and `run_agent_stream_async` (async streaming). See [agents/index.md](docs/agents/index.md).

Tool specs may be Python callables, OpenAI-style JSON schemas with `tool_functions`, or Pydantic model classes. See [agents/tool-calling.md](docs/agents/tool-calling.md).

`max_iterations` defaults to 150. To bound tokens and wall-clock time too, pass `budget=AgentBudget(...)` from `magic_llm.agent.types`. A budget that goes over its limit raises `AgentBudgetExceeded`. If you set `AgentBudget(final_answer_notice="...")`, the call at the iteration limit runs without tools and the loop returns the model's text answer instead of raising. Lifecycle hooks, `request_guard`, and provider attempt control are covered in [agents/budget-hooks.md](docs/agents/budget-hooks.md), [agents/request-integrity.md](docs/agents/request-integrity.md), and [agents/production-integration.md](docs/agents/production-integration.md).

## Model discovery

```python
client = MagicLLM(engine="openai", private_key="sk-your-key")

for model in client.list_models():
    print(model.external_id, model.capabilities.chat)
```

An async variant, `client.async_list_models()`, is also available.

`client.list_models()` supports OpenAI, OpenAI-compatible clients whose provider offers a listing endpoint, Google, Anthropic, and Cohere. Clients with `private_transport=True` do not list models. For Azure OpenAI and Foundry, use their discovery adapters directly; see [model discovery](docs/usage/model-discovery.md).

## Development

```bash
git clone https://github.com/Andres77872/magic-llm.git
cd magic-llm
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Run the same offline coverage gate as CI (all tests live in `tests/`):

```bash
python -m pytest --cov=magic_llm --cov-branch
```

The configured default excludes live/billable provider tests. Run those only with an explicit provider marker and `MAGIC_LLM_KEYS=/path/to/private-keys.json`. Do not commit real keys. Optional live media tests use `MAGIC_LLM_AUDIO_FILE` and `MAGIC_LLM_IMAGE_B64_FILE`; debug payload logging uses `MAGIC_LLM_DEBUG_PAYLOAD` and `MAGIC_LLM_DEBUG_PAYLOAD_FULL`.

More details: [development/setup.md](docs/development/setup.md), [development/testing.md](docs/development/testing.md), and [development/contributing.md](docs/development/contributing.md).
