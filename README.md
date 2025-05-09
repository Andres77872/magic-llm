[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Andres77872/magic-llm)

# Magic LLM

Magic LLM is a unified, simplified wrapper for connecting to a wide range of LLM providers, including:

- [OpenAI](https://platform.openai.com/docs/api-reference)
- [Cloudflare](https://developers.cloudflare.com/workers-ai/models/text-generation/#responses)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Google AI Studio](https://ai.google.dev/tutorials/rest_quickstart)
- [Cohere](https://docs.cohere.com/reference/chat)
- [Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [LeptonAI](hhttps://www.lepton.ai/docs/guides)
- [Cerebras](https://inference-docs.cerebras.ai/introduction)
- [SambaNova](https://docs.sambanova.ai/cloud/docs/get-started/overview)
- [DeepInfra](https://deepinfra.com/docs)
- [Deepseek](https://api-docs.deepseek.com/)
- [Parasail](https://docs.parasail.io/parasail-docs)
- [x.ai (Grok)](https://grok.x.ai/)
- [Together.AI](https://docs.together.ai/docs/openai-api-compatibility)
- [OpenRouter](https://openrouter.ai/docs#requests)
- [NovitaAI](https://novita.ai/docs/guides/introduction)
- [Mistral](https://docs.mistral.ai/)
- [Hyperbolic](https://docs.hyperbolic.xyz/docs/getting-started)
- [Groq](https://console.groq.com/docs/openai)
- [Fireworks.AI](https://docs.fireworks.ai/getting-started/introduction)
- [Perplexity AI](https://docs.perplexity.ai/home)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
  
*Note: Many of these providers have been verified compatible with OpenAI's API or are supported natively. Some may require non-standard argument/credential patterns.*

---

## Supported Providers & Capabilities

| Provider         | Streaming | Completion | Embedding | Audio | Async Streaming | Async Completion | Fallback | Callback |
|------------------|-----------|------------|-----------|-------|-----------------|------------------|----------|----------|
| OpenAI           | ✅         | ✅          | ✅         | ✅    | ✅               | ✅                | ✅       | ✅       |
| Cloudflare       | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| AWS Bedrock      | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Google AI Studio | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Cohere           | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Anthropic        | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| LeptonAI         | ✅         | ✅          | ❌         | ❌    | ❓               | ✅                | ✅       | ✅       |
| Cerebras         | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| SambaNova        | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| DeepInfra        | ✅         | ✅          | ✅         | ✅    | ✅               | ✅                | ✅       | ✅       |
| Deepseek         | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Parasail         | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| x.ai             | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Together.AI      | ✅         | ✅          | ✅         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Perplexity AI    | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| OpenRouter       | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| NovitaAI         | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Mistral          | ✅         | ✅          | ✅         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Hyperbolic       | ✅         | ✅          | ❌         | ❌    | ✅               | ✅                | ✅       | ✅       |
| Groq             | ✅         | ✅          | ❌         | ✅    | ✅               | ✅                | ✅       | ✅       |
| Fireworks.AI     | ✅         | ✅          | ✅         | ✅    | ✅               | ✅                | ✅       | ✅       |
| Azure OpenAI     | ✅         | ✅          | ✅         | ✅    | ✅               | ✅                | ✅       | ✅       |

*Legend:*
- **Streaming:** Supports incremental streamed responses
- **Completion:** Supports single result completion
- **Embedding:** Supports text embedding generation
- **Audio:** Supports transcription (speech-to-text)
- **Async:** Async streaming/async completion methods available
- **Fallback:** Can automatically fallback to alternate client
- **Callback:** Callback support on streamed outputs
- ❓ = Not fully tested

---

## Features

- [x] Streamed chat completion (sync & async)
- [x] Non-stream (single result) completion (sync & async)
- [x] Usage metrics for every call (tokens, latency, etc.)
- [x] Callback hook on streamed outputs (for logging, progress bar, etc.)
- [x] Fallback support: transparently try alternate client on error
- [x] Unified error handling (raises `ChatException`)
- [x] Embedding
- [x] Audio transcription (speech-to-text, for compatible models/providers)
- [x] Return types compatible with OpenAI client (for easy integration)
- [x] Vision models (OpenAI/Anthropic; partial Google support)
- [x] Function calling (tested with OpenAI)
- [ ] Text-to-Speech (currently: OpenAI only)
- [ ] Vision adapter for Google AI Studio
- [ ] More TTS/vision support in other providers

---

## Purpose

Magic LLM is designed as the backend core for [Magic UI](https://talk.novus.chat/), an application generator (RAG) and multivendor LLM front-end. It is **not** a full OpenAI client replacement, but strives for wide usability and API/response shape compatibility.

---

# Quickstart & Usage

## Install

```bash
pip install git+https://github.com/Andres77872/magic-llm.git
```

## Basic Usage Pattern

### 1. Build a client for any provider

```python
from magic_llm import MagicLLM

# Example for an OpenAI API compatible endpoint
client = MagicLLM(
    engine='openai',
    model='gpt-4o',
    private_key='sk-...',
)
```

*Other providers (e.g. `engine='anthropic'`, `engine='google'`, ... ) use analogous fields, see below.*

### 2. Compose a conversation

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="You are a helpful assistant.")
chat.add_user_message("What is the largest animal on earth?")
```

### 3. Request an answer (Non-streaming: returns full response)

```python
response = client.llm.generate(chat)
print(response.content)
print("Prompt tokens:", response.usage.prompt_tokens)
print("Completion tokens:", response.usage.completion_tokens)
```

### 4. Streamed response (token by token, OpenAI style, sync)

```python
for chunk in client.llm.stream_generate(chat):
    print(chunk.choices[0].delta.content or '', end='', flush=True)
print()  # Newline at end
```

### 5. Async usage (across all providers!)

```python
import asyncio

async def main():
    async for chunk in client.llm.async_stream_generate(chat):
        print(chunk.choices[0].delta.content or '', end='', flush=True)

asyncio.run(main())
```

---

## Error Handling & Fallback

If you provide an invalid model or the provider fails, MagicLLM can fallback to a second client automatically.

```python
# Setup a fallback chain (model 'bad-model' fails, fallback is 'gpt-4o')
client_fallback = MagicLLM(engine='openai', model='gpt-4o', private_key='sk-...')
client = MagicLLM(
    engine='openai',
    model='bad-model',
    private_key='sk-...',
    fallback=client_fallback,
)

response = client.llm.generate(chat)
print(response.content)  # Uses fallback auto-magically if first fails!
```

## Advanced: Callbacks (monitoring, live UI, logging, etc.)

Attach a callback for every final output (after fallback, if needed):

```python
def on_chunk(msg, content, usage, model_name, meta):
    print(f"Used model {model_name}: [{usage.prompt_tokens}pt >> {usage.completion_tokens}ct] {content}")

client = MagicLLM(
    engine='openai',
    model='bad-model',
    private_key='sk-...',
    fallback=client_fallback,
    callback=on_chunk,
)

for chunk in client.llm.stream_generate(chat):
    pass  # Content handled by callback!
```

---

## Audio Transcription (Speech-to-Text)

If your provider or endpoint supports OpenAI Whisper API (e.g. OpenAI, Azure, DeepInfra, Groq, Fireworks), you can transcribe audio files with a unified API:

```python
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest
from magic_llm import MagicLLM

client = MagicLLM(engine='openai', private_key='sk-...')
with open('speech.mp3', 'rb') as f:
    data = AudioTranscriptionsRequest(file=f.read(), model="whisper-1")
    response = client.llm.sync_audio_transcriptions(data)
    print(response['text'])

# Async version:
# await client.llm.async_audio_transcriptions(data)
```

---

## Embeddings

Some providers support embeddings using the unified API. Example (Together.AI shown):

```python
client = MagicLLM(engine='together.ai', private_key='sk-...', model='BAAI/bge-base-en-v1.5')
resp = client.llm.embedding(text="How much wood would a woodchuck chuck?")
print(resp)  # List[float]
```

---

## Supported Provider Configurations

**OpenAI (and any OpenAI-compatible API endpoint):**

```python
client = MagicLLM(
    engine='openai',
    model='gpt-4o',
    private_key='sk-...',
)
```

**Cloudflare:**
```python
client = MagicLLM(
    engine='cloudflare',
    model='@cf/meta/llama-2-7b-chat-int8',
    private_key='api-key',
    account_id='cf-account',
)
```

**AWS Bedrock:**
```python
client = MagicLLM(
    engine='amazon',
    model='amazon.nova-pro-v1:0',
    aws_access_key_id='AKIA....',
    aws_secret_access_key='...',
    region_name='us-east-1',
)
```

**Google AI Studio:**
```python
client = MagicLLM(
    engine='google',
    model='gemini-1.5-flash',
    private_key='GOOG...',
)
```

**Cohere:**
```python
client = MagicLLM(
    engine='cohere',
    model='command-light',
    private_key='cohere-...',
)
```

**Anthropic:**
```python
client = MagicLLM(
    engine='anthropic',
    model='claude-3-haiku-20240307',
    private_key='...',
)
```

**Other providers**: use the same builder pattern, see provider docs or examples. Some models/engines require special names.

---

# API Summary

All LLM providers share the same interface:

- `llm.generate(chat)`<br> — Synchronous, whole-result
- `llm.stream_generate(chat)`<br> — Synchronous streaming (token by token)
- `await llm.async_generate(chat)`<br> — Async, whole-result
- `async for chunk in llm.async_stream_generate(chat)`<br> — Async, token stream
- `llm.embedding(text=...)`<br> — Get embeddings (supported providers only)
- `llm.audio_transcriptions(data)`<br> — Speech-to-text (supported providers/models only)
- All return OpenAI-compatible objects, wherever possible

---

# Design Principles

- **Frictionless provider swapping:** One codebase, many clouds.
- **Error handling**: Consistent exceptions (`ChatException`) for failed requests.
- **Streaming-first:** Low-latency, responsive outputs.
- **Callbacks & metrics:** Plug in your logger/progressbar/UI easily.
- **Fallback:** Failover to backup provider/credentials automatic.
- **Minimum dependency:** No hard OpenAI client requirement.
