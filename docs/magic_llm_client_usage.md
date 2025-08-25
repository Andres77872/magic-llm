# MagicLLM Client Usage Guide

This guide shows how to use the MagicLLM client across providers for chat, streaming, tools, images, audio transcription, embeddings, multi‑turn conversations, fallbacks, and callbacks. It focuses strictly on usage.

---

## 1) Create a client

```python
from magic_llm import MagicLLM

# OpenAI-compatible endpoint example
client = MagicLLM(
    engine="openai",
    model="gpt-4o",
    private_key="sk-...",
)
```

You can pass other provider credentials/args analogously (see provider docs). The API surface on `client.llm` is unified across providers.

---

## 2) Compose a chat

Use `magic_llm.model.ModelChat` to build conversations.

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="You are a concise assistant.")
chat.add_user_message("What is the largest animal on earth?")
```

---

## 3) Generate responses

- `llm.generate(chat)` — sync, whole result
- `llm.stream_generate(chat)` — sync, streaming
- `await llm.async_generate(chat)` — async, whole result
- `async for ... in llm.async_stream_generate(chat)` — async, streaming

```python
# Sync, non-streaming
resp = client.llm.generate(chat)
print(resp.content)
print("Prompt tokens:", resp.usage.prompt_tokens)
print("Completion tokens:", resp.usage.completion_tokens)
```

```python
# Sync, streaming
content = ""
for chunk in client.llm.stream_generate(chat):
    content += chunk.choices[0].delta.content or ""
print(content)
```

```python
# Async, non-streaming
import asyncio

async def main():
    resp = await client.llm.async_generate(chat)
    print(resp.content)

asyncio.run(main())
```

```python
# Async, streaming
import asyncio

async def main():
    content = ""
    async for chunk in client.llm.async_stream_generate(chat):
        content += chunk.choices[0].delta.content or ""
    print(content)

asyncio.run(main())
```

### Inspect responses and tool calls

Synchronous response object (`ModelChatResponse`):

```python
resp = client.llm.generate(chat)
print(resp.content)

# Tool calls (if any)
if resp.tool_calls:
    for tc in resp.tool_calls:
        print("tool:", tc.function.name, "args:", tc.function.arguments)

print("finish_reason:", resp.finish_reason)
print("usage:", resp.usage.dict() if resp.usage else None)
```

Streaming chunks (`ChatCompletionModel`):

```python
for chunk in client.llm.stream_generate(chat):
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="")
    if delta.tool_calls:
        for tc in delta.tool_calls:
            print(f"\n[tool call] {tc.function.name} args={tc.function.arguments}")
```

Note: usage metrics during streaming are available on chunks (when provided) and via the callback after completion.

---

## 4) Callbacks and usage metrics

Attach a callback to receive the final output and usage (also after fallback if used). The callback signature is `(msg: ModelChat, content: str, usage, model_name: str, meta)`.

```python
def on_finish(msg, content, usage, model_name, meta):
    print(f"Model {model_name} | PT={usage.prompt_tokens} CT={usage.completion_tokens}")

client = MagicLLM(
    engine="openai",
    model="gpt-4o",
    private_key="sk-...",
    callback=on_finish,
)

# Trigger any generation (streaming shown)
for _ in client.llm.stream_generate(chat):
    pass  # content handled by callback
```

---

## 5) Function calling (tools)

MagicLLM supports multiple tool styles with the same `tools` and `tool_choice` parameters on `llm.generate(...)` (and streaming variants).

Tool choice options:

- auto — let the model decide whether to call a tool
- none — disable tool calling
- required — force the model to call a tool
- {"type":"function","function":{"name":"<tool_name>"}} — force a specific tool

### A. Python callables (optionally Pydantic models)

- Provide callables directly in `tools`.
- Optionally include Pydantic models (if installed) to describe tool schemas.
- Choose a specific tool with `tool_choice={"type":"function","function":{"name":"..."}}` or let the model choose via `tool_choice="auto"`.

```python
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

# Define Python tools
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return ""

try:
    from pydantic import BaseModel  # Pydantic v2
    class GetForecast(BaseModel):
        """Forecast for a given location and days."""
        location: str
        days: int
    tools = [get_weather, GetForecast]
except Exception:
    tools = [get_weather]

chat = ModelChat()
chat.add_user_message("Please check the weather for Bogotá and maybe use tools if needed.")

# Pass tools at init time, and direct tool_choice to a specific function name
client = MagicLLM(
    engine="openai",
    model="gpt-4o",
    private_key="sk-...",
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)
res = client.llm.generate(chat)
print(res)

# Or provide defaults at init and override at call time
client = MagicLLM(engine="openai", model="gpt-4o", private_key="sk-...", tools=tools[:1], tool_choice="auto")
res = client.llm.generate(chat, tools=tools, tool_choice={"type":"function","function":{"name":"get_weather"}})
print(res)
```

```python
# Streaming with tools
content = ""
for chunk in client.llm.stream_generate(chat, tools=tools, tool_choice="auto"):
    delta = chunk.choices[0].delta
    content += delta.content or ""
print(content)
```

### B. OpenAI-style JSON function specs

Provide function schemas as JSON entries in `tools`.

```python
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

FUNCTION_DEF = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": True
    }
}

chat = ModelChat()
chat.add_user_message("What is the weather like in Paris today?")

client = MagicLLM(engine="openai", model="gpt-4o", private_key="sk-...", tools=[FUNCTION_DEF],
                  tool_choice={"type":"function","function":{"name":"get_weather"}})
res = client.llm.generate(chat)
print(res)
```

---

## 6) Agentic tool workflow (multi-turn tool loops)

Use `magic_llm.util.agentic.run_agentic()` to let the model call tools iteratively until it produces normal content. See `magic_llm/util/agentic.py` for behavior.

```python
from typing import Any, List
from magic_llm import MagicLLM
from magic_llm.util.agentic import run_agentic

client = MagicLLM(engine="openai", model="gpt-4o", private_key="sk-...")

def add(a: int, b: int) -> int:
    return a + b

def top_k(items: List[Any], k: int = 3) -> List[Any]:
    return list(items)[:k]

resp = run_agentic(
    client=client,
    user_input="Compute 17 + 25, then take the first 2 fruits from ['apple','banana','cherry'] and summarize.",
    system_prompt="Use tools for arithmetic and list selection before answering.",
    tools=[add, top_k],
    tool_choice="auto",
    max_iterations=4,
)
print(resp.content)
```

OpenAI-style tool specs with a Python function map:

```python
from magic_llm.util.agentic import run_agentic

specs = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}]

def add(a: int, b: int) -> int:
    return a + b

resp = run_agentic(
    client=client,
    user_input="Use the add tool to add 7 and 35, then explain the result.",
    tools=specs,
    tool_functions={"add": add},
)
print(resp.content)
```

---

## 7) Multi‑turn conversations (manual)

Append assistant replies and continue.

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="You are helpful.")
chat.add_user_message("Hello! Who are you?")
resp = client.llm.generate(chat)

# Option 1: add explicitly
chat.add_assistant_message(resp.content)
chat.add_user_message("Thanks. Summarize your capabilities in 1 sentence.")
print(client.llm.generate(chat).content)

# Option 2: use ModelChat.__add__ to append
chat = chat + resp  # adds the assistant message from the response
chat.add_user_message("Great. One more example?")
print(client.llm.generate(chat).content)
```

---

## 8) Images in chat messages (vision)

Use `ModelChat.add_user_message(content, image=..., media_type=...)`.

- Image can be a URL string, a base64 string, bytes, or a list mixing those.
- `media_type` defaults to `image/jpeg` and is used when embedding base64/bytes as data URLs.
- Content must not be empty when passing an image (otherwise an exception is raised).

```python
from magic_llm.model import ModelChat

SAMPLE_URL = "https://example.com/sample.webp"
SAMPLE_B64 = "...base64..."  # your base64 string
SAMPLE_BYTES = b"..."         # your image bytes

# URL
chat = ModelChat()
chat.add_user_message("What do you see?", image=SAMPLE_URL, media_type="image/webp")
print(client.llm.generate(chat).content)

# Base64
chat = ModelChat()
chat.add_user_message("Describe this.", image=SAMPLE_B64, media_type="image/webp")
print(client.llm.generate(chat).content)

# Bytes
chat = ModelChat()
chat.add_user_message("Describe this.", image=SAMPLE_BYTES, media_type="image/png")
print(client.llm.generate(chat).content)

# Multiple images (mix URL + bytes)
chat = ModelChat()
chat.add_user_message("Describe both images.", image=[SAMPLE_URL, SAMPLE_BYTES])
print(client.llm.generate(chat).content)
```

Error case (image alone is not allowed):

```python
from magic_llm.model import ModelChat
chat = ModelChat()
# Raises Exception("Image cannot be alone")
chat.add_user_message("", image=SAMPLE_URL)
```

---

## 9) Audio transcription (speech‑to‑text)

Use `AudioTranscriptionsRequest` with `llm.sync_audio_transcriptions(...)` or `await llm.async_audio_transcriptions(...)`.

```python
from magic_llm import MagicLLM
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest

client = MagicLLM(engine="openai", private_key="sk-...")

with open("speech.mp3", "rb") as f:
    data = AudioTranscriptionsRequest(
        file=f.read(),
        model="whisper-1",  # provider/model-specific
    )

# Sync
resp = client.llm.sync_audio_transcriptions(data)
print(resp["text"])  # provider-compatible response

# Async
# resp = await client.llm.async_audio_transcriptions(data)
```

---

## 10) Embeddings

Use `llm.embedding(text=...)` or `await llm.async_embedding(text=...)`. Returns a provider‑compatible response (shape may vary by provider/model).

```python
# Sync
vec = client.llm.embedding(text="How much wood would a woodchuck chuck?")
print(vec)

# Async
# vec = await client.llm.async_embedding(text="...")
```

---

## 11) Fallback clients (multi‑client orchestration)

If a request fails (e.g., invalid model), configure a fallback client that will be used automatically.

```python
fallback_client = MagicLLM(engine="openai", model="gpt-4o", private_key="sk-...")
client = MagicLLM(
    engine="openai",
    model="bad-model",
    private_key="sk-...",
    fallback=fallback_client,
)

resp = client.llm.generate(chat)  # transparently uses fallback on failure
print(resp.content)
```

---

## 12) Error handling

All errors during generation raise `magic_llm.exception.ChatException.ChatException`.

```python
from magic_llm.exception.ChatException import ChatException

try:
    resp = client.llm.generate(chat)
    print(resp.content)
except ChatException as e:
    print("Generation failed:", e)
```

For streaming, you can handle exceptions around the iterator/async iterator:

```python
from magic_llm.exception.ChatException import ChatException

try:
    for chunk in client.llm.stream_generate(chat):
        print(chunk.choices[0].delta.content or "", end="")
except ChatException as e:
    print("Streaming failed:", e)
```

---

## 13) Parameter recap (most used)

- `MagicLLM(engine, model, private_key, callback=None, fallback=None, tools=None, tool_choice=None, ...)`
- `llm.generate(chat, tools=None, tool_choice=None, ...)`
- `llm.stream_generate(chat, tools=None, tool_choice=None, ...)`
- `await llm.async_generate(chat, tools=None, tool_choice=None, ...)`
- `async for chunk in llm.async_stream_generate(chat, tools=None, tool_choice=None, ...)`
- `llm.embedding(text, ...)`, `await llm.async_embedding(text, ...)`
- `llm.sync_audio_transcriptions(AudioTranscriptionsRequest, ...)`, `await llm.async_audio_transcriptions(...)`
- Images via `ModelChat.add_user_message(content, image=..., media_type=...)`

This completes the end‑to‑end usage of the MagicLLM client.
