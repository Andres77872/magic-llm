# Quickstart

This page shows the smallest useful path: install, create a client, build a chat, generate a response, stream output, and switch providers.

## 1. Install

```bash
pip install git+https://github.com/Andres77872/magic-llm.git
```

## 2. Create a client

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
)
```

## 3. Build a conversation

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="You are precise and helpful.")
chat.add_user_message("Give me three practical uses for embeddings.")
```

## 4. Generate a full response

```python
response = client.llm.generate(chat)
print(response.content)

if response.usage:
    print(response.usage.prompt_tokens, response.usage.completion_tokens)
```

## 5. Stream a response

```python
for chunk in client.llm.stream_generate(chat):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

## 6. Use async APIs

```python
import asyncio

async def main():
    response = await client.llm.async_generate(chat)
    print(response.content)

    async for chunk in client.llm.async_stream_generate(chat):
        print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
```

## 7. Switch providers

Native engines use different `engine` values:

```python
anthropic = MagicLLM(
    engine="anthropic",
    model="claude-3-haiku-20240307",
    private_key="sk-ant-your-key",
)

google = MagicLLM(
    engine="google",
    model="gemini-1.5-flash",
    private_key="google-api-key",
)
```

OpenAI-compatible providers keep `engine="openai"` and set `base_url`:

```python
groq = MagicLLM(
    engine="openai",
    model="llama-3.1-8b-instant",
    private_key="gsk-your-key",
    base_url="https://api.groq.com/openai/v1",
)
```

Next: read [chat usage](usage/chat.md) and [provider configuration](providers/index.md).
