# Chat

The chat API is centered on `ModelChat` and `client.llm`.

## Build a chat

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="You are helpful and concise.")
chat.add_user_message("What is retrieval augmented generation?")
chat.add_assistant_message("RAG combines search with generation.")
chat.add_user_message("Give me one concrete use case.")
```

## Generate

```python
response = client.llm.generate(chat)
print(response.content)
print(response.finish_reason)
```

`ModelChatResponse` exposes convenience properties such as `content`, `role`, `tool_calls`, and `finish_reason`.

## Stream

```python
for chunk in client.llm.stream_generate(chat):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Stream chunks use `ChatCompletionModel`, with OpenAI-style `choices` and optional usage/meta data.

## Async generate

```python
response = await client.llm.async_generate(chat)
print(response.content)
```

## Async stream

```python
async for chunk in client.llm.async_stream_generate(chat):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Provider kwargs

Extra keyword arguments pass through to the provider transformation layer:

```python
response = client.llm.generate(
    chat,
    temperature=0.2,
    max_tokens=512,
    json_output=True,
)
```

Important: on official OpenAI only, `max_tokens` is converted to `max_completion_tokens`. See [OpenAI provider notes](../providers/openai.md).

## Fallback clients

```python
backup = MagicLLM(engine="openai", model="gpt-4o-mini", private_key="sk-backup")
client = MagicLLM(
    engine="openai",
    model="primary-model",
    private_key="sk-primary",
    fallback=backup,
)
```

If the primary request fails, Magic LLM can delegate to the fallback client.
