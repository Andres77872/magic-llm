# Anthropic

Use `engine="anthropic"` for Claude models.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="anthropic",
    model="claude-3-haiku-20240307",
    private_key="sk-ant-your-key",
)
```

## Notes

- Chat, streaming, async chat, and async streaming are supported through the native Messages API (`https://api.anthropic.com/v1/messages`).
- Tool calling is supported through Magic LLM's normalized tooling layer.
- Vision input is supported through `ModelChat.add_user_message(..., image=...)` for vision-capable Claude models.
- Model discovery is supported through `client.list_models()` and `client.async_list_models()`.
- When `max_tokens` is not supplied, the Anthropic engine sets a default of `4096`.
- Audio is unsupported, and TTS/STT methods raise `UNSUPPORTED_MEDIA_OPERATION`.
- Embeddings are not implemented: `embedding()` and `async_embedding()` return `None`.

## Example

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="Answer in plain English.")
chat.add_user_message("What is a context window?")

response = client.llm.generate(chat)
print(response.content)
```
