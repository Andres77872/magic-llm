# Cohere

Use `engine="cohere"` for Cohere chat models.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="cohere",
    model="command-light",
    private_key="cohere-api-key",
)
```

## Notes

- Chat, streaming, async chat, and async streaming are supported through the v1 chat API (`https://api.cohere.ai/v1/chat`).
- The engine maps messages into Cohere's role format.
- Tool calling is unsupported. Passing `tools` raises `TOOL_WIRING_NOT_IMPLEMENTED`.
- Vision input is unsupported. Images are rejected before any request is sent; see [unsupported-feature errors](index.md#unsupported-feature-errors).
- Model discovery is supported through `client.list_models()` and `client.async_list_models()`.
- Audio is unsupported, and TTS/STT methods raise `UNSUPPORTED_MEDIA_OPERATION`.
- Embeddings are not implemented: `embedding()` and `async_embedding()` return `None`.

## Example

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="Answer as a senior engineer.")
chat.add_user_message("What should I log around LLM retries?")

response = client.llm.generate(chat)
print(response.content)
```
