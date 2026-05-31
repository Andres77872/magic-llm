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

- Chat, streaming, async chat, and async streaming are supported.
- Tool calling is supported through Magic LLM's normalized tooling layer.
- Model discovery is supported.
- When `max_tokens` is not supplied, the Anthropic engine sets a default of `4096`.

## Example

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="Answer in plain English.")
chat.add_user_message("What is a context window?")

response = client.llm.generate(chat)
print(response.content)
```
