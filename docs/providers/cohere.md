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

- Chat, streaming, async chat, and async streaming are supported.
- The engine maps messages into Cohere's role format.
- Tool calling is unsupported.
- Model discovery is supported.
- Embeddings and audio are not documented as supported by this engine.

## Example

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="Answer as a senior engineer.")
chat.add_user_message("What should I log around LLM retries?")

response = client.llm.generate(chat)
print(response.content)
```
