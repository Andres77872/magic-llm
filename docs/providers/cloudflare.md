# Cloudflare Workers AI

Use `engine="cloudflare"` for Cloudflare Workers AI.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="cloudflare",
    model="@cf/meta/llama-2-7b-chat-int8",
    private_key="cloudflare-api-token",
    account_id="cloudflare-account-id",
)
```

## Notes

- `account_id` is required.
- Chat, streaming, async chat, and async streaming are supported.
- Tool calling is unsupported.
- Model discovery is unsupported and raises `NotImplementedError`.
- Embeddings and audio are not documented as supported by this engine.

## Example

```python
from magic_llm.model import ModelChat

chat = ModelChat()
chat.add_user_message("Write a two-line project status update.")

response = client.llm.generate(chat)
print(response.content)
```
