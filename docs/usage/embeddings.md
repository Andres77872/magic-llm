# Embeddings

Use `client.llm.embedding(text=...)` for supported providers.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="openai",
    model="text-embedding-3-small",
    private_key="sk-your-key",
)

response = client.llm.embedding(text="How much wood would a woodchuck chuck?")
print(response)
```

## Supported provider families

Research confirmed embedding support for:

- OpenAI.
- DeepInfra via `engine="openai"` plus DeepInfra `base_url`.
- Together via `engine="openai"` plus Together `base_url`.
- Mistral via `engine="openai"` plus Mistral `base_url`.
- Fireworks via `engine="openai"` plus Fireworks `base_url`.

## Together example

```python
client = MagicLLM(
    engine="openai",
    model="BAAI/bge-base-en-v1.5",
    private_key="together-key",
    base_url="https://api.together.xyz/v1",
)

embedding = client.llm.embedding(text="semantic search query")
```

Provider response shape can vary. The library exposes `ModelEmbeddingResponse` where adapters normalize the provider output.
