# Providers

Magic LLM has native engines and OpenAI-compatible provider adapters.

## Engine names

| Engine | Provider family | Chat | Tools | Discovery | TTS | STT | Vision input | Image output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai` | OpenAI and compatible APIs | Yes | Yes for most adapters | Yes for many adapters | Provider-specific | Provider-specific | Official OpenAI only by default | No |
| `anthropic` | Anthropic Claude | Yes | Yes | Yes | No | No | Yes for vision-capable Claude models | No |
| `google` | Google AI Studio / Gemini | Yes | Yes | Yes | Sync + async | No | Yes where model supports images | No |
| `amazon` | AWS Bedrock / Polly | Yes | No | No | Polly sync only | No | Nova disabled until native transform exists | No |
| `cloudflare` | Cloudflare Workers AI | Yes | No | No | No | No | No | No |
| `cohere` | Cohere | Yes | No | Yes | No | No | No | No |
| `azure` | Azure Speech runtime audio features | No | No | Not advertised for Azure speech | Async only | Async only | N/A | No |

## Common constructor shape

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
)
```

Provider-specific arguments are documented in the child pages.

## OpenAI-compatible URL routing

These providers are selected through `engine="openai"` plus `base_url`: Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, Together, and other compatible endpoints.

```python
client = MagicLLM(
    engine="openai",
    model="provider-model-name",
    private_key="provider-key",
    base_url="https://provider.example/v1",
)
```

See [openai.md](openai.md) for the important `max_tokens` behavior.

First-class image generation/image output is not part of the current core API. `vision` means image input only.
