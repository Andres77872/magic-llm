# Providers

Magic LLM has native engines and OpenAI-compatible provider adapters.

## Engine names

| Engine | Provider family | Chat | Tools | Model discovery | Notes |
| --- | --- | --- | --- | --- | --- |
| `openai` | OpenAI and compatible APIs | Yes | Yes for most adapters | Yes for many adapters | Provider selected by `base_url` regex. |
| `anthropic` | Anthropic Claude | Yes | Yes | Yes | Defaults `max_tokens` to 4096 when not supplied. |
| `google` | Google AI Studio / Gemini | Yes | Yes | Yes | Uses Gemini-specific message format. |
| `amazon` | AWS Bedrock | Yes | No | No | Routes by model prefix. |
| `cloudflare` | Cloudflare Workers AI | Yes | No | No | Requires `account_id`. |
| `cohere` | Cohere | Yes | No | Yes | Uses Cohere role/message format. |
| `azure` | Azure Speech runtime audio features | No | No | Not advertised for Azure speech | `engine="azure"` is speech-only; chat methods raise `NotImplementedError`. |

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
