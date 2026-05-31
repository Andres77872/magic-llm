# Magic LLM documentation

Magic LLM is a Python 3.10+ client library that normalizes access to 21+ LLM providers through the `MagicLLM` class. The library focuses on application integration: chat, streaming, async generation, embeddings, audio, model discovery, tool calling, agents, and subagents.

## Start here

1. [Install Magic LLM](installation.md)
2. [Run the 5-minute quickstart](quickstart.md)
3. [Choose a provider](providers/index.md)
4. [Learn the chat API](usage/chat.md)
5. [Add tools or agents](agents/index.md)

## Usage guides

- [Chat](usage/chat.md) — `generate`, `stream_generate`, `async_generate`, `async_stream_generate`.
- [Vision](usage/vision.md) — text plus image inputs through `ModelChat.add_user_message`.
- [Embeddings](usage/embeddings.md) — text embedding support and provider caveats.
- [Audio](usage/audio.md) — speech-to-text and text-to-speech request models.
- [Error handling](usage/error-handling.md) — `ChatException`, fallback clients, callbacks, retry behavior.
- [Model discovery](usage/model-discovery.md) — normalized model listing and unsupported engines.

## Provider guides

- [Provider overview](providers/index.md)
- [OpenAI and OpenAI-compatible providers](providers/openai.md)
- [Anthropic](providers/anthropic.md)
- [Google AI Studio](providers/google.md)
- [Amazon Bedrock](providers/amazon.md)
- [Cloudflare Workers AI](providers/cloudflare.md)
- [Cohere](providers/cohere.md)
- [Azure Speech](providers/azure.md)

## Agents

- [Agent overview](agents/index.md)
- [Tool calling](agents/tool-calling.md)
- [Budgets and hooks](agents/budget-hooks.md)
- [Subagents](agents/subagents.md)
- [Prompt fragments](agents/prompt-fragment.md)

## Development

- [Setup](development/setup.md)
- [Testing](development/testing.md)
- [Contributing](development/contributing.md)

## References

- [Troubleshooting](troubleshooting.md)
- [FAQ](faq.md)

## Non-goals

Magic LLM does not provide a CLI, web server, REST API, or `.env` loader. Use it as a Python library inside your own application and pass credentials explicitly from your own configuration layer.
