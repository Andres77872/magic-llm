# Troubleshooting

## Azure chat fails with `STREAM_GENERATION_ERROR` after retries

`engine="azure"` is speech-only. Chat generation raises `NotImplementedError` internally; the chat wrapper retries it and then raises `ChatException` (`error_code="STREAM_GENERATION_ERROR"`, `__context__` is the `NotImplementedError`). Use Azure for speech APIs or switch chat workloads to another engine.

## Model discovery errors

- `NotImplementedError`: Amazon and Cloudflare do not support model discovery.
- A custom OpenAI-compatible endpoint returns 404: verify the provider exposes a model-listing endpoint at the configured base URL. See [model discovery](usage/model-discovery.md).
- `ValueError` on `engine="azure"`: the Azure discovery adapters need Azure OpenAI or Foundry credentials; see [Azure](providers/azure.md#model-discovery).

## Tool-calling error on Amazon Bedrock, Cohere, or Cloudflare

Amazon Bedrock, Cohere, Cloudflare, and Azure raise `TOOL_WIRING_NOT_IMPLEMENTED` when `tools` are passed. Use direct chat or choose OpenAI, Anthropic, Google, or a tool-capable OpenAI-compatible endpoint.

## `embedding()` returns `None`

Embeddings are implemented only for `engine="openai"` and OpenAI-compatible endpoints. Every other engine returns `None` instead of raising.

## OpenAI-compatible provider rejects `max_completion_tokens`

Magic LLM only converts `max_tokens` to `max_completion_tokens` for official `api.openai.com`. If a compatible provider rejects a token argument, pass the argument that provider expects and check the provider docs.

## Official OpenAI ignores `max_tokens`

For official OpenAI, `max_tokens` is transformed to `max_completion_tokens`. If you pass both, `max_completion_tokens` wins.

## `load_subagents()` returns an empty bundle

Subagents are disabled by default. Enable them first:

```python
from magic_llm.agent.config import enable_subagents
enable_subagents()
```

Then call `await client.load_subagents(...)`.

## Tests cannot find API keys

Set `MAGIC_LLM_KEYS` to a JSON file containing your provider keys. There is no implicit local fallback.

```bash
export MAGIC_LLM_KEYS=/path/to/keys.json
chmod 600 /path/to/keys.json
```

## Need to see request payloads

Set `MAGIC_LLM_DEBUG_PAYLOAD=1` for compact summaries or `MAGIC_LLM_DEBUG_PAYLOAD_FULL=1` for full JSON payloads. Do not use full payload logging with sensitive prompts or data.
