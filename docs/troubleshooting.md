# Troubleshooting

## `NotImplementedError` with Azure chat

`engine="azure"` is speech-only. Chat generation methods are not implemented. Use Azure for speech APIs or switch chat workloads to another engine.

## `NotImplementedError` during model discovery

Amazon and Cloudflare do not support model discovery. Avoid `client.list_models()` for those engines.

## Tool-calling error on Amazon Bedrock, Cohere, or Cloudflare

Amazon Bedrock, Cohere, and Cloudflare do not support tool calling in Magic LLM v0.1.36. Use direct chat or choose OpenAI, Anthropic, Google, or a tool-capable OpenAI-compatible endpoint.

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

Set `MAGIC_LLM_KEYS` to a JSON file containing your provider keys. Do not rely on the maintainer-local fallback path.

```bash
export MAGIC_LLM_KEYS=/path/to/keys.json
```

## Need to see request payloads

Set `MAGIC_LLM_DEBUG_PAYLOAD=1` for compact summaries or `MAGIC_LLM_DEBUG_PAYLOAD_FULL=1` for full JSON payloads. Do not use full payload logging with sensitive prompts or data.
