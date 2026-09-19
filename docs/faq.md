# FAQ

## Is Magic LLM a replacement for the OpenAI Python SDK?

Not exactly. It aims to provide a unified client surface across many providers while keeping response shapes broadly OpenAI-compatible where possible.

## Does Magic LLM include a CLI?

No. There is no CLI entry point.

## Does Magic LLM include a REST server?

No. It is a Python client library only.

## Does it load `.env` files?

No. Read secrets with your own application configuration and pass them to `MagicLLM(...)`.

## Which engines can I pass to `MagicLLM`?

Use `openai`, `google`, `cloudflare`, `amazon`, `cohere`, `anthropic`, or `azure`.

## How do I use Groq, OpenRouter, Mistral, Together, or DeepInfra?

Use `engine="openai"` plus the provider's OpenAI-compatible `base_url`.

## Why does Azure chat fail?

Azure is speech-only in v0.1.37. Chat methods raise `NotImplementedError`.

## Why does model discovery fail for Amazon or Cloudflare?

Those engines do not have discovery adapters.

## Why do subagents not load?

Subagents default to disabled. Call `enable_subagents()` before `load_subagents()`.

## What test command should I run?

The project test command is:

```bash
python -m pytest
```

The configured default is offline. Live provider tests require an explicit marker selection and `MAGIC_LLM_KEYS` pointing to a private JSON file; they may call paid APIs.
