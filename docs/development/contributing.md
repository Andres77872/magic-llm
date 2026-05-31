# Contributing

## Scope

Magic LLM is a Python client library. It has no CLI, no REST server, and no built-in `.env` loader. Keep contributions aligned with that scope unless a design decision explicitly changes it.

## Local workflow

1. Create a virtual environment.
2. Install with `pip install -e .`.
3. Add or update tests for behavior changes.
4. Run the relevant pytest command before opening a PR.
5. Update documentation when public API or provider behavior changes.

## Provider changes

When changing provider behavior, document:

- Engine name and constructor args.
- Supported chat methods.
- Streaming/async support.
- Tool-calling support.
- Embedding/audio support.
- Model discovery support.
- Any argument transformations such as OpenAI `max_tokens` handling.

## Security

- Never commit real API keys or sample files containing private data.
- Do not enable full payload logging in shared environments.
- Prefer placeholder keys in docs: `sk-your-key`, `gsk-your-key`, `cloudflare-api-token`.

## Documentation standard

Docs should be accurate over optimistic. If a behavior is uncertain or provider-dependent, say so clearly instead of guessing.
