# Contributing

## Scope

Magic LLM is a Python client library. It has no CLI, no REST server, and no built-in `.env` loader. Keep contributions aligned with that scope unless a design decision explicitly changes it.

## Local workflow

1. Create a virtual environment.
2. Install with `pip install -e .`.
3. Install pre-commit into that virtual environment with `python -m pip install pre-commit`.
4. Install hooks once with `python -m pre_commit install`.
5. Add or update tests for behavior changes.
6. Run the relevant pytest command before opening a PR.
7. Run `python -m pre_commit run --all-files` before opening a PR.
8. Update documentation when public API or provider behavior changes.

## Formatting baseline

The repository uses `.editorconfig` as the shared source of truth for indentation, LF line endings, final newlines, UTF-8, and trailing whitespace cleanup.

PyCharm users should keep EditorConfig support enabled. Reformat Code can stay enabled, but avoid Optimize Imports on save until the project chooses an import-ordering tool in a later phase.

The Phase 1 pre-commit baseline only runs safe hygiene and syntax checks. It does not run Black, Ruff, isort, pyupgrade, autoflake, or mypy.

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
