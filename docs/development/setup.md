# Development setup

## Clone and install

```bash
git clone https://github.com/Andres77872/magic-llm.git
cd magic-llm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project uses setuptools through `setup.cfg` and `pyproject.toml`.

## Pre-commit hooks

Install hook tooling into the active virtual environment, then register the repository hooks:

```bash
python -m pip install pre-commit
python -m pre_commit install
```

Before opening a PR, run:

```bash
python -m pre_commit run --all-files
```

## Python version

Use Python 3.10+. The codebase uses `str | None` syntax and other modern typing forms.

## Test credentials

Integration tests look for API keys in a JSON file. Set:

```bash
export MAGIC_LLM_KEYS=/path/to/keys.json
```

If unset, tests fall back to a maintainer-local path. Create your own file instead of relying on that fallback.

Example shape:

```json
{
  "openai": "sk-...",
  "anthropic": "sk-ant-...",
  "google": "...",
  "cohere": "...",
  "cloudflare": "...",
  "aws_access_key_id": "AKIA...",
  "aws_secret_access_key": "...",
  "region_name": "us-east-1",
  "azure_speech_key": "...",
  "azure_speech_region": "eastus"
}
```

Optional test fixture files:

```bash
export MAGIC_LLM_AUDIO_FILE=/path/to/sample.wav
export MAGIC_LLM_IMAGE_B64_FILE=/path/to/image-base64.txt
```

## Debugging provider payloads

```bash
export MAGIC_LLM_DEBUG_PAYLOAD=1
export MAGIC_LLM_DEBUG_PAYLOAD_FULL=1
```

Do not enable full payload logging with sensitive prompts, keys, or customer data.
