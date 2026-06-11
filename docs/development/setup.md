# Development setup

## Clone and install

```bash
git clone https://github.com/Andres77872/magic-llm.git
cd magic-llm
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pip install -r requirements.txt
```

The project uses setuptools through `setup.cfg` and `pyproject.toml`. The managed test stack is pinned in `requirements.txt`; editable installs keep pytest/plugin dependencies below unsupported major versions.

## Python version

Use Python 3.10+. The codebase uses `str | None` syntax and other modern typing forms.

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

## Test credentials

Default offline tests do not require credentials and must not open credential files during collection.

For explicit live/provider jobs only, point `MAGIC_LLM_KEYS` at a local JSON file:

```bash
export MAGIC_LLM_KEYS=/path/to/local/keys.json
```

The maintainer-local fallback `/home/andres/Documents/keys.json` is supported only as a local convenience for explicit live runs. It is not a CI contract, not required for offline tests, and its contents must never be printed, copied, committed, or persisted.

Credential file shape is provider-keyed JSON. Use your real values locally; do not paste them into docs, logs, commits, or SDD/RDD artifacts.

Optional explicit live media fixtures:

```bash
export MAGIC_LLM_AUDIO_FILE=/path/to/sample.wav
export MAGIC_LLM_IMAGE_B64_FILE=/path/to/image-base64.txt
```

## Safe test commands

```bash
python -m pytest --collect-only -q -m "not provider_functional and not provider_health"
python -m pytest test/ -m "not provider_functional and not provider_health" -v
```

Live commands must select `provider_functional` or `provider_health` explicitly and should set `MAGIC_LLM_KEYS`.

## Debugging provider payloads

```bash
export MAGIC_LLM_DEBUG_PAYLOAD=1
export MAGIC_LLM_DEBUG_PAYLOAD_FULL=1
```

Do not enable full payload logging with sensitive prompts, keys, or customer data.
