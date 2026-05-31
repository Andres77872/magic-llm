# Installation

## Requirements

- Python 3.10 or newer. The codebase uses modern union type syntax such as `str | None`.
- Network access to the provider APIs you plan to use.
- API keys or cloud credentials for those providers.

## Install from GitHub

The primary install path is the Git repository:

```bash
pip install git+https://github.com/Andres77872/magic-llm.git
```

For local development:

```bash
git clone https://github.com/Andres77872/magic-llm.git
cd magic-llm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Install from PyPI, if available

The package metadata names the distribution `magic_llm`. If the package is published in your environment, one of these may be available:

```bash
pip install magic-llm
# or
pip install magic_llm
```

If that fails, use the GitHub install command above.

## Core dependencies

Project metadata includes `aiohttp`, `requests`, `pydantic` v2, `PyYAML`, `tiktoken`, `tokenizers`, and `botocore`, plus testing and utility packages.

## Credentials

Magic LLM does not load `.env` files. Pass credentials directly:

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
)
```

In real applications, read the key from your own secret manager or environment layer first, then pass it to the constructor. Do not hard-code real keys in source files.
