# Installation

## Requirements

- Python 3.10 or newer. The codebase uses modern union type syntax such as `str | None`.
- Network access to the provider APIs you plan to use.
- API keys or cloud credentials for those providers.

## Install from GitHub

The primary install path is the Git repository:

```bash
python -m pip install 'magic_llm @ git+https://github.com/Andres77872/magic-llm.git@master'
```

For local development:

```bash
git clone https://github.com/Andres77872/magic-llm.git
cd magic-llm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Don't install by bare name from public PyPI; `magic-llm` and `magic_llm` resolve to an unrelated project there. This repo installs independently of `magic-agents`.

## Core dependencies

`setup.cfg` declares direct runtime dependencies and test tools separately. For Jupyter, install the `notebook` extra (`python -m pip install -e '.[notebook]'` from the checkout).

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
