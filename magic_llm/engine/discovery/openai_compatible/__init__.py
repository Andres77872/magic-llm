"""OpenAI-compatible discovery adapters.

Each provider in this subpackage gets its own concrete adapter that owns
its default discovery URL. No central URL table, no base_url normalization.

Submodules are imported here so their ``register_adapter()`` calls fire
when the package is imported.
"""

# Import all concrete adapter submodules so they self-register.
# fmt: off
from magic_llm.engine.discovery.openai_compatible import deepinfra  # noqa: F401
from magic_llm.engine.discovery.openai_compatible import groq       # noqa: F401
from magic_llm.engine.discovery.openai_compatible import novita     # noqa: F401
from magic_llm.engine.discovery.openai_compatible import perplexity # noqa: F401
from magic_llm.engine.discovery.openai_compatible import together   # noqa: F401
from magic_llm.engine.discovery.openai_compatible import mistral    # noqa: F401
from magic_llm.engine.discovery.openai_compatible import deepseek   # noqa: F401
from magic_llm.engine.discovery.openai_compatible import hyperbolic # noqa: F401
from magic_llm.engine.discovery.openai_compatible import cerebras   # noqa: F401
from magic_llm.engine.discovery.openai_compatible import xai        # noqa: F401
from magic_llm.engine.discovery.openai_compatible import parasail   # noqa: F401
from magic_llm.engine.discovery.openai_compatible import nebius     # noqa: F401
# fmt: on
