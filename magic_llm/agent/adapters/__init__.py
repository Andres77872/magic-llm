# magic_llm.agent.adapters — Internal provider-specific tool adapter implementations.
#
# This subpackage is NOT part of the public API. Adapters are accessed
# via the ToolAdapterFactory or by direct import for explicit override.
#
# No eager imports — adapters are loaded on demand by the factory.

from magic_llm.agent.adapters.openai_adapter import OpenAIToolAdapter
from magic_llm.agent.adapters.anthropic_adapter import AnthropicToolAdapter

__all__ = ["OpenAIToolAdapter", "AnthropicToolAdapter"]
