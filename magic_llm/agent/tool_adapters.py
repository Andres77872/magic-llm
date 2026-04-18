"""ToolAdapter protocol and ToolAdapterFactory for provider-specific tool serialization.

This module defines:
- ToolAdapter: Runtime-checkable Protocol for provider-specific tool call
  serialization/deserialization. This is the PUBLIC SDK-facing contract that
  AgentLoop depends on.
- ToolAdapterFactory: Factory that selects the appropriate ToolAdapter based
  on engine type, with auto-registration of builtin adapters.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.agent.types import CanonicalToolCall, ToolResult


@runtime_checkable
class ToolAdapter(Protocol):
    """Protocol for provider-specific tool call serialization/deserialization.

    This is the PUBLIC SDK-facing contract that AgentLoop depends on.
    All provider-specific concerns are encapsulated in concrete implementations.
    """

    def serialize_tool_defs(self, tools: list[Any]) -> Any:
        """Convert tool definitions (callables, dicts, Pydantic models) to
        provider-specific request format.

        Returns the value to pass as the 'tools' kwarg to the LLM call.
        """
        ...

    def deserialize_tool_calls(
        self, response: ModelChatResponse
    ) -> list[CanonicalToolCall]:
        """Extract tool calls from a provider response into canonical form."""
        ...

    def serialize_tool_results(
        self, results: list[ToolResult], chat: ModelChat
    ) -> None:
        """Inject tool results into the ModelChat message history in provider-specific format.
        Mutates chat.messages in place.
        """
        ...

    def is_finished(self, response: ModelChatResponse) -> bool:
        """Return True if the response indicates the loop should stop."""
        ...

    def extract_final_text(self, response: ModelChatResponse) -> str:
        """Extract the final text content from a response."""
        ...

    def validate_pair_integrity(self, chat: ModelChat) -> bool:
        """Check that all tool calls have matching results (provider-specific rules).
        Returns True if integrity is valid, False otherwise.
        """
        ...


class ToolAdapterFactory:
    """Factory that selects the appropriate ToolAdapter based on engine type."""

    _registry: dict[str, type[ToolAdapter]] = {}

    @classmethod
    def register(cls, engine_type: str, adapter_cls: type[ToolAdapter]) -> None:
        """Register an adapter class for a given engine type."""
        cls._registry[engine_type] = adapter_cls

    @classmethod
    def create(cls, engine_type: str) -> ToolAdapter:
        """Create an adapter instance for the given engine type.
        Falls back to OpenAIToolAdapter for unrecognized engines.
        """
        adapter_cls = cls._registry.get(engine_type)
        if adapter_cls is None:
            from magic_llm.agent.adapters.openai_adapter import OpenAIToolAdapter

            adapter_cls = OpenAIToolAdapter
        return adapter_cls()

    @classmethod
    def create_for_client(cls, client: Any) -> ToolAdapter:
        """Auto-detect engine type from a MagicLLM client instance.

        Uses client.llm to access the engine, then reads the engine's
        class to determine the engine type string.
        """
        engine_instance = getattr(client, "llm", None)
        if engine_instance is None:
            engine_type = getattr(client, "engine", "unknown")
        else:
            engine_class = type(engine_instance)
            engine_type = getattr(engine_class, "engine", "unknown")
        return cls.create(engine_type)


def _register_builtin_adapters() -> None:
    """Register all builtin adapter classes with the factory.

    Uses lazy imports to avoid circular import issues.
    """
    from magic_llm.agent.adapters.openai_adapter import OpenAIToolAdapter
    from magic_llm.agent.adapters.anthropic_adapter import AnthropicToolAdapter

    ToolAdapterFactory.register("openai", OpenAIToolAdapter)
    ToolAdapterFactory.register("anthropic", AnthropicToolAdapter)

    # OpenAI-compatible providers (all use OpenAIToolAdapter)
    for provider in (
        "openrouter",
        "deepinfra",
        "groq",
        "together",
        "fireworks",
        "anyscale",
        "perplexity",
        "mistral",
        "cerebras",
        "friendliai",
        "novita",
        "deepseek",
        "sambanova",
        "azure",
        "cloudflare",
        "cohere",
    ):
        ToolAdapterFactory.register(provider, OpenAIToolAdapter)

    # NOTE: "google" and "amazon" are NOT registered — they fall back to
    # OpenAIToolAdapter. GeminiToolAdapter and BedrockToolAdapter are OUT OF SCOPE.


# Auto-register adapters on import
_register_builtin_adapters()
