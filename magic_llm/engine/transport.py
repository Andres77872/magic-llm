"""A host's stand-in for an engine: routing, model chains, request shaping.

A host that decides where each call goes (a chain of models, its own retry
policy, its own request layout) wraps the engine instead of replacing it::

    class Router(EngineTransport):
        async def async_stream_generate(self, chat, **kwargs):
            ...  # choose a model, shape the request, call self.wrapped
            async for chunk in self.wrapped.async_stream_generate(chat, **kwargs):
                yield chunk

    client = MagicLLM(engine="openai", model="m", private_key=key,
                      transport=lambda engine: Router(engine))

``client.llm`` is then the transport. Everything it does not define is the
wrapped engine's, and the agent loop speaks the wrapped engine's tool-calling
dialect (``engine``).
"""

from __future__ import annotations

from typing import Any


class EngineTransport:
    """Stands in for ``client.llm`` and delegates what it does not override."""

    def __init__(self, engine: Any) -> None:
        self.wrapped = engine

    @property
    def engine(self) -> str:
        """The wrapped engine's key, which selects the tool-calling dialect."""
        return getattr(type(self.wrapped), "engine", None) or getattr(self.wrapped, "engine", "openai")

    def __getattr__(self, name: str) -> Any:
        # Private state is never delegated; this also keeps copy and pickle
        # from recursing before ``wrapped`` exists.
        if name.startswith("_") or name == "wrapped":
            raise AttributeError(name)
        return getattr(self.wrapped, name)


__all__ = ["EngineTransport"]
