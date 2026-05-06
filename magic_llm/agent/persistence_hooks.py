"""Reusable AgentHooks persistence implementations for magic-llm.

These hooks own generic magic-llm agent-loop lifecycle translation only.
They do not import service/API repositories. Persistence side effects are
performed by injected sinks supplied by the consuming application.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from magic_llm.agent.types import AgentState
from magic_llm.model import ModelChatResponse


@runtime_checkable
class AgentPersistenceSink(Protocol):
    """Sink contract for agent-loop persistence side effects."""

    def on_loop_start(self, **payload: Any) -> None: ...

    def on_iteration_start(self, **payload: Any) -> None: ...

    def record_usage(self, **payload: Any) -> None: ...

    def on_loop_complete(self, **payload: Any) -> None: ...

    def on_loop_error(self, **payload: Any) -> None: ...


class AgentPersistenceHooks:
    """AgentHooks implementation backed by an injected persistence sink.

    The hook forwards loop lifecycle and usage facts with the canonical
    assistant ``id_message``. Failures are explicit and never fabricate usage.
    """

    def __init__(
        self,
        *,
        sink: AgentPersistenceSink,
        id_message: str | None = None,
        provider: str | None = None,
        model_alias: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._sink = sink
        self._id_message = id_message
        self._provider = provider
        self._model_alias = model_alias
        self._run_metadata = dict(run_metadata or {})
        self._started = False
        self._failed = False

    @property
    def id_message(self) -> str | None:
        """Canonical assistant message ID forwarded to sinks."""

        return self._id_message

    def on_iteration_start(self, iteration: int, state: AgentState) -> None:
        """Start loop lazily and record iteration metadata from ``state.step``."""

        self._ensure_started()
        effective_iteration = _state_iteration(state, fallback=iteration)
        self._sink.on_iteration_start(
            id_message=self._id_message,
            iteration=effective_iteration,
            loop_iteration_arg=iteration,
            metadata={
                "iteration": effective_iteration,
                "message_count": len(state.messages or []),
            },
        )

    def on_llm_response(self, response: ModelChatResponse, state: AgentState) -> None:
        """Forward provider usage when usage exists and has real tokens."""

        usage = getattr(response, "usage", None)
        usage_data = _usage_to_dict(usage)
        if not usage_data:
            return

        self._sink.record_usage(
            id_message=self._id_message,
            provider=self._provider,
            model_alias=self._model_alias or getattr(response, "model", None),
            usage_data=usage_data,
            metadata={"iteration": _state_iteration(state)},
        )

    def on_loop_complete(self, final_response: ModelChatResponse, state: AgentState) -> None:
        """Record successful loop completion."""

        self._ensure_started()
        self._sink.on_loop_complete(
            status="success",
            id_message=self._id_message,
            content=getattr(final_response, "content", None),
            metadata={
                "iteration": _state_iteration(state),
                "total_input_tokens": state.total_input_tokens,
                "total_output_tokens": state.total_output_tokens,
                "finish_reason": getattr(final_response, "finish_reason", None),
                "model": getattr(final_response, "model", None),
            },
        )

    def on_loop_error(self, error: Exception, state: AgentState | None = None) -> None:
        """Record loop failure without emitting fabricated usage."""

        self._ensure_started()
        self._failed = True
        self._sink.on_loop_error(
            status="error",
            id_message=self._id_message,
            error_type=type(error).__name__,
            error_message=str(error),
            metadata={"iteration": _state_iteration(state)} if state is not None else {},
        )

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._sink.on_loop_start(
            id_message=self._id_message,
            metadata=dict(self._run_metadata),
        )


def _state_iteration(state: AgentState | None, *, fallback: int | None = None) -> int | None:
    if state is not None and getattr(state, "step", None) is not None:
        return state.step
    return fallback


def _usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    total_tokens = total_tokens or 0
    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


__all__ = [
    "AgentPersistenceHooks",
    "AgentPersistenceSink",
]
