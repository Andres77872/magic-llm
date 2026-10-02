"""Mandatory request validation and observer-only tool result views."""
from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

from magic_llm.model import ModelChat
from magic_llm.util.tokenizer import from_openai


@dataclass(frozen=True)
class AgentRequestContext:
    """Complete canonical input immediately before a provider request.

    Estimates include nested call/results and tool schemas, but are not a
    provider-specific tokenizer guarantee. Hosts own model limits/headroom.
    Control callbacks are never forwarded as provider generation options.
    """
    chat: ModelChat
    tools: list[Any]
    tool_choice: Any
    provider: str
    model: str | None
    generation_options: dict[str, Any]

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.chat.messages

    def estimated_input_tokens(self) -> int:
        from magic_llm.engine.tooling import normalize_openai_tools
        schemas = normalize_openai_tools(self.tools)
        options = {"tools": schemas, "tool_choice": self.tool_choice}
        return self.chat.num_tokens_from_messages() + len(from_openai(json.dumps(options, default=str)))


def validate_agent_request(guard: Callable | None, *, chat: ModelChat,
                           tools: list[Any], tool_choice: Any, provider: str,
                           client: Any, generation_options: dict[str, Any]) -> None:
    if guard is None:
        return
    chat.require_complete_context()
    engine = getattr(client, 'llm', None)
    options = {**getattr(engine, 'kwargs', {}), **generation_options}
    context = AgentRequestContext(chat=chat, tools=list(tools),
        tool_choice=tool_choice, provider=provider,
        model=options.get('model', getattr(engine, 'model', None)),
        generation_options=options)
    result = guard(context)
    if inspect.isawaitable(result):
        close = getattr(result, 'close', None)
        if close is not None:
            close()
        raise TypeError('request_guard must be synchronous')


def observer_tool_result(transform: Callable | None, result: Any) -> Any:
    """An observer callback receives a copy; canonical inference stays intact."""
    cloned = copy.deepcopy(result)
    return transform(cloned) if transform is not None else cloned
