"""AgentLoop — synchronous ReAct-style agent loop.

Encapsulates the full ReAct-style loop as a state machine with
dependency-injected orchestration components (ToolExecutor, AgentHooks).
Engine/core tooling owns provider-specific tool mapping and result injection;
ToolAdapter remains only as a deprecated compatibility constructor option.

NOT thread-safe. Each instance should be used by a single thread.
Concurrent .run()/.stream() calls on the same instance raise RuntimeError.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Iterator, Optional

from magic_llm.engine.tooling import (
    StreamIterationSummary,
    accumulate_stream_chunk,
    append_tool_results,
    extract_tool_calls,
    infer_provider_from_client,
    is_finished,
    stream_summary_tool_calls,
    validate_tool_result_integrity,
)
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
from magic_llm.agent.types import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentState,
    CanonicalToolCall,
)
from magic_llm.agent.hooks import AgentHooks
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.tool_adapters import ToolAdapter, ToolAdapterFactory
from magic_llm.agent._loop_shared import (
    _build_initial_chat,
    _check_budget,
    _compute_fingerprint,
    _finalize_response,
    _invoke_hook_safely,
    _register_tools_with_executor,
)

logger = logging.getLogger(__name__)


class AgentLoop:
    """Synchronous ReAct-style agent loop.

    NOT thread-safe. Each instance should be used by a single thread.
    Concurrent .run()/.stream() calls on the same instance raise RuntimeError.

    Args:
        client: A MagicLLM client instance (or any object with .llm.generate()).
        tools: A list of tool definitions (callables, dict specs, or Pydantic models).
        tool_functions: Dict mapping custom names to callables.
        budget: An optional AgentBudget instance (defaults to AgentBudget()).
        hooks: An optional AgentHooks implementation (defaults to no-op).
        adapter: Deprecated compatibility option retained for direct callers.
            Canonical tooling behavior does not use this adapter.
        tool_executor: An optional ToolExecutor instance. Created internally if None.
        deduplicate: Enable tool deduplication (default: False, opt-in).
        content_separator: String to join content between iterations (default: "\\n\\n").
        tool_choice: Tool choice parameter for the LLM (default: "auto").
        **kwargs: Extra kwargs stored for passthrough to LLM calls.
            Note: 'engine_type' is explicitly ignored.
    """

    def __init__(
        self,
        client: Any,
        tools: Optional[list[Any]] = None,
        tool_functions: Optional[dict[str, Callable[..., Any]]] = None,
        budget: Optional[AgentBudget] = None,
        hooks: Optional[AgentHooks] = None,
        adapter: Optional[ToolAdapter] = None,
        tool_executor: Optional[ToolExecutor] = None,
        deduplicate: bool = False,
        content_separator: str = "\n\n",
        tool_choice: str | dict[str, Any] | None = "auto",
        prompt_fragment: str | Callable[..., str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._client = client
        self._prompt_fragment = prompt_fragment

        # Store tools for registration at run time
        self._tools = tools or []
        self._tool_functions = tool_functions or {}

        # Budget defaults
        self._budget = budget if budget is not None else AgentBudget()
        self._hooks = hooks  # None = no-op (handled by _invoke_hook_safely)

        # Adapter: explicit override takes precedence over auto-detection
        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = ToolAdapterFactory.create_for_client(client)

        # Tool executor: explicit override or create internally
        if tool_executor is not None:
            self._executor = tool_executor
        else:
            self._executor = ToolExecutor(enable_dedup=deduplicate)

        self._deduplicate = deduplicate
        self._content_separator = content_separator
        self._tool_choice = tool_choice
        self._provider = infer_provider_from_client(client)

        # Store kwargs for passthrough, but explicitly drop engine_type
        engine_type = kwargs.pop("engine_type", None)
        if engine_type is not None:
            logger.warning(
                "engine_type=%r is ignored; provider is inferred from client.llm",
                engine_type,
            )
        self._generate_kwargs = kwargs

        # Concurrency guard
        self._lock = threading.Lock()
        self._running = False

        # Internal state
        self._state = AgentState()

    @property
    def state(self) -> AgentState:
        """Return a read-only copy of the current agent state.

        Mutations to the returned state do NOT affect internal loop state.
        """
        return AgentState(
            messages=list(self._state.messages),
            step=self._state.step,
            total_input_tokens=self._state.total_input_tokens,
            total_output_tokens=self._state.total_output_tokens,
            executed_fingerprints=set(self._state.executed_fingerprints),
            start_time=self._state.start_time,
        )

    def _resolve_prompt_fragment(self, **kwargs: Any) -> str:
        """Resolve the prompt_fragment at generation time.

        C4: If prompt_fragment is None, return empty string.
            If callable, invoke with forwarded kwargs and return result.
            Otherwise return as static string.

        Returns:
            The resolved prompt_fragment string, or "" if None.
        """
        if self._prompt_fragment is None:
            return ""
        if callable(self._prompt_fragment):
            return self._prompt_fragment(**kwargs)
        return self._prompt_fragment

    def _acquire_lock(self) -> None:
        """Acquire the concurrency lock. Raises RuntimeError if already running."""
        if self._running:
            raise RuntimeError(
                "AgentLoop instance is already running. "
                "Do not call .run() or .stream() concurrently on the same instance."
            )
        self._running = True

    def _release_lock(self) -> None:
        """Release the concurrency lock (reset _running flag)."""
        self._running = False

    def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[list[dict[str, Any]]] = None,
    ) -> ModelChatResponse:
        """Execute the full ReAct loop synchronously.

        State machine order:
        1. INIT → 2. LLM_CALL → 3. CHECK_BUDGET → 4. HOOK → 5. EXTRACT →
        6. CHECK_DONE → 7. RECORD_CONTENT → 8. VALIDATE_INTEGRITY →
        9. EXECUTE → 10. INJECT → 11. LOOP

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            extra_messages: Optional list of message dicts before user_input.

        Returns:
            The final ModelChatResponse with concatenated content.

        Raises:
            RuntimeError: If called while the loop is already running.
            AgentBudgetExceeded: If any budget constraint is violated.
        """
        # INIT: Resolve prompt_fragment and prepend to system prompt (C3)
        pf = self._resolve_prompt_fragment(**self._generate_kwargs)
        if pf:
            system_prompt = (
                f"{pf}\n\n{system_prompt}".strip()
                if system_prompt
                else pf
            )

        # Build initial chat
        chat = _build_initial_chat(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        )

        # Register tools
        _register_tools_with_executor(
            self._executor,
            tools=self._tools,
            tool_functions=self._tool_functions,
        )

        # Reset dedup fingerprints for this run
        if self._deduplicate:
            self._executor._dedup_cache.clear()

        # Initialize state
        self._state = AgentState(
            messages=chat.messages,
            step=0,
            start_time=time.monotonic(),
        )

        collected_content: list[str] = []
        response: Optional[ModelChatResponse] = None

        # Acquire concurrency guard
        self._acquire_lock()
        try:
            while True:
                # Step 3: CHECK_BUDGET (pre-call: iterations + wall-clock)
                try:
                    _check_budget(self._state, self._budget, include_tokens=False)
                except AgentBudgetExceeded as exc:
                    # Fire on_budget_exceeded BEFORE exception propagates
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_budget_exceeded", None),
                        exc.budget_type,
                        str(exc),
                        state=self.state,
                    )
                    raise

                # Hook: on_iteration_start
                _invoke_hook_safely(
                    getattr(self._hooks, "on_iteration_start", None),
                    self._state.step,
                    self.state,
                    state=self.state,
                )

                # Step 2: LLM_CALL — pass raw tools to engine/core tooling.
                response = self._client.llm.generate(
                    chat,
                    tools=self._tools,
                    tool_choice=self._tool_choice,
                    **self._generate_kwargs,
                )

                # Update token counts from response usage
                if response.usage is not None:
                    self._state.total_input_tokens += getattr(
                        response.usage, "prompt_tokens", 0
                    ) or 0
                    self._state.total_output_tokens += getattr(
                        response.usage, "completion_tokens", 0
                    ) or 0

                # Step 3 (post-call): CHECK_BUDGET (tokens)
                try:
                    _check_budget(self._state, self._budget, include_tokens=True)
                except AgentBudgetExceeded as exc:
                    # Fire on_budget_exceeded BEFORE exception propagates
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_budget_exceeded", None),
                        exc.budget_type,
                        str(exc),
                        state=self.state,
                    )
                    raise

                # Step 4: HOOK — on_llm_response
                _invoke_hook_safely(
                    getattr(self._hooks, "on_llm_response", None),
                    response,
                    self.state,
                    state=self.state,
                )

                # Step 5: EXTRACT — consume normalized engine response tool calls.
                tool_calls: list[CanonicalToolCall] = extract_tool_calls(response)

                # Step 6: RECORD content BEFORE checking done/break
                content = response.content
                if content and not tool_calls:
                    collected_content.append(content)
                    chat.add_assistant_message(content)

                # Step 7: CHECK_DONE — no tool calls OR is_finished
                # INVARIANT: Content recording runs BEFORE break so the final
                # content-only assistant message is in state when the loop exits.
                if not tool_calls or is_finished(self._provider, response):
                    # Exit loop
                    break

                # Step 7b: ADD_TOOL_CALL — tool-call message (no speculative content)
                if tool_calls:
                    # Convert CanonicalToolCall to dict format for chat
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                    chat.add_tool_call_message(
                        tool_calls=tool_call_dicts,
                        content=None,  # INVARIANT: no speculative content in LLM context
                    )

                # Step 8: VALIDATE_INTEGRITY
                validate_tool_result_integrity(self._provider, chat)

                # Hook: on_tool_start — invoke BEFORE execution for EACH tool call
                for tc in tool_calls:
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_tool_start", None),
                        tc.name,
                        tc.id,
                        tc.arguments,  # actual arguments, not empty dict
                        self.state,
                        state=self.state,
                    )

                # Step 9: EXECUTE — run tools in parallel
                results = self._executor.execute_parallel(tool_calls)

                # Hook: on_tool_complete — invoke AFTER execution for each result
                for result in results:
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_tool_complete", None),
                        result,
                        self.state,
                        state=self.state,
                    )

                # Step 10: INJECT — engine/core builds provider-correct messages.
                append_tool_results(self._provider, chat, results)

                # Update state messages reference
                self._state.messages = chat.messages

                # Step 11: LOOP — increment step
                self._state.step += 1

        finally:
            self._release_lock()

        # Finalize response
        if response is not None:
            _finalize_response(
                response, collected_content, self._content_separator
            )

        # Hook: on_loop_complete
        if response is not None:
            _invoke_hook_safely(
                getattr(self._hooks, "on_loop_complete", None),
                response,
                self.state,
                state=self.state,
            )

        return response

    def stream(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[list[dict[str, Any]]] = None,
    ) -> Iterator[ChatCompletionModel]:
        """Stream chunks from the LLM, executing tools between iterations.

        Yields ChatCompletionModel chunks. Between iterations with tool calls,
        a separator chunk is yielded.

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            extra_messages: Optional list of message dicts before user_input.

        Yields:
            ChatCompletionModel chunks from the streaming LLM.

        Raises:
            RuntimeError: If called while the loop is already running.
        """
        # INIT: Resolve prompt_fragment and prepend to system prompt (C3)
        pf = self._resolve_prompt_fragment(**self._generate_kwargs)
        if pf:
            system_prompt = (
                f"{pf}\n\n{system_prompt}".strip()
                if system_prompt
                else pf
            )

        # Build initial chat
        chat = _build_initial_chat(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        )

        # Register tools
        _register_tools_with_executor(
            self._executor,
            tools=self._tools,
            tool_functions=self._tool_functions,
        )

        # Reset dedup fingerprints for this run
        if self._deduplicate:
            self._executor._dedup_cache.clear()

        # Initialize state
        self._state = AgentState(
            messages=chat.messages,
            step=0,
            start_time=time.monotonic(),
        )

        # Accumulated content across all iterations (for on_loop_complete)
        collected_content: list[str] = []
        response: Optional[ModelChatResponse] = None

        # Acquire concurrency guard
        self._acquire_lock()
        # Track whether budget was exceeded — if so, skip on_loop_complete
        # in the finally block (budget-exceeded uses on_budget_exceeded instead)
        _budget_exceeded = False
        try:
            first_iteration = True

            while True:
                # Pre-call budget check (iterations + wall-clock)
                try:
                    _check_budget(self._state, self._budget, include_tokens=False)
                except AgentBudgetExceeded as exc:
                    _budget_exceeded = True
                    # Fire on_budget_exceeded BEFORE exception propagates
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_budget_exceeded", None),
                        exc.budget_type,
                        str(exc),
                        state=self.state,
                    )
                    raise

                # Hook: on_iteration_start
                _invoke_hook_safely(
                    getattr(self._hooks, "on_iteration_start", None),
                    self._state.step,
                    self.state,
                    state=self.state,
                )

                # Stream from LLM. Engine/provider chunks are already normalized;
                # the agent loop only accumulates via the engine/core summary helper.
                summary = StreamIterationSummary()
                last_chunk: Optional[ChatCompletionModel] = None

                for chunk in self._client.llm.stream_generate(
                    chat,
                    tools=self._tools,
                    tool_choice=self._tool_choice,
                    **self._generate_kwargs,
                ):
                    accumulate_stream_chunk(summary, chunk)
                    yield chunk
                    last_chunk = chunk

                # Build a synthetic response from normalized engine/core stream summary.
                if last_chunk is not None:
                    response = ModelChatResponse(
                        id=last_chunk.id or "stream-synthetic",
                        object="chat.completion",
                        created=last_chunk.created or 0.0,
                        model=last_chunk.model,
                        choices=[
                            Choice(
                                index=0,
                                message=Message(
                                    role="assistant",
                                    content=summary.content if summary.content else None,
                                    tool_calls=None,
                                ),
                                finish_reason=summary.finish_reason,
                            )
                        ],
                    )

                    # Update token counts
                    if last_chunk.usage is not None:
                        self._state.total_input_tokens += getattr(
                            last_chunk.usage, "prompt_tokens", 0
                        ) or 0
                        self._state.total_output_tokens += getattr(
                            last_chunk.usage, "completion_tokens", 0
                        ) or 0

                    # Post-call budget check (tokens)
                    try:
                        _check_budget(self._state, self._budget, include_tokens=True)
                    except AgentBudgetExceeded as exc:
                        _budget_exceeded = True
                        # Fire on_budget_exceeded BEFORE exception propagates
                        _invoke_hook_safely(
                            getattr(self._hooks, "on_budget_exceeded", None),
                            exc.budget_type,
                            str(exc),
                            state=self.state,
                        )
                        raise

                    # Hook: on_llm_response
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_llm_response", None),
                        response,
                        self.state,
                        state=self.state,
                    )

                    tool_calls = stream_summary_tool_calls(summary)

                    # Record content — runs for EVERY iteration (including final no-tool answer)
                    # INVARIANT: When tool_calls are present, pre-tool content is
                    # speculative and MUST be suppressed — condition `not tool_calls`
                    # ensures only content-only iterations are recorded.
                    if summary.content and not tool_calls:
                        iter_content = summary.content
                        chat.add_assistant_message(iter_content)
                        collected_content.append(iter_content)
                        self._state.messages = chat.messages

                    # Check done — AFTER content recording so final content-only
                    # assistant message is synced to state before the break.
                    if not tool_calls or is_finished(self._provider, response):
                        break

                    # Add tool_call message (no speculative content)
                    if tool_calls:
                        tool_call_dicts = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in tool_calls
                        ]
                        chat.add_tool_call_message(
                            tool_calls=tool_call_dicts,
                            content=None,  # INVARIANT: no speculative content in LLM context
                        )

                    # Validate integrity
                    validate_tool_result_integrity(self._provider, chat)

                    # Hook: on_tool_start — invoke BEFORE execution for EACH tool call
                    for tc in tool_calls:
                        _invoke_hook_safely(
                            getattr(self._hooks, "on_tool_start", None),
                            tc.name,
                            tc.id,
                            tc.arguments,  # actual arguments, not empty dict
                            self.state,
                            state=self.state,
                        )

                    # Execute tools
                    results = self._executor.execute_parallel(tool_calls)

                    # Hook: on_tool_complete — invoke AFTER execution for each result
                    for result in results:
                        _invoke_hook_safely(
                            getattr(self._hooks, "on_tool_complete", None),
                            result,
                            self.state,
                            state=self.state,
                        )

                    # Inject results
                    append_tool_results(self._provider, chat, results)
                    self._state.messages = chat.messages

                    # Yield separator between iterations
                    if summary.content and self._content_separator and last_chunk:
                        separator_chunk = ChatCompletionModel(
                            id=f"separator-{self._state.step}",
                            model=last_chunk.model,
                            choices=[
                                ChoiceModel(
                                    index=0,
                                    delta=DeltaModel(content=self._content_separator),
                                )
                            ],
                        )
                        yield separator_chunk

                self._state.step += 1
                first_iteration = False

        finally:
            # Fire on_loop_complete with accumulated response
            # Runs for normal exit, generator close(), and exceptions.
            # Does NOT fire on budget-exceeded — that path uses on_budget_exceeded instead.
            if response is not None:
                _finalize_response(response, collected_content, self._content_separator)
            if not _budget_exceeded:
                _invoke_hook_safely(
                    getattr(self._hooks, "on_loop_complete", None),
                    response,
                    self.state,
                    state=self.state,
                )
            self._release_lock()
