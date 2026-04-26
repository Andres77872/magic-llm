"""AsyncAgentLoop — asynchronous ReAct-style agent loop.

Mirrors the AgentLoop API but uses async def for run() and stream().
Uses asyncio.Lock for concurrency guard (not threading.Lock).

NOT safe for concurrent asyncio tasks. Each instance should be used by
a single asyncio task. Concurrent .run()/.stream() calls on the same
instance raise RuntimeError.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, Optional

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel,
    ChoiceModel,
    DeltaModel,
)
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
    _finalize_response,
    _invoke_hook_safely,
    _register_tools_with_executor,
    # Parent context ContextVars for nested LLM node execution
    PARENT_BUDGET,
    PARENT_STATE,
)


class AsyncAgentLoop:
    """Asynchronous ReAct-style agent loop.

    NOT safe for concurrent asyncio tasks. Each instance should be used by
    a single asyncio task. Concurrent .run()/.stream() calls on the same
    instance raise RuntimeError.

    Args:
        client: A MagicLLM client instance (or any object with .llm.async_generate()).
        tools: A list of tool definitions (callables, dict specs, or Pydantic models).
        tool_functions: Dict mapping custom names to callables.
        budget: An optional AgentBudget instance (defaults to AgentBudget()).
        hooks: An optional AgentHooks implementation (defaults to no-op).
        adapter: An optional ToolAdapter instance. Auto-detected if None.
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
        **kwargs: Any,
    ) -> None:
        self._client = client

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

        # Store kwargs for passthrough, but explicitly drop engine_type
        kwargs.pop("engine_type", None)
        self._generate_kwargs = kwargs

        # Concurrency guard (asyncio.Lock, NOT threading.Lock)
        self._lock = asyncio.Lock()
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

    def _acquire_lock(self) -> None:
        """Check the _running flag. Raises RuntimeError if already running.

        Note: The actual asyncio.Lock is acquired in run()/stream() with await.
        This method only checks the flag for a clear error message.
        """
        if self._running:
            raise RuntimeError(
                "AsyncAgentLoop instance is already running. "
                "Do not call .run() or .stream() concurrently on the same instance."
            )
        self._running = True

    def _release_lock(self) -> None:
        """Release the concurrency lock (reset _running flag)."""
        self._running = False

    async def run(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[list[dict[str, Any]]] = None,
        initial_chat: Optional[ModelChat] = None,
    ) -> ModelChatResponse:
        """Execute the full ReAct loop asynchronously.

        State machine order (same as sync AgentLoop):
        1. INIT → 2. LLM_CALL → 3. CHECK_BUDGET → 4. HOOK → 5. EXTRACT →
        6. CHECK_DONE → 7. RECORD_CONTENT → 8. VALIDATE_INTEGRITY →
        9. EXECUTE → 10. INJECT → 11. LOOP

        Uses await client.llm.async_generate() and
        await executor.execute_parallel_async().

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            extra_messages: Optional list of message dicts before user_input.
            initial_chat: Optional prebuilt chat to use directly.

        Returns:
            The final ModelChatResponse with concatenated content.

        Raises:
            RuntimeError: If called while the loop is already running.
            AgentBudgetExceeded: If any budget constraint is violated.
        """
        # INIT: Build initial chat
        chat = _build_initial_chat(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
            initial_chat=initial_chat,
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

        # Set parent budget/state ContextVars for nested LLM node execution
        # Child tasks can read these via PARENT_BUDGET.get() and PARENT_STATE.get()
        # Capture tokens for cleanup in finally block (prevent cross-run contamination)
        parent_budget_token = PARENT_BUDGET.set(self._budget)
        parent_state_token = PARENT_STATE.set(self._state)

        collected_content: list[str] = []
        response: Optional[ModelChatResponse] = None

        # Acquire concurrency guard
        self._acquire_lock()
        try:
            await self._lock.acquire()
            try:
                while True:
                    # Step 3: CHECK_BUDGET (pre-call: iterations + wall-clock)
                    _check_budget(self._state, self._budget, include_tokens=False)

                    # Hook: on_iteration_start
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_iteration_start", None),
                        self._state.step,
                        self.state,
                    )

                    # Step 2: LLM_CALL (async)
                    tool_defs = self._adapter.serialize_tool_defs(self._tools)
                    response = await self._client.llm.async_generate(
                        chat,
                        tools=tool_defs,
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
                    _check_budget(self._state, self._budget, include_tokens=True)

                    # Step 4: HOOK — on_llm_response
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_llm_response", None),
                        response,
                        self.state,
                    )

                    # Step 5: EXTRACT — use adapter to deserialize tool calls
                    tool_calls: list[CanonicalToolCall] = (
                        self._adapter.deserialize_tool_calls(response)
                    )

                    # Step 6: CHECK_DONE — no tool calls OR is_finished
                    if not tool_calls or self._adapter.is_finished(response):
                        break

                    # Step 7: RECORD_CONTENT
                    content = response.content
                    if content:
                        collected_content.append(content)
                        chat.add_assistant_message(content)

                    # Also add tool_call message when tool_calls exist
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
                            content=content,
                        )

                    # Step 8: VALIDATE_INTEGRITY
                    self._adapter.validate_pair_integrity(chat)

                    # Step 9: EXECUTE — run tools in parallel (async)
                    results = await self._executor.execute_parallel_async(tool_calls)

                    # Invoke hooks for each tool result
                    for result in results:
                        _invoke_hook_safely(
                            getattr(self._hooks, "on_tool_start", None),
                            result.name,
                            result.tool_call_id or "",
                            {},
                            self.state,
                        )
                        _invoke_hook_safely(
                            getattr(self._hooks, "on_tool_complete", None),
                            result,
                            self.state,
                        )

                    # Step 10: INJECT
                    self._adapter.serialize_tool_results(results, chat)

                    # Update state messages reference
                    self._state.messages = chat.messages

                    # Step 11: LOOP
                    self._state.step += 1

            finally:
                self._lock.release()

        finally:
            self._release_lock()
            # Reset parent budget/state ContextVars (prevent cross-run contamination)
            # Use tokens captured at set() to restore previous values
            PARENT_BUDGET.reset(parent_budget_token)
            PARENT_STATE.reset(parent_state_token)

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
            )

        return response

    async def stream(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[list[dict[str, Any]]] = None,
        initial_chat: Optional[ModelChat] = None,
    ) -> AsyncIterator[ChatCompletionModel]:
        """Stream chunks from the LLM asynchronously, executing tools between iterations.

        Returns an AsyncIterator[ChatCompletionModel]. NO sync facade is provided.

        Args:
            user_input: The primary user message.
            system_prompt: Optional system prompt.
            extra_messages: Optional list of message dicts before user_input.
            initial_chat: Optional prebuilt chat to use directly.

        Yields:
            ChatCompletionModel chunks from the streaming LLM.

        Raises:
            RuntimeError: If called while the loop is already running.
            TypeError: If used with sync iteration (for in ...).
        """
        # INIT: Build initial chat
        chat = _build_initial_chat(
            user_input=user_input,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
            initial_chat=initial_chat,
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

        # Set parent budget/state ContextVars for nested LLM node execution
        # Child tasks can read these via PARENT_BUDGET.get() and PARENT_STATE.get()
        # Capture tokens for cleanup in finally block (prevent cross-run contamination)
        parent_budget_token = PARENT_BUDGET.set(self._budget)
        parent_state_token = PARENT_STATE.set(self._state)

        # Acquire concurrency guard
        self._acquire_lock()
        try:
            await self._lock.acquire()
            try:
                while True:
                    # Pre-call budget check
                    _check_budget(self._state, self._budget, include_tokens=False)

                    # Hook: on_iteration_start
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_iteration_start", None),
                        self._state.step,
                        self.state,
                    )

                    # Stream from LLM (async)
                    tool_defs = self._adapter.serialize_tool_defs(self._tools)
                    accumulated_tool_calls: dict[int, dict[str, Any]] = {}
                    iteration_content: list[str] = []
                    last_chunk: Optional[ChatCompletionModel] = None
                    response = None

                    async for chunk in self._client.llm.async_stream_generate(
                        chat,
                        tools=tool_defs,
                        tool_choice=self._tool_choice,
                        **self._generate_kwargs,
                    ):
                        # Accumulate tool calls from deltas
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.tool_calls:
                                for tc in delta.tool_calls:
                                    idx = tc.index if tc.index is not None else 0
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {}
                                    if tc.id:
                                        accumulated_tool_calls[idx]["id"] = tc.id
                                    if tc.function:
                                        if "function" not in accumulated_tool_calls[idx]:
                                            accumulated_tool_calls[idx]["function"] = {}
                                        if tc.function.name:
                                            accumulated_tool_calls[idx]["function"]["name"] = tc.function.name
                                        if tc.function.arguments:
                                            if "arguments" not in accumulated_tool_calls[idx]["function"]:
                                                accumulated_tool_calls[idx]["function"]["arguments"] = ""
                                            accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
                            if delta and delta.content:
                                iteration_content.append(delta.content)

                        yield chunk
                        last_chunk = chunk

                    # Build synthetic response for adapter methods
                    if last_chunk is not None:
                        from magic_llm.model.ModelChatResponse import (
                            ModelChatResponse, Choice, Message,
                        )

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
                                        content="".join(iteration_content) if iteration_content else None,
                                        tool_calls=None,
                                    ),
                                    finish_reason=(
                                        last_chunk.choices[0].finish_reason
                                        if last_chunk.choices
                                        else None
                                    ),
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

                        # Post-call budget check
                        _check_budget(self._state, self._budget, include_tokens=True)

                        # Hook: on_llm_response
                        _invoke_hook_safely(
                            getattr(self._hooks, "on_llm_response", None),
                            response,
                            self.state,
                        )

                        # Extract tool calls from accumulated data
                        tool_calls = []
                        for idx in sorted(accumulated_tool_calls.keys()):
                            tc_data = accumulated_tool_calls[idx]
                            func_data = tc_data.get("function", {})
                            try:
                                arguments = json.loads(func_data.get("arguments", "{}"))
                            except (json.JSONDecodeError, TypeError):
                                arguments = {}
                            tool_calls.append(
                                CanonicalToolCall(
                                    id=tc_data.get("id", ""),
                                    name=func_data.get("name", ""),
                                    arguments=arguments,
                                )
                            )

                        # Check done
                        if not tool_calls or self._adapter.is_finished(response):
                            break

                        # Record content
                        if iteration_content:
                            chat.add_assistant_message("".join(iteration_content))

                        # Add tool_call message
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
                                content="".join(iteration_content) if iteration_content else None,
                            )

                        # Validate integrity
                        self._adapter.validate_pair_integrity(chat)

                        # Execute tools (async)
                        results = await self._executor.execute_parallel_async(tool_calls)

                        # Hooks
                        for result in results:
                            _invoke_hook_safely(
                                getattr(self._hooks, "on_tool_start", None),
                                result.name,
                                result.tool_call_id or "",
                                {},
                                self.state,
                            )
                            _invoke_hook_safely(
                                getattr(self._hooks, "on_tool_complete", None),
                                result,
                                self.state,
                            )

                        # Inject results
                        self._adapter.serialize_tool_results(results, chat)
                        self._state.messages = chat.messages

                        # Yield separator between iterations
                        if iteration_content and self._content_separator and last_chunk:
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

            finally:
                self._lock.release()

        finally:
            self._release_lock()
            # Reset parent budget/state ContextVars (prevent cross-run contamination)
            # Use tokens captured at set() to restore previous values
            PARENT_BUDGET.reset(parent_budget_token)
            PARENT_STATE.reset(parent_state_token)
