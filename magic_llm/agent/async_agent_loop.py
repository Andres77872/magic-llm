"""AsyncAgentLoop — asynchronous ReAct-style agent loop.

Mirrors the AgentLoop API but uses async def for run() and stream().
Uses asyncio.Lock for concurrency guard (not threading.Lock).

NOT safe for concurrent asyncio tasks. Each instance should be used by
a single asyncio task. Concurrent .run()/.stream() calls on the same
instance raise RuntimeError.
"""

from __future__ import annotations

from magic_llm.engine.attempt_control import ProviderAttemptControl, require_attempt_capability
from magic_llm.agent.control import (AgentLoopControl, AgentLoopCheckpoint, InboxMessage,
    AgentControlError, CheckpointBudget, utc_timestamp, parse_timestamp, require_complete_tool_history)
from magic_llm.agent.request import validate_agent_request, observer_tool_result

import asyncio
import hashlib
import json
import copy
import logging
import time

try:
    from asyncio import timeout as async_timeout
except ImportError:  # Python 3.10
    from async_timeout import timeout as async_timeout
from typing import Any, AsyncIterator, Callable, Optional

from magic_llm.agent import config as agent_config
from magic_llm.agent._loop_shared import (
    _build_initial_chat,
    _check_budget,
    _finalize_response,
    _invoke_hook_safely,
    _register_tools_with_executor,
    _initial_tool_call_ids,
    _reserve_tool_call_ids,
    # Parent context ContextVars for nested LLM node execution
    PARENT_BUDGET,
    PARENT_HOOKS,
    PARENT_STATE,
    PARENT_TODO_TOOLS,
)
from magic_llm.agent.builtin_tools import create_builtin_todo_bundle
from magic_llm.agent.hooks import AgentHooks
from magic_llm.agent.tool_adapters import ToolAdapter, ToolAdapterFactory
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentState,
    CanonicalToolCall,
    ToolResult,
)
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
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatResponse import (
    ModelChatResponse, Choice, Message,
)
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel,
    ChoiceModel,
    DeltaModel,
)

logger = logging.getLogger(__name__)


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
        adapter: Deprecated compatibility option retained for direct callers.
            Canonical tooling behavior does not use this adapter.
        tool_executor: An optional ToolExecutor instance. Created internally if None.
        deduplicate: Enable tool deduplication (default: False, opt-in).
        content_separator: String to join content between iterations (default: "\\n\\n").
        tool_choice: Tool choice parameter for the LLM (default: "auto").
        heartbeat_cb: Optional async callback invoked periodically (~8s) during
            tool execution to yield SSE keepalive events. Only active when set.
            Defaults to None (no heartbeat).
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
        heartbeat_cb: Optional[Callable[[], Any]] = None,
        prompt_fragment: str | Callable[..., str] | None = None,
        builtin_todo_tools: Optional[bool] = None,
        tool_executor_options: Optional[dict[str, Any]] = None,
        request_guard: Optional[Callable[..., Any]] = None,
        tool_result_observer: Optional[Callable[..., Any]] = None,
        provider_attempt_control: ProviderAttemptControl | None = None,
        control: AgentLoopControl | None = None,
        control_clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        if control is not None and provider_attempt_control is None:
            raise AgentControlError("Loop control requires provider attempt admission", "CONTROL_CAPABILITY_REQUIRED")
        self._control = control
        self._control_clock = control_clock or time.time
        self._last_checkpoint = None
        self._absolute_deadline = None
        self._client = client
        self._prompt_fragment = prompt_fragment
        self._request_guard = request_guard
        self._tool_result_observer = tool_result_observer
        self._provider_attempt_control = provider_attempt_control
        # Base system prompt WITHOUT fragment — used for per-iteration resolution
        self._base_system_prompt: Optional[str] = None

        # Store tools for registration at run time
        self._user_tools = list(tools or [])
        self._builtin_tool_functions: dict[str, Callable[..., Any]] = {}
        self._builtin_todo_enabled = (agent_config.is_builtin_todo_tools_enabled()
                                      if builtin_todo_tools is None else builtin_todo_tools)
        if self._builtin_todo_enabled:
            builtin_schemas, self._builtin_tool_functions = create_builtin_todo_bundle()
            self._tools = [*builtin_schemas, *self._user_tools]
        else:
            self._tools = self._user_tools
        self._tool_functions = tool_functions or {}

        # Budget defaults
        self._budget = budget if budget is not None else AgentBudget()
        self._configured_budget = copy.deepcopy(self._budget)
        self._hooks = hooks  # None = no-op (handled by _invoke_hook_safely)

        # Adapter: explicit override takes precedence over auto-detection
        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = ToolAdapterFactory.create_for_client(client)

        # Tool executor: explicit override or create internally with defaults
        if tool_executor is not None:
            self._executor = tool_executor
        else:
            self._executor = ToolExecutor(
                per_tool_timeout=30.0,
                enable_dedup=deduplicate,
            )

        if tool_executor_options is not None:
            self._executor = self._executor.with_options(tool_executor_options)

        self._deduplicate = deduplicate
        self._content_separator = content_separator
        self._tool_choice = tool_choice
        self._heartbeat_cb = heartbeat_cb
        self._provider = infer_provider_from_client(client)

        # Store kwargs for passthrough, but explicitly drop engine_type
        engine_type = kwargs.pop("engine_type", None)
        if engine_type is not None:
            logger.warning(
                "engine_type=%r is ignored; provider is inferred from client.llm",
                engine_type,
            )
        self._generate_kwargs = kwargs

        # Concurrency guard (asyncio.Lock, NOT threading.Lock)
        self._lock = asyncio.Lock()
        self._running = False

        # Internal state
        self._state = AgentState()

    @property
    def checkpoint(self) -> AgentLoopCheckpoint | None:
        """Last successfully committed safe checkpoint (detached private data)."""
        return self._last_checkpoint.detached() if self._last_checkpoint is not None else None

    def _controlled_chat(self, user_input, system_prompt, extra_messages, initial_chat, continuation):
        self._resume_checkpoint = None
        if continuation is None:
            return _build_initial_chat(user_input=user_input, system_prompt=system_prompt,
                extra_messages=extra_messages, initial_chat=initial_chat)
        if self._control is None:
            raise AgentControlError('Continuation requires an authoritative control port', 'INVALID_CONTINUATION')
        if user_input not in (None, '') or system_prompt is not None or extra_messages or initial_chat is not None:
            raise AgentControlError('Deliver new input through the continuation mailbox', 'INVALID_CONTINUATION')
        try:
            checkpoint = AgentLoopCheckpoint.model_validate(
                continuation.model_dump() if isinstance(continuation, AgentLoopCheckpoint) else continuation)
        except (ValueError, TypeError) as error:
            raise AgentControlError('Unsupported or invalid continuation checkpoint', 'INVALID_CONTINUATION') from error
        if checkpoint.provider != self._provider:
            raise AgentControlError('Continuation provider is incompatible', 'INVALID_CONTINUATION')
        if checkpoint.requires_context_guard and self._request_guard is None:
            raise AgentControlError('Continuation requires reconstructed context guards', 'INVALID_CONTINUATION')
        require_complete_tool_history(checkpoint.messages)
        self._resume_checkpoint = checkpoint.detached()
        chat = ModelChat(max_input_tokens=checkpoint.max_input_tokens,
                         extra_args=copy.deepcopy(checkpoint.chat_extra_args))
        chat.messages = copy.deepcopy(checkpoint.messages)
        chat.require_complete_context()
        return chat

    def _start_control(self, chat, seen_tool_call_ids):
        if self._control is None:
            return
        self._budget = copy.deepcopy(self._configured_budget)
        self._last_checkpoint = None
        self._message_digests = {}
        self._output_candidate = None
        self._started_at = self._control_clock()
        self._absolute_deadline = (self._started_at + self._budget.wall_clock_timeout
                                   if self._budget.wall_clock_timeout is not None else None)
        self._active_seen_tool_ids = seen_tool_call_ids
        chat.require_complete_context()
        require_complete_tool_history(chat.messages)
        self._executor._dedup_cache.clear()
        restored = self._resume_checkpoint
        if restored is not None:
            if (restored.tool_manifest_digest != self._manifest_digest()
                    or restored.builtin_todo_enabled != self._builtin_todo_enabled
                    or restored.deduplicate != self._executor._enable_dedup):
                raise AgentControlError('Continuation tool configuration changed', 'INVALID_CONTINUATION')
            if not set(seen_tool_call_ids).issubset(restored.seen_tool_call_ids):
                raise AgentControlError('Continuation lost historical tool IDs', 'INVALID_CONTINUATION')
            self._active_seen_tool_ids.update(restored.seen_tool_call_ids)
            self._message_digests = dict(restored.message_digests)
            self._started_at = parse_timestamp(restored.started_at)
            for name in ('max_iterations', 'max_input_tokens', 'max_output_tokens', 'wall_clock_timeout'):
                values = [value for value in (getattr(self._budget, name), getattr(restored.budget, name))
                          if value is not None]
                setattr(self._budget, name, min(values) if values else None)
            deadlines = [parse_timestamp(restored.absolute_deadline)] if restored.absolute_deadline else []
            if self._budget.wall_clock_timeout is not None:
                deadlines.append(self._started_at + self._budget.wall_clock_timeout)
            self._absolute_deadline = min(deadlines) if deadlines else None
            if self._absolute_deadline is not None:
                self._budget.wall_clock_timeout = self._absolute_deadline - self._started_at
                remaining = min(self._budget.wall_clock_timeout,
                                max(0.0, self._absolute_deadline - self._control_clock()))
                self._state.start_time = time.monotonic() - (self._budget.wall_clock_timeout - remaining)
            self._state.step = restored.step
            self._state.total_input_tokens = restored.total_input_tokens
            self._state.total_output_tokens = restored.total_output_tokens
            self._executor._dedup_cache = {key: ToolResult.model_validate(value)
                                          for key, value in restored.dedup_results.items()}
            if self._builtin_todo_enabled:
                self._builtin_tool_functions['todowrite'](todos=copy.deepcopy(restored.todos))
            self._base_system_prompt = copy.deepcopy(restored.base_system_prompt)
            self._output_candidate = restored.output_candidate
            self._last_checkpoint = restored.detached()
        self._control_deadline()

    def _manifest_digest(self):
        from magic_llm.engine.tooling import normalize_openai_tools
        return hashlib.sha256(json.dumps(
            {'tools': normalize_openai_tools(self._tools), 'tool_choice': self._tool_choice},
            sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(',', ':')).encode('utf-8')).hexdigest()

    def _control_deadline(self):
        if self._control is not None and self._absolute_deadline is not None:
            now = self._control_clock()
            if now >= self._absolute_deadline:
                raise AgentBudgetExceeded('wall_clock_timeout', self._budget.wall_clock_timeout,
                                          now - self._started_at)

    def _control_snapshot(self, chat):
        require_complete_tool_history(chat.messages)
        return AgentLoopCheckpoint(
            schema_version=1,
            provider=self._provider, tool_manifest_digest=self._manifest_digest(),
            messages=copy.deepcopy(chat.messages),
            seen_tool_call_ids=sorted(self._active_seen_tool_ids),
            consumed_message_ids=list(self._message_digests),
            message_digests=dict(self._message_digests), step=self._state.step,
            total_input_tokens=self._state.total_input_tokens,
            total_output_tokens=self._state.total_output_tokens,
            started_at=utc_timestamp(self._started_at),
            absolute_deadline=utc_timestamp(self._absolute_deadline) if self._absolute_deadline is not None else None,
            budget=CheckpointBudget(**vars(self._budget)),
            deduplicate=self._executor._enable_dedup,
            dedup_results={key: value.model_dump() for key, value in self._executor._dedup_cache.items()},
            builtin_todo_enabled=self._builtin_todo_enabled,
            todos=self._builtin_tool_functions['todoread']()['todos'] if self._builtin_todo_enabled else [],
            base_system_prompt=copy.deepcopy(self._base_system_prompt),
            max_input_tokens=chat.max_input_tokens, chat_extra_args=copy.deepcopy(chat.extra_args),
            requires_context_guard=(self._request_guard is not None or chat._provider_payload_guard is not None
                                    or chat._observer_projection is not None),
            output_candidate=self._output_candidate,
        )

    async def _save_control_checkpoint(self, chat, boundary):
        if self._control is None:
            return
        snapshot = self._control_snapshot(chat)
        # Hosts bound their persistence operations. Do not cancel a final safe
        # state commit merely because model compute's horizon just expired.
        await self._control.checkpoint(snapshot.detached(), boundary)
        self._last_checkpoint = snapshot

    async def _before_control_turn(self, chat):
        if self._control is None:
            return
        self._control_deadline()
        batch = await self._await_with_budget(self._control.before_turn(self._control_snapshot(chat)))
        if not isinstance(batch, list) or len(batch) > 32:
            raise AgentControlError('Control returned an invalid or oversized inbox batch')
        messages = [InboxMessage.model_validate(item) for item in batch]
        if sum(len(item.render().encode('utf-8')) for item in messages) > 65536:
            raise AgentControlError('Control inbox batch exceeds its byte bound')
        for item in messages:
            digest = item.digest()
            previous = self._message_digests.get(item.message_id)
            if previous is not None:
                if previous != digest:
                    raise AgentControlError('Immutable message ID has conflicting content', 'MESSAGE_CONFLICT')
                continue
            chat.add_user_message(item.render())
            self._message_digests[item.message_id] = digest
        self._state.messages = chat.messages
        await self._save_control_checkpoint(chat, 'input')
        self._control_deadline()

    async def _finish_control_candidate(self, chat, content):
        if self._control is None:
            return True
        self._state.step += 1
        self._state.messages = chat.messages
        self._output_candidate = content or ''
        await self._save_control_checkpoint(chat, 'candidate')
        decision = await self._await_with_budget(self._control.finish_candidate(self.checkpoint))
        if decision not in ('continue', 'candidate_ready'):
            raise AgentControlError('Control returned an invalid candidate decision')
        return decision == 'candidate_ready'

    def _attempt_options(self, method: Any) -> dict[str, Any]:
        if self._provider_attempt_control is None:
            return {}
        require_attempt_capability(method)
        return {'provider_attempt_control': self._provider_attempt_control}

    @property
    def state(self) -> AgentState:
        """Return a read-only copy of the current agent state.

        Mutations to the returned state do NOT affect internal loop state.
        """
        return AgentState(
            messages=copy.deepcopy(self._state.messages),
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

    async def run(self, user_input=None, system_prompt=None, extra_messages=None, initial_chat=None, *, continuation=None) -> ModelChatResponse:
        """Run exclusively; reject concurrent use before mutating state."""
        self._acquire_lock()
        try:
            return await self._run(user_input, system_prompt, extra_messages, initial_chat, continuation)
        finally:
            self._release_lock()

    async def stream(self, user_input=None, system_prompt=None, extra_messages=None, initial_chat=None, *, continuation=None) -> AsyncIterator[ChatCompletionModel]:
        """Stream exclusively and close the underlying stream on cancellation."""
        self._acquire_lock()
        source = self._stream(user_input, system_prompt, extra_messages, initial_chat, continuation)
        try:
            async for chunk in source:
                yield chunk
        finally:
            try:
                await source.aclose()
            finally:
                self._release_lock()

    async def _await_with_budget(self, awaitable):
        """Interrupt a stalled provider/tool await at the run's deadline."""
        timeout = self._budget.wall_clock_timeout
        if timeout is None:
            return await awaitable
        remaining = timeout - (time.monotonic() - self._state.start_time)
        if self._control is not None and self._absolute_deadline is not None:
            remaining = min(remaining, self._absolute_deadline - self._control_clock())
        deadline = async_timeout(max(0.0, remaining))
        try:
            async with deadline:
                return await awaitable
        except asyncio.TimeoutError:
            expired = deadline.expired
            if not (expired() if callable(expired) else expired):
                raise  # TimeoutError raised by the provider itself.
            exc = AgentBudgetExceeded("wall_clock_timeout", timeout, time.monotonic() - self._state.start_time)
            _invoke_hook_safely(getattr(self._hooks, "on_budget_exceeded", None),
                                exc.budget_type, str(exc), state=self.state)
            raise exc from None

    async def _stream_with_budget(self, source):
        try:
            while True:
                try:
                    chunk = await self._await_with_budget(source.__anext__())
                except StopAsyncIteration:
                    break
                yield chunk
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()

    async def _run(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[list[dict[str, Any]]] = None,
        initial_chat: Optional[ModelChat] = None,
        continuation: AgentLoopCheckpoint | dict[str, Any] | None = None,
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
        # Build initial chat — NO prompt_fragment prepended at init
        chat = self._controlled_chat(user_input, system_prompt, extra_messages, initial_chat, continuation)
        seen_tool_call_ids = _initial_tool_call_ids(chat)

        # Capture the base system prompt WITHOUT prompt_fragment from the chat.
        # This covers all cases: system_prompt param, initial_chat system,
        # or the merged combination of both (from _build_initial_chat).
        self._base_system_prompt = None
        for msg in chat.messages:
            if msg.get("role") == "system":
                self._base_system_prompt = msg["content"]
                break

        # Register tools
        if self._builtin_todo_enabled:
            _, self._builtin_tool_functions = create_builtin_todo_bundle()
        _register_tools_with_executor(
            self._executor,
            tools=self._user_tools,
            tool_functions=self._tool_functions,
            builtin_tool_functions=self._builtin_tool_functions,
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
        self._start_control(chat, seen_tool_call_ids)

        # Set parent context ContextVars for nested LLM node execution
        # Child tasks can read these via PARENT_BUDGET.get(), PARENT_STATE.get(),
        # and PARENT_HOOKS.get()
        # Capture tokens for cleanup in finally block (prevent cross-run contamination)
        parent_budget_token = PARENT_BUDGET.set(self._budget)
        parent_state_token = PARENT_STATE.set(self._state)
        parent_hooks_token = PARENT_HOOKS.set(self._hooks)
        parent_todo_token = PARENT_TODO_TOOLS.set(self._builtin_todo_enabled)

        collected_content: list[str] = []
        response: Optional[ModelChatResponse] = None

        # Acquire concurrency guard
        try:
            await self._lock.acquire()
            try:
                while True:
                    # Step 3: CHECK_BUDGET (pre-call: iterations + wall-clock)
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

                    await self._before_control_turn(chat)

                    # Hook: on_iteration_start
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_iteration_start", None),
                        self._state.step,
                        self.state,
                        state=self.state,
                    )

                    # Resolve prompt_fragment per-iteration and inject into system message.
                    # Prepend the resolved PF to the base system prompt so the agent
                    # sees fresh document context (e.g., updated doc JSON) each turn.
                    if self._prompt_fragment is not None:
                        pf = self._resolve_prompt_fragment(**self._generate_kwargs)
                        iteration_system = (
                            f"{pf}\n\n{self._base_system_prompt}".strip()
                            if self._base_system_prompt
                            else pf or ""
                        )
                        # Find or create system message
                        sys_idx = None
                        for i, msg in enumerate(chat.messages):
                            if msg.get("role") == "system":
                                sys_idx = i
                                break
                        if sys_idx is not None:
                            chat.messages[sys_idx]["content"] = iteration_system
                        else:
                            chat.messages.insert(
                                0, {"role": "system", "content": iteration_system}
                            )

                    # Step 2: LLM_CALL (async) — pass raw tools to engine/core tooling.
                    validate_agent_request(self._request_guard, chat=chat,
                        tools=self._tools, tool_choice=self._tool_choice,
                        provider=self._provider, client=self._client,
                        generation_options=self._generate_kwargs)
                    response = await self._await_with_budget(self._client.llm.async_generate(
                        chat,
                        tools=self._tools,
                        tool_choice=self._tool_choice,
                        **self._generate_kwargs,
                        **self._attempt_options(self._client.llm.async_generate),
                    ))

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
                    _reserve_tool_call_ids(tool_calls, seen_tool_call_ids)

                    # Step 6: RECORD_CONTENT — INVARIANT: suppress when tool_calls present
                    content = response.content
                    if not tool_calls and (content or self._control is not None):
                        if content:
                            collected_content.append(content)
                        chat.add_assistant_message(content or '', responses_output=response.responses_output, gemini_parts=response.gemini_parts)

                    # Step 7: CHECK_DONE — AFTER content recording (Phase 7)
                    # INVARIANT: Final content-only iteration must persist to state BEFORE break
                    if not tool_calls or (self._provider not in {"google", "gemini"} and is_finished(self._provider, response)):
                        if self._control is not None and tool_calls:
                            raise AgentControlError('Final response carries unresolved tool calls')
                        if await self._finish_control_candidate(chat, content):
                            break
                        collected_content.clear()
                        continue

                    # Step 8: ADD_TOOL_CALL — tool-call message (no speculative content)
                    if tool_calls:
                        tool_call_dicts = [
                            {
                                "id": tc.id,
                                "provider_metadata": tc.provider_metadata,
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
                            responses_output=response.responses_output,
                            gemini_parts=response.gemini_parts,
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

                    # Step 9: EXECUTE — run tools in parallel (async)
                    results = await self._await_with_budget(self._executor.execute_parallel_async(tool_calls))

                    # Hook: on_tool_complete — invoke AFTER execution for each result
                    for result in results:
                        _invoke_hook_safely(
                            getattr(self._hooks, "on_tool_complete", None),
                            observer_tool_result(self._tool_result_observer, result),
                            self.state,
                            state=self.state,
                        )

                    # Step 10: INJECT — engine/core builds provider-correct messages.
                    append_tool_results(self._provider, chat, results)

                    # Update state messages reference
                    self._state.messages = chat.messages

                    # Step 11: LOOP — checkpoint only after the complete batch.
                    self._state.step += 1
                    await self._save_control_checkpoint(chat, 'tool_results')

            finally:
                self._lock.release()

        finally:
            # Reset parent context ContextVars (prevent cross-run contamination)
            # Use tokens captured at set() to restore previous values
            PARENT_BUDGET.reset(parent_budget_token)
            PARENT_STATE.reset(parent_state_token)
            PARENT_HOOKS.reset(parent_hooks_token)
            PARENT_TODO_TOOLS.reset(parent_todo_token)

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

    async def _stream(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[list[dict[str, Any]]] = None,
        initial_chat: Optional[ModelChat] = None,
        continuation: AgentLoopCheckpoint | dict[str, Any] | None = None,
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
        # Build initial chat — NO prompt_fragment prepended at init
        chat = self._controlled_chat(user_input, system_prompt, extra_messages, initial_chat, continuation)
        seen_tool_call_ids = _initial_tool_call_ids(chat)

        # Capture the base system prompt WITHOUT prompt_fragment from the chat.
        # This covers all cases: system_prompt param, initial_chat system,
        # or the merged combination of both (from _build_initial_chat).
        self._base_system_prompt = None
        for msg in chat.messages:
            if msg.get("role") == "system":
                self._base_system_prompt = msg["content"]
                break

        # Register tools
        if self._builtin_todo_enabled:
            _, self._builtin_tool_functions = create_builtin_todo_bundle()
        _register_tools_with_executor(
            self._executor,
            tools=self._user_tools,
            tool_functions=self._tool_functions,
            builtin_tool_functions=self._builtin_tool_functions,
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
        self._start_control(chat, seen_tool_call_ids)

        # Accumulated content across all iterations (for on_loop_complete)
        collected_content: list[str] = []
        response: Optional[ModelChatResponse] = None

        # Set parent context ContextVars for nested LLM node execution
        # Child tasks can read these via PARENT_BUDGET.get(), PARENT_STATE.get(),
        # and PARENT_HOOKS.get()
        # Capture tokens for cleanup in finally block (prevent cross-run contamination)
        parent_budget_token = PARENT_BUDGET.set(self._budget)
        parent_state_token = PARENT_STATE.set(self._state)
        parent_hooks_token = PARENT_HOOKS.set(self._hooks)
        parent_todo_token = PARENT_TODO_TOOLS.set(self._builtin_todo_enabled)

        # Acquire concurrency guard
        try:
            await self._lock.acquire()
            # Track whether budget was exceeded — if so, skip on_loop_complete
            # in the finally block (budget-exceeded uses on_budget_exceeded instead)
            _budget_exceeded = False
            _completed = False
            try:
                while True:
                    # Pre-call budget check (iterations + wall-clock)
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

                    await self._before_control_turn(chat)

                    # Hook: on_iteration_start
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_iteration_start", None),
                        self._state.step,
                        self.state,
                        state=self.state,
                    )

                    # Resolve prompt_fragment per-iteration and inject into system message
                    if self._prompt_fragment is not None:
                        pf = self._resolve_prompt_fragment(**self._generate_kwargs)
                        iteration_system = (
                            f"{pf}\n\n{self._base_system_prompt}".strip()
                            if self._base_system_prompt
                            else pf or ""
                        )
                        # Update the system message (search for existing system msg)
                        sys_idx = None
                        for i, msg in enumerate(chat.messages):
                            if msg.get("role") == "system":
                                sys_idx = i
                                break
                        if sys_idx is not None:
                            chat.messages[sys_idx]["content"] = iteration_system
                        else:
                            chat.messages.insert(
                                0, {"role": "system", "content": iteration_system}
                            )

                    # Stream from LLM (async). Engine/provider chunks are already
                    # normalized; the loop accumulates via engine/core summary only.
                    summary = StreamIterationSummary()
                    last_chunk: Optional[ChatCompletionModel] = None

                    validate_agent_request(self._request_guard, chat=chat,
                        tools=self._tools, tool_choice=self._tool_choice,
                        provider=self._provider, client=self._client,
                        generation_options=self._generate_kwargs)
                    source = self._stream_with_budget(self._client.llm.async_stream_generate(
                        chat,
                        tools=self._tools,
                        tool_choice=self._tool_choice,
                        **self._generate_kwargs,
                        **self._attempt_options(self._client.llm.async_stream_generate),
                    ))
                    try:
                        async for chunk in source:
                            before_input = getattr(summary.usage, "prompt_tokens", 0) or 0
                            before_output = getattr(summary.usage, "completion_tokens", 0) or 0
                            accumulate_stream_chunk(summary, chunk)
                            self._state.total_input_tokens += (getattr(summary.usage, "prompt_tokens", 0) or 0) - before_input
                            self._state.total_output_tokens += (getattr(summary.usage, "completion_tokens", 0) or 0) - before_output
                            try:
                                _check_budget(self._state, self._budget, include_tokens=True)
                            except AgentBudgetExceeded as exc:
                                _budget_exceeded = True
                                _invoke_hook_safely(getattr(self._hooks, "on_budget_exceeded", None),
                                                    exc.budget_type, str(exc), state=self.state)
                                raise
                            last_chunk = chunk
                            yield chunk
                    finally:
                        await source.aclose()

                    # Build a synthetic response from normalized engine/core stream summary.
                    if last_chunk is not None:
                        response = ModelChatResponse(
                            id=last_chunk.id or "stream-synthetic",
                            object="chat.completion",
                            created=last_chunk.created or 0.0,
                            model=last_chunk.model,
                            responses_output=summary.responses_output,
                            gemini_parts=summary.gemini_parts,
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
                        _reserve_tool_call_ids(tool_calls, seen_tool_call_ids)

                        # Record content — runs for EVERY iteration (including final no-tool answer)
                        if not tool_calls and (summary.content or self._control is not None):
                            iter_content = summary.content or ''
                            chat.add_assistant_message(iter_content, responses_output=summary.responses_output, gemini_parts=summary.gemini_parts)
                            collected_content.append(iter_content)
                            self._state.messages = chat.messages  # State sync BEFORE break

                        # Check done — AFTER content recording
                        if not tool_calls or (self._provider not in {"google", "gemini"} and is_finished(self._provider, response)):
                            if self._control is not None and tool_calls:
                                raise AgentControlError('Final response carries unresolved tool calls')
                            if await self._finish_control_candidate(chat, summary.content):
                                _completed = True
                                break
                            collected_content.clear()
                            continue

                        # Add tool_call message (no speculative content)
                        if tool_calls:
                            tool_call_dicts = [
                                {
                                    "id": tc.id,
                                    "provider_metadata": tc.provider_metadata,
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
                                responses_output=response.responses_output,
                                gemini_parts=response.gemini_parts,
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

                        # Execute tools (async) with heartbeat support
                        heartbeat_task: Optional[asyncio.Task[None]] = None
                        try:
                            if self._heartbeat_cb is not None:
                                async def _heartbeat_loop() -> None:
                                    while True:
                                        await asyncio.sleep(8)
                                        await self._heartbeat_cb()  # type: ignore[misc]

                                heartbeat_task = asyncio.create_task(_heartbeat_loop())

                            results = await self._await_with_budget(self._executor.execute_parallel_async(tool_calls))

                            # INVARIANT: inject results immediately — even partial results are valid
                            append_tool_results(self._provider, chat, results)
                            self._state.messages = chat.messages
                        finally:
                            if heartbeat_task is not None:
                                heartbeat_task.cancel()
                                try:
                                    await heartbeat_task
                                except asyncio.CancelledError:
                                    pass

                        if self._control is not None:
                            self._state.step += 1
                            await self._save_control_checkpoint(chat, 'tool_results')

                        # Hook: on_tool_complete — invoke AFTER execution for each result
                        for result in results:
                            _invoke_hook_safely(
                                getattr(self._hooks, "on_tool_complete", None),
                                observer_tool_result(self._tool_result_observer, result),
                                self.state,
                                state=self.state,
                            )

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

                    if self._control is None:
                        self._state.step += 1
                    elif last_chunk is None:
                        raise AgentControlError('Provider returned an empty stream')

            finally:
                # Fire on_loop_complete with accumulated response
                # Runs for normal exit, generator close(), and exceptions.
                # Does NOT fire on budget-exceeded — that path uses on_budget_exceeded instead.
                if response is not None:
                    _finalize_response(response, collected_content, self._content_separator)
                if _completed and not _budget_exceeded:
                    _invoke_hook_safely(
                        getattr(self._hooks, "on_loop_complete", None),
                        response,
                        self.state,
                        state=self.state,
                    )
                self._lock.release()

        finally:
            # Reset parent context ContextVars (prevent cross-run contamination)
            # Use tokens captured at set() to restore previous values
            PARENT_BUDGET.reset(parent_budget_token)
            PARENT_STATE.reset(parent_state_token)
            PARENT_HOOKS.reset(parent_hooks_token)
            PARENT_TODO_TOOLS.reset(parent_todo_token)
