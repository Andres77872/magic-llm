"""Tests for Phase 3: _loop_shared.py and AgentLoop.

TDD: react-like-agent-loop-refactor — Phase 3 (Slices 1-9, 14)

Covers:
- Slice 1: _check_budget (iteration, wall-clock, tokens)
- Slice 2: _build_initial_chat, _register_tools_with_executor, _finalize_response
- Slice 3: _invoke_hook_safely, _compute_fingerprint
- Slice 4: AgentLoop constructor, defaults, adapter auto-detect, engine_type rejection
- Slice 5: AgentLoop concurrency guard (threading.Lock)
- Slice 6: AgentLoop.run() happy path (no tools, single tool, parallel tools)
- Slice 7: AgentLoop.run() budget enforcement, error paths
- Slice 8: AgentLoop.run() deduplication, adapter override, state read-only
- Slice 9: AgentLoop.stream() chunk ordering, separator
- Slice 14: _loop_shared.py sync-only audit
"""

import hashlib
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from magic_llm import MagicLLM
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message, UsageModel
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
from magic_llm.agent.types import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentState,
    CanonicalToolCall,
)
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


# ─── Helpers ───────────────────────────────────────────────────────────────

def _make_response(content=None, tool_calls=None, finish_reason="stop",
                   prompt_tokens=10, completion_tokens=5):
    """Build a valid ModelChatResponse."""
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ModelChatResponse(
        id="test-1",
        object="chat.completion",
        created=1700000000.0,
        model="test-model",
        choices=[choice],
        usage=UsageModel(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_tool_call(id="call_1", name="get_weather",
                    arguments='{"city":"London"}'):
    """Build a valid ToolCall."""
    from magic_llm.model.ModelChatResponse import ToolCall, FunctionCall
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=arguments))


def _make_mock_adapter(is_finished=True, tool_calls=None):
    """Create a mock ToolAdapter."""
    adapter = MagicMock(spec=ToolAdapter)
    adapter.serialize_tool_defs.return_value = None
    adapter.deserialize_tool_calls.return_value = tool_calls or []
    adapter.is_finished.return_value = is_finished
    adapter.extract_final_text.return_value = ""
    adapter.validate_pair_integrity.return_value = True
    adapter.serialize_tool_results.return_value = None
    return adapter


# ─── Slice 1: _check_budget ────────────────────────────────────────────────

class TestCheckBudget:
    """_check_budget validates iteration, wall-clock, and token budgets."""

    def test_check_budget_iteration_limit_not_exceeded(self):
        state = AgentState(step=0)
        budget = AgentBudget(max_iterations=10)
        _check_budget(state, budget)  # no raise

    def test_check_budget_iteration_limit_exceeded(self):
        state = AgentState(step=10)
        budget = AgentBudget(max_iterations=10)
        with pytest.raises(AgentBudgetExceeded) as exc_info:
            _check_budget(state, budget)
        exc = exc_info.value
        assert exc.budget_type == "max_iterations"
        assert exc.limit == 10
        assert exc.current == 10
        assert "max_iterations" in str(exc)

    def test_check_budget_wall_clock_not_exceeded(self):
        state = AgentState(start_time=time.monotonic() - 1)
        budget = AgentBudget(wall_clock_timeout=5.0)
        _check_budget(state, budget)  # no raise

    def test_check_budget_wall_clock_exceeded(self):
        state = AgentState(start_time=time.monotonic() - 10)
        budget = AgentBudget(wall_clock_timeout=5.0)
        with pytest.raises(AgentBudgetExceeded) as exc_info:
            _check_budget(state, budget)
        exc = exc_info.value
        assert exc.budget_type == "wall_clock_timeout"
        assert exc.limit == 5.0

    def test_check_budget_output_tokens_not_exceeded(self):
        state = AgentState(total_output_tokens=500)
        budget = AgentBudget(max_output_tokens=1000)
        _check_budget(state, budget, include_tokens=True)  # no raise

    def test_check_budget_output_tokens_exceeded(self):
        state = AgentState(total_output_tokens=1500)
        budget = AgentBudget(max_output_tokens=1000)
        with pytest.raises(AgentBudgetExceeded) as exc_info:
            _check_budget(state, budget, include_tokens=True)
        exc = exc_info.value
        assert exc.budget_type == "max_output_tokens"
        assert exc.limit == 1000
        assert exc.current == 1500

    def test_check_budget_input_tokens_exceeded(self):
        state = AgentState(total_input_tokens=5000)
        budget = AgentBudget(max_input_tokens=4000)
        with pytest.raises(AgentBudgetExceeded) as exc_info:
            _check_budget(state, budget, include_tokens=True)
        exc = exc_info.value
        assert exc.budget_type == "max_input_tokens"

    def test_check_budget_all_none_limits_passes(self):
        state = AgentState(step=0)
        budget = AgentBudget(
            max_iterations=10,
            max_input_tokens=None,
            max_output_tokens=None,
            wall_clock_timeout=None,
        )
        _check_budget(state, budget, include_tokens=True)  # no raise

    def test_check_budget_tokens_not_checked_without_flag(self):
        """Token budgets should NOT be checked when include_tokens=False."""
        state = AgentState(
            total_input_tokens=99999,
            total_output_tokens=99999,
        )
        budget = AgentBudget(
            max_input_tokens=100,
            max_output_tokens=100,
        )
        # Should NOT raise because include_tokens=False
        _check_budget(state, budget, include_tokens=False)

    def test_check_budget_iteration_checked_even_without_tokens(self):
        state = AgentState(step=10)
        budget = AgentBudget(max_iterations=10)
        with pytest.raises(AgentBudgetExceeded) as exc_info:
            _check_budget(state, budget, include_tokens=False)
        assert exc_info.value.budget_type == "max_iterations"


# ─── Slice 2: _build_initial_chat, _register_tools_with_executor, _finalize_response

class TestBuildInitialChat:
    """_build_initial_chat creates correct ModelChat structure."""

    def test_build_initial_chat_with_system_and_user(self):
        chat = _build_initial_chat(
            user_input="hello",
            system_prompt="sys",
        )
        assert len(chat.messages) == 2
        assert chat.messages[0] == {"role": "system", "content": "sys"}
        assert chat.messages[1]["role"] == "user"

    def test_build_initial_chat_with_extra_messages(self):
        extra = [{"role": "user", "content": "pre"}]
        chat = _build_initial_chat(
            user_input="hello",
            system_prompt="sys",
            extra_messages=extra,
        )
        # system + extra + user = 3
        assert len(chat.messages) == 3

    def test_build_initial_chat_no_system_prompt(self):
        chat = _build_initial_chat(user_input="hello", system_prompt=None)
        assert len(chat.messages) == 1
        assert chat.messages[0]["role"] == "user"

    def test_build_initial_chat_with_initial_chat_and_system_prompt(self):
        """system_prompt is merged into initial_chat when both provided."""
        initial_chat = ModelChat(system="existing system")
        initial_chat.add_user_message("hello")
        chat = _build_initial_chat(
            initial_chat=initial_chat,
            system_prompt="additional guidance",
        )
        # Should have 2 messages (system + user)
        assert len(chat.messages) == 2
        assert chat.messages[0]["role"] == "system"
        # System prompt should be prepended to existing system content
        assert "additional guidance" in chat.messages[0]["content"]
        assert "existing system" in chat.messages[0]["content"]
        assert chat.messages[0]["content"].startswith("additional guidance")

    def test_build_initial_chat_with_initial_chat_and_no_system_prompt(self):
        """When no system_prompt, initial_chat is returned as-is (cloned)."""
        initial_chat = ModelChat(system="existing system")
        initial_chat.add_user_message("hello")
        chat = _build_initial_chat(
            initial_chat=initial_chat,
            system_prompt=None,
        )
        assert len(chat.messages) == 2
        assert chat.messages[0]["content"] == "existing system"

    def test_build_initial_chat_with_initial_chat_no_existing_system(self):
        """system_prompt is inserted at position 0 when initial_chat has no system message."""
        initial_chat = ModelChat()  # no system
        initial_chat.add_user_message("hello")
        chat = _build_initial_chat(
            initial_chat=initial_chat,
            system_prompt="new system prompt",
        )
        assert len(chat.messages) == 2
        assert chat.messages[0]["role"] == "system"
        assert chat.messages[0]["content"] == "new system prompt"
        assert chat.messages[1]["role"] == "user"

    def test_build_initial_chat_does_not_mutate_original(self):
        """Caller's initial_chat is not mutated."""
        initial_chat = ModelChat(system="original")
        original_messages = [dict(m) for m in initial_chat.messages]
        _build_initial_chat(
            initial_chat=initial_chat,
            system_prompt="new prompt",
        )
        assert initial_chat.messages == original_messages


class TestRegisterToolsWithExecutor:
    """_register_tools_with_executor handles both callables and dicts."""

    def test_register_tools_with_executor_callables(self):
        def fn1(): pass
        def fn2(): pass
        executor = ToolExecutor()
        _register_tools_with_executor(executor, tools=[fn1, fn2])
        assert "fn1" in executor._registry
        assert "fn2" in executor._registry

    def test_register_tools_with_executor_dict(self):
        def fn(): pass
        executor = ToolExecutor()
        _register_tools_with_executor(
            executor, tool_functions={"custom_name": fn}
        )
        assert "custom_name" in executor._registry

    def test_register_tools_with_executor_both_sources(self):
        def fn1(): pass
        def fn2(): pass
        executor = ToolExecutor()
        _register_tools_with_executor(
            executor,
            tools=[fn1],
            tool_functions={"custom": fn2},
        )
        assert "fn1" in executor._registry
        assert "custom" in executor._registry

    def test_register_tools_with_executor_dict_tool_spec(self):
        """Dict tool specs in tools list resolve callable from tool_functions."""
        tool_spec = {
            "type": "function",
            "function": {"name": "search_db", "description": "Search"},
        }
        def search_db(query): return {"results": []}
        executor = ToolExecutor()
        _register_tools_with_executor(
            executor,
            tools=[tool_spec],
            tool_functions={"search_db": search_db},
        )
        assert "search_db" in executor._registry
        assert executor._registry["search_db"] is search_db

    def test_register_tools_with_executor_dict_spec_without_tool_functions_skipped(self):
        """Dict tool spec with no matching tool_functions entry is silently skipped."""
        tool_spec = {
            "type": "function",
            "function": {"name": "search_db", "description": "Search"},
        }
        executor = ToolExecutor()
        _register_tools_with_executor(
            executor,
            tools=[tool_spec],
            tool_functions={},
        )
        assert "search_db" not in executor._registry

    def test_register_tools_with_executor_schema_name_mismatch_tool_functions_key(self):
        """Schema function.name MUST match tool_functions dict key for resolution."""
        tool_spec = {
            "type": "function",
            "function": {"name": "schema_tool_name", "description": "Search"},
        }
        def actual_callable(query):
            return {"results": []}
        executor = ToolExecutor()
        _register_tools_with_executor(
            executor,
            tools=[tool_spec],
            tool_functions={"different_key": actual_callable},
        )
        assert "schema_tool_name" not in executor._registry
        assert "different_key" in executor._registry


class TestFinalizeResponse:
    """_finalize_response concatenates content into the final response."""

    def test_finalize_response_with_content(self):
        response = _make_response(content="initial")
        collected = ["a", "b"]
        result = _finalize_response(response, collected, separator="\n\n")
        assert result.choices[0].message.content == "a\n\nb"

    def test_finalize_response_empty_content(self):
        response = _make_response(content="initial")
        result = _finalize_response(response, [])
        assert result.choices[0].message.content == "initial"  # unchanged

    def test_finalize_response_no_choices(self):
        response = ModelChatResponse(
            id="test", object="x", created=0.0, model="m", choices=[]
        )
        result = _finalize_response(response, ["a", "b"])
        # No crash, no mutation
        assert response.choices == []


# ─── Slice 3: _invoke_hook_safely, _compute_fingerprint

class TestInvokeHookSafely:
    """_invoke_hook_safely calls hooks and isolates exceptions (logs, does not propagate)."""

    def test_invoke_hook_safely_calls_method(self):
        calls = []
        def hook(a, b):
            calls.append((a, b))
        _invoke_hook_safely(hook, 1, 2)
        assert calls == [(1, 2)]

    def test_invoke_hook_safely_isolates_exception(self):
        """Hook exceptions are caught and logged, NOT propagated to caller."""
        def hook():
            raise ValueError("boom")
        # Exception should be logged but NOT raised
        _invoke_hook_safely(hook)  # no crash

    def test_invoke_hook_safely_noop_when_hook_is_none(self):
        _invoke_hook_safely(None, 1, 2, 3)  # no crash

    def test_invoke_hook_safely_logs_with_state_context(self):
        """Hook failures are logged with state context when provided."""
        from magic_llm.agent.types import AgentState
        
        def hook():
            raise ValueError("boom")
        
        state = AgentState(
            messages=[],
            step=5,
            start_time=0.0,
            total_input_tokens=100,
            total_output_tokens=50,
        )
        
        # Exception should be logged with state context but NOT raised
        _invoke_hook_safely(hook, state=state)  # no crash


class TestComputeFingerprint:
    """_compute_fingerprint produces deterministic SHA-256 hashes."""

    def test_compute_fingerprint_deterministic(self):
        fp1 = _compute_fingerprint("get_weather", {"city": "London"})
        fp2 = _compute_fingerprint("get_weather", {"city": "London"})
        assert fp1 == fp2

    def test_compute_fingerprint_different_args_different_hash(self):
        fp1 = _compute_fingerprint("get_weather", {"city": "London"})
        fp2 = _compute_fingerprint("get_weather", {"city": "Paris"})
        assert fp1 != fp2

    def test_compute_fingerprint_matches_spec(self):
        name = "get_weather"
        args = {"city": "London"}
        expected = hashlib.sha256(
            json.dumps({"name": name, "args": args}, sort_keys=True).encode()
        ).hexdigest()
        actual = _compute_fingerprint(name, args)
        assert actual == expected


# ─── Slice 4: AgentLoop constructor

class TestAgentLoopConstructor:
    """AgentLoop.__init__ accepts all spec parameters."""

    def test_agent_loop_constructor_with_defaults(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        with patch.object(
            ToolAdapterFactory, "create_for_client", return_value=adapter
        ):
            from magic_llm.agent.agent_loop import AgentLoop
            loop = AgentLoop(client, tools=[lambda: None])
        assert loop._adapter is adapter  # auto-detected
        assert loop._budget.max_iterations == 150
        assert loop._deduplicate is False

    def test_agent_loop_constructor_with_explicit_adapter(self):
        client = MagicMock()
        client.llm = MagicMock()
        custom_adapter = _make_mock_adapter()
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[lambda: None],
            adapter=custom_adapter,
        )
        assert loop._adapter is custom_adapter

    def test_agent_loop_constructor_with_explicit_executor(self):
        client = MagicMock()
        client.llm = MagicMock()
        custom_executor = ToolExecutor()
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[lambda: None],
            tool_executor=custom_executor,
        )
        assert loop._executor is custom_executor

    def test_agent_loop_constructor_with_deduplicate_true(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        with patch.object(
            ToolAdapterFactory, "create_for_client", return_value=adapter
        ):
            from magic_llm.agent.agent_loop import AgentLoop
            loop = AgentLoop(client, tools=[lambda: None], deduplicate=True)
        assert loop._executor._enable_dedup is True

    def test_agent_loop_constructor_ignores_engine_type(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        with patch.object(
            ToolAdapterFactory, "create_for_client", return_value=adapter
        ):
            from magic_llm.agent.agent_loop import AgentLoop
            loop = AgentLoop(
                client,
                tools=[lambda: None],
                engine_type="anthropic",
            )
        # No error raised, engine_type not in kwargs
        assert "engine_type" not in loop._generate_kwargs

    def test_agent_loop_constructor_stores_kwargs(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        with patch.object(
            ToolAdapterFactory, "create_for_client", return_value=adapter
        ):
            from magic_llm.agent.agent_loop import AgentLoop
            loop = AgentLoop(
                client,
                tools=[lambda: None],
                custom_kwarg="value",
            )
        assert loop._generate_kwargs.get("custom_kwarg") == "value"


# ─── Slice 5: AgentLoop concurrency guard

class TestAgentLoopConcurrencyGuard:
    """AgentLoop raises RuntimeError on concurrent access."""

    def test_concurrent_run_raises_runtime_error(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        # Manually set _running to True
        loop._running = True

        with pytest.raises(RuntimeError, match="already running"):
            loop.run("hello")

    def test_concurrent_stream_raises_runtime_error(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        loop._running = True

        with pytest.raises(RuntimeError, match="already running"):
            list(loop.stream("hello"))

    def test_lock_released_after_normal_completion(self):
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.generate.return_value = _make_response(
            content="done", finish_reason="stop"
        )
        adapter = _make_mock_adapter(is_finished=True)
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        loop.run("hello")
        assert loop._running is False

        # Subsequent run should succeed
        loop.run("hello again")
        assert loop._running is False

    def test_lock_released_after_exception(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter(is_finished=False, tool_calls=[])
        # Force budget exceeded on first iteration
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[],
            adapter=adapter,
            budget=AgentBudget(max_iterations=0),
        )

        with pytest.raises(AgentBudgetExceeded):
            loop.run("hello")

        # Lock should be released
        assert loop._running is False

        # Subsequent run with valid budget should work
        loop._budget = AgentBudget(max_iterations=10)
        client.llm.generate.return_value = _make_response(
            content="done", finish_reason="stop"
        )
        adapter.is_finished.return_value = True
        loop.run("hello")
        assert loop._running is False

    def test_lock_released_after_stream_completion(self):
        client = MagicMock()
        client.llm = MagicMock()

        # Create a single chunk with finish_reason="stop"
        chunk = ChatCompletionModel(
            id="chunk-1",
            model="test-model",
            choices=[
                ChoiceModel(
                    index=0,
                    delta=DeltaModel(content="hi"),
                    finish_reason="stop",
                )
            ],
        )
        client.llm.stream_generate.return_value = iter([chunk])

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        list(loop.stream("hello"))
        assert loop._running is False


# ─── Slice 6: AgentLoop.run() happy path

class TestAgentLoopRun:
    """AgentLoop.run() implements the ReAct state machine."""

    def test_run_no_tools_single_iteration(self):
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.generate.return_value = _make_response(
            content="hello world", finish_reason="stop"
        )
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        response = loop.run("hello")

        assert client.llm.generate.call_count == 1
        assert response.content == "hello world"

    def test_run_single_tool_call_then_done(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        # First call: tool_calls, second call: done
        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="final answer", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={"city": "London"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[get_weather], adapter=adapter)

        response = loop.run("What's the weather?")

        assert client.llm.generate.call_count == 2
        assert response.content == "final answer"

    def test_run_parallel_tool_calls(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc1 = _make_tool_call(id="call_1", name="tool_a")
        tc2 = _make_tool_call(id="call_2", name="tool_b")

        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc1, tc2], finish_reason="tool_calls"),
            _make_response(content="done", finish_reason="stop"),
        ]

        tool_calls = [
            CanonicalToolCall(id="call_1", name="tool_a", arguments={}),
            CanonicalToolCall(id="call_2", name="tool_b", arguments={}),
        ]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def tool_a(): return "a"
        def tool_b(): return "b"

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[tool_a, tool_b], adapter=adapter)

        response = loop.run("run both")

        assert client.llm.generate.call_count == 2
        assert response.content == "done"

    def test_run_adds_assistant_message_when_content_exists(self):
        """When response has content AND tool_calls, assistant message is added."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        client.llm.generate.side_effect = [
            _make_response(content="thinking...", tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="final", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={"city": "London"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[get_weather], adapter=adapter)

        loop.run("weather?")

        # Check that chat has assistant message (from iteration 0 with content)
        assistant_msgs = [
            m for m in loop.state.messages if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) >= 1

    def test_run_adds_tool_call_message_when_tool_calls_exist(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="final", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={"city": "London"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[get_weather], adapter=adapter)

        loop.run("weather?")

        # Check for tool_call message (has tool_calls key)
        has_tool_call_msg = any(
            msg.get("role") == "assistant" and "tool_calls" in msg
            for msg in loop.state.messages
        )
        assert has_tool_call_msg

    def test_run_content_suppressed_when_tool_calls_present(self):
        """INVARIANT: When response has both content AND tool_calls, content is
        suppressed — no add_assistant_message for that iteration, only
        add_tool_call_message(content=None). The content-invariant spec enforces
        that pre-tool speculative text is never persisted."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        client.llm.generate.side_effect = [
            _make_response(content="thinking...", tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="final", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={"city": "London"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[get_weather], adapter=adapter)

        response = loop.run("weather?")

        # Count assistant messages
        assistant_msgs = [
            m for m in loop.state.messages if m.get("role") == "assistant"
        ]
        # Phase 7: 2 assistant messages in chat for this turn:
        #  - [0]: tool-call message from iter-1 (content="thinking..." suppressed,
        #         tool_calls preserved)
        #  - [1]: final answer from iter-2 (content="final", no tool_calls)
        # Before Phase 7, the second message was never added because break
        # fired before content recording. Now content is recorded BEFORE break.
        assert len(assistant_msgs) == 2
        # The first assistant message is the tool-call message — content is None
        # (suppressed per invariant), tool_calls are preserved
        tool_call_msg = assistant_msgs[0]
        assert tool_call_msg["role"] == "assistant"
        assert "tool_calls" in tool_call_msg
        assert tool_call_msg.get("content") is None
        # The second assistant message is the final answer — content preserved,
        # no tool_calls
        final_msg = assistant_msgs[1]
        assert final_msg["role"] == "assistant"
        assert "tool_calls" not in final_msg
        assert final_msg.get("content") == "final"
        # The final output is also on the returned response
        assert response.content == "final"

    def test_sync_run_content_preserved_no_tool_calls(self):
        """Normal behavior: run() preserves content when no tool_calls
        present — content flows through the response AND is recorded
        in state via add_assistant_message before break (Phase 7 fix)."""
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.generate.return_value = _make_response(
            content="hello world", finish_reason="stop"
        )
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        response = loop.run("hello")

        # Normal behavior: content is preserved in the response
        assert response.content == "hello world"
        # Phase 7: content is recorded in state BEFORE break, so
        # state has 1 assistant message with the content
        assistant_msgs = [
            m for m in loop.state.messages if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("content") == "hello world"

    def test_sync_run_content_with_tool_calls_no_content_suppressed(self):
        """When tool_calls are present BUT content is None/empty,
        no content to suppress — add_tool_call_message is called
        normally."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()
        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="done", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={"city": "London"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[get_weather], adapter=adapter)

        response = loop.run("weather?")

        # Phase 7: 2 assistant messages — tool-call iter-1 + final answer iter-2
        assistant_msgs = [
            m for m in loop.state.messages if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 2
        tool_call_msg = assistant_msgs[0]
        assert tool_call_msg["role"] == "assistant"
        assert "tool_calls" in tool_call_msg
        # content was None originally, no suppression needed
        assert tool_call_msg.get("content") is None
        # Second assistant message is the final answer recorded before break
        final_msg = assistant_msgs[1]
        assert final_msg["role"] == "assistant"
        assert "tool_calls" not in final_msg
        assert final_msg.get("content") == "done"
        assert response.content == "done"

    def test_run_is_finished_takes_precedence_over_tool_calls(self):
        """is_finished=True exits regardless of tool_calls presence."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()
        client.llm.generate.return_value = _make_response(
            content="done", tool_calls=[tc], finish_reason="stop"
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.return_value = tool_calls
        adapter.is_finished.return_value = True  # finished despite tool_calls
        adapter.extract_final_text.return_value = "done"
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        response = loop.run("hello")

        # Should exit after 1 iteration (is_finished=True)
        assert client.llm.generate.call_count == 1
        assert loop.state.step == 0

    def test_magic_llm_run_agent_executes_python_callable_tool_canonically(self):
        """MagicLLM.run_agent() preserves callable auto-registration without legacy agentic."""

        class FakeOpenAILLM:
            engine = "openai"

            def __init__(self):
                self.calls = []

            def generate(self, chat, **kwargs):
                self.calls.append((list(chat.messages), kwargs))
                if len(self.calls) == 1:
                    return _make_response(
                        content=None,
                        tool_calls=[
                            _make_tool_call(
                                id="call_weather",
                                name="get_weather",
                                arguments='{"city":"Montevideo"}',
                            )
                        ],
                        finish_reason="tool_calls",
                    )
                return _make_response(content="It is 21C in Montevideo.", finish_reason="stop")

        def get_weather(city: str) -> dict:
            """Get the current weather for a city."""
            return {"city": city, "temperature_c": 21}

        client = MagicLLM.__new__(MagicLLM)
        client.llm = FakeOpenAILLM()

        response = client.run_agent(
            user_input="What's the weather in Montevideo?",
            tools=[get_weather],
            max_iterations=3,
        )

        assert response.content == "It is 21C in Montevideo."
        assert len(client.llm.calls) == 2

        first_kwargs = client.llm.calls[0][1]
        assert first_kwargs["tools"] == [get_weather]
        assert first_kwargs["tool_choice"] == "auto"

        second_messages = client.llm.calls[1][0]
        tool_messages = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_weather"
        assert "Montevideo" in tool_messages[0]["content"]
        assert not hasattr(MagicLLM, "agentic")
        assert not hasattr(MagicLLM, "agentic_stream")


# ─── Slice 7: AgentLoop.run() budget enforcement, error paths

class TestAgentLoopBudgetErrors:
    """AgentLoop.run() enforces budgets and handles errors."""

    def test_run_budget_exceeded_max_iterations(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()
        # Always returns tool_calls
        client.llm.generate.return_value = _make_response(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.return_value = tool_calls
        adapter.is_finished.return_value = False
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[lambda: None],
            adapter=adapter,
            budget=AgentBudget(max_iterations=2),
        )

        with pytest.raises(AgentBudgetExceeded) as exc_info:
            loop.run("hello")
        assert exc_info.value.budget_type == "max_iterations"
        # Should have run 2 iterations (step 0 and step 1), then fail on step 2
        assert exc_info.value.current == 2

    def test_run_budget_exceeded_wall_clock(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        def slow_generate(*args, **kwargs):
            time.sleep(0.2)
            return _make_response(
                content=None, tool_calls=[tc], finish_reason="tool_calls"
            )

        client.llm.generate.side_effect = slow_generate

        tool_calls = [CanonicalToolCall(id="call_1", name="slow_tool", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.return_value = tool_calls
        adapter.is_finished.return_value = False
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def slow_tool():
            time.sleep(0.3)
            return "done"

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[slow_tool],
            adapter=adapter,
            budget=AgentBudget(wall_clock_timeout=0.1),
        )

        with pytest.raises(AgentBudgetExceeded) as exc_info:
            loop.run("hello")
        assert exc_info.value.budget_type == "wall_clock_timeout"

    def test_run_unknown_tool_continues_loop(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="unknown_tool")

        # First: unknown tool, second: done
        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="done", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="unknown_tool", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        response = loop.run("hello")

        # Should have run 2 iterations (unknown tool → error result → done)
        assert client.llm.generate.call_count == 2
        assert response.content == "done"

    def test_run_hook_invocation_order(self):
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.generate.return_value = _make_response(
            content="done", finish_reason="stop"
        )
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])

        class RecordingHooks:
            def __init__(self):
                self.calls = []

            def on_iteration_start(self, iteration, state):
                self.calls.append(("on_iteration_start", iteration))

            def on_llm_response(self, response, state):
                self.calls.append(("on_llm_response",))

            def on_loop_complete(self, response, state):
                self.calls.append(("on_loop_complete",))

        hooks = RecordingHooks()

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter, hooks=hooks)

        loop.run("hello")

        assert hooks.calls == [
            ("on_iteration_start", 0),
            ("on_llm_response",),
            ("on_loop_complete",),
        ]


# ─── Slice 2b: tool_functions end-to-end in AgentLoop.run()

class TestAgentLoopToolFunctions:
    """AgentLoop.run() with tool_functions dict and mixed sources."""

    def test_run_with_tool_functions_dict(self):
        """tool_functions={"custom_name": fn} is called during loop execution."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="custom_search", arguments='{"q":"test"}')

        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="done", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="custom_search", arguments={"q": "test"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        call_log = []

        def custom_search(q):
            call_log.append(q)
            return {"results": [q]}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[],
            tool_functions={"custom_search": custom_search},
            adapter=adapter,
        )

        response = loop.run("search for test")

        assert client.llm.generate.call_count == 2
        assert call_log == ["test"]
        assert response.content == "done"

    def test_run_mixed_tools_and_tool_functions(self):
        """Both tools=[callable_a] and tool_functions={"custom_b": fn_b} are callable."""
        client = MagicMock()
        client.llm = MagicMock()
        tc_a = _make_tool_call(id="call_a", name="tool_a", arguments="{}")
        tc_b = _make_tool_call(id="call_b", name="custom_b", arguments='{"x": 1}')

        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc_a, tc_b], finish_reason="tool_calls"),
            _make_response(content="done", finish_reason="stop"),
        ]

        tool_calls = [
            CanonicalToolCall(id="call_a", name="tool_a", arguments={}),
            CanonicalToolCall(id="call_b", name="custom_b", arguments={"x": 1}),
        ]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        executed = []

        def tool_a():
            executed.append("tool_a")
            return "a"

        def custom_b(x):
            executed.append(("custom_b", x))
            return {"x": x}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[tool_a],
            tool_functions={"custom_b": custom_b},
            adapter=adapter,
        )

        response = loop.run("run both")

        assert client.llm.generate.call_count == 2
        assert "tool_a" in executed
        assert ("custom_b", 1) in executed
        assert response.content == "done"

    def test_run_with_dict_tool_spec_and_tool_functions(self):
        """Dict tool spec in tools list resolves callable from tool_functions."""
        # Realistic OpenAI-style tool definition
        tool_spec = {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }

        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="search_database", arguments='{"query":"climate"}')

        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="found results", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="search_database", arguments={"query": "climate"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        call_log = []

        def search_database(query):
            call_log.append(query)
            return {"results": [query]}

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[tool_spec],
            tool_functions={"search_database": search_database},
            adapter=adapter,
        )

        response = loop.run("search for climate")

        assert client.llm.generate.call_count == 2
        assert call_log == ["climate"]
        assert response.content == "found results"

    def test_run_tool_functions_override_callable_with_same_name(self):
        """tool_functions entry overrides callable with same __name__."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="get_data", arguments="{}")

        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="done", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="get_data", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_data():
            return "from_callable"

        def get_data_override():
            return "from_tool_functions"

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[get_data],
            tool_functions={"get_data": get_data_override},
            adapter=adapter,
        )

        response = loop.run("test")

        # The override should have been used (registered last, wins)
        assert response.content == "done"
        # Verify through engine/core-injected tool messages; canonical loop no
        # longer calls adapter.serialize_tool_results().
        tool_messages = [m for m in loop.state.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "from_tool_functions" in tool_messages[0]["content"]
        assert "from_callable" not in tool_messages[0]["content"]


# ─── Slice 8: AgentLoop.run() deduplication, adapter override, state read-only

class TestAgentLoopDedupAndState:
    """Deduplication, adapter override, and read-only state."""

    def test_run_deduplication_prevents_reexecution(self):
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1
            return call_count

        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="counter", arguments="{}")

        # First iteration: tool_call, second: done
        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="done", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="counter", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, tool_calls, []]
        adapter.is_finished.side_effect = [False, False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[counter],
            adapter=adapter,
            deduplicate=True,
        )

        loop.run("hello")

        # Tool should only be called once (second call is deduplicated)
        assert call_count == 1

    def test_run_deduplication_resets_between_runs(self):
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1
            return call_count

        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="counter", arguments="{}")

        # Each run: one tool call then done
        client.llm.generate.side_effect = [
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="done1", finish_reason="stop"),
            _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            _make_response(content="done2", finish_reason="stop"),
        ]

        tool_calls = [CanonicalToolCall(id="call_1", name="counter", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [
            tool_calls, [], tool_calls, []
        ]
        # is_finished is only called when tool_calls is non-empty (short-circuit otherwise)
        # Run 1 iter 0: tool_calls non-empty → is_finished called → False
        # Run 1 iter 1: tool_calls empty → is_finished NOT called (short-circuit)
        # Run 2 iter 0: tool_calls non-empty → is_finished called → False
        # Run 2 iter 1: tool_calls empty → is_finished NOT called (short-circuit)
        adapter.is_finished.side_effect = [False, False]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[counter],
            adapter=adapter,
            deduplicate=True,
        )

        loop.run("first")
        first_count = call_count

        loop.run("second")
        second_count = call_count

        # Tool should be called in both runs (dedup resets between runs)
        assert first_count == 1
        assert second_count == 2

    def test_run_explicit_adapter_takes_precedence(self):
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.generate.return_value = _make_response(
            content="done", finish_reason="stop"
        )
        custom_adapter = _make_mock_adapter(is_finished=True, tool_calls=[])

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=custom_adapter)

        loop.run("hello")

        # Adapter override is retained for compatibility/storage, but canonical
        # agent-loop tooling now goes through magic_llm.engine.tooling.
        assert loop._adapter is custom_adapter
        custom_adapter.serialize_tool_defs.assert_not_called()
        custom_adapter.deserialize_tool_calls.assert_not_called()
        custom_adapter.serialize_tool_results.assert_not_called()

    def test_state_property_returns_copy(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        state1 = loop.state
        state1.step = 999
        state2 = loop.state
        assert state2.step != 999

    def test_state_messages_are_copied(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        state = loop.state
        state.messages.append({"role": "user", "content": "intruder"})
        assert {"role": "user", "content": "intruder"} not in loop.state.messages

    def test_state_fingerprints_are_copied(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        state = loop.state
        state.executed_fingerprints.add("x")
        assert "x" not in loop.state.executed_fingerprints


# ─── Slice 9: AgentLoop.stream()

class TestAgentLoopStream:
    """AgentLoop.stream() yields chunks correctly."""

    def test_stream_yields_chunks_in_order(self):
        client = MagicMock()
        client.llm = MagicMock()

        chunks = [
            ChatCompletionModel(
                id=f"chunk-{i}",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content=f"c{i}"),
                        finish_reason=None,
                    )
                ],
            )
            for i in range(5)
        ]
        client.llm.stream_generate.return_value = iter(chunks)

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        result = list(loop.stream("hello"))

        assert len(result) == 5
        for i, chunk in enumerate(result):
            assert chunk.id == f"chunk-{i}"

    def test_stream_yields_separator_between_iterations(self):
        client = MagicMock()
        client.llm = MagicMock()

        # First iteration: tool call (finish_reason="tool_calls")
        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )
        chunks_iter0 = [
            ChatCompletionModel(
                id="chunk-0",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="thinking"),
                        finish_reason="tool_calls",
                    )
                ],
            )
        ]
        # Add tool_calls in delta
        chunks_iter0[0].choices[0].delta.tool_calls = [
            StreamToolCall(index=0, id="call_1",
                           function=StreamFunctionCall(name="get_weather", arguments="{}"))
        ]
        # Second iteration: done
        chunks_iter1 = [
            ChatCompletionModel(
                id="chunk-1",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="done"),
                        finish_reason="stop",
                    )
                ],
            )
        ]

        client.llm.stream_generate.side_effect = [iter(chunks_iter0), iter(chunks_iter1)]

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[lambda: None], adapter=adapter)

        result = list(loop.stream("hello"))

        # Should have: chunks from iter0 + separator + chunks from iter1
        # At minimum: 1 chunk + separator + 1 chunk = 3
        assert len(result) >= 3

        # Check separator is present (has "separator" in id)
        separator_chunks = [c for c in result if "separator" in c.id]
        assert len(separator_chunks) >= 1

    def test_stream_with_no_tools_yields_all_chunks(self):
        client = MagicMock()
        client.llm = MagicMock()

        chunks = [
            ChatCompletionModel(
                id=f"chunk-{i}",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content=f"c{i}"),
                        finish_reason="stop" if i == 4 else None,
                    )
                ],
            )
            for i in range(5)
        ]
        client.llm.stream_generate.return_value = iter(chunks)

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        result = list(loop.stream("hello"))

        assert len(result) == 5
        # No separator should be yielded (no tool calls)
        separator_chunks = [c for c in result if "separator" in c.id]
        assert len(separator_chunks) == 0

    def test_magic_llm_run_agent_stream_continues_after_tool_call_with_canonical_adapter(self):
        """MagicLLM.run_agent_stream() injects canonical tool results before continuing."""
        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )

        class FakeOpenAILLM:
            engine = "openai"

            def __init__(self):
                self.calls = []

            def stream_generate(self, chat, **kwargs):
                self.calls.append((list(chat.messages), kwargs))
                if len(self.calls) == 1:
                    chunk = ChatCompletionModel(
                        id="chunk-tool",
                        model="test-model",
                        choices=[
                            ChoiceModel(
                                index=0,
                                delta=DeltaModel(content=""),
                                finish_reason="tool_calls",
                            )
                        ],
                    )
                    chunk.choices[0].delta.tool_calls = [
                        StreamToolCall(
                            index=0,
                            id="call_weather",
                            function=StreamFunctionCall(
                                name="get_weather",
                                arguments='{"city":"Montevideo"}',
                            ),
                        )
                    ]
                    return iter([chunk])

                return iter([
                    ChatCompletionModel(
                        id="chunk-final",
                        model="test-model",
                        choices=[
                            ChoiceModel(
                                index=0,
                                delta=DeltaModel(content="Weather result used."),
                                finish_reason="stop",
                            )
                        ],
                    )
                ])

        def get_weather(city: str) -> dict:
            """Get weather for a city."""
            return {"city": city, "temperature_c": 21}

        client = MagicLLM.__new__(MagicLLM)
        client.llm = FakeOpenAILLM()

        chunks = list(client.run_agent_stream(
            user_input="Stream the weather.",
            tools=[get_weather],
            max_iterations=3,
        ))

        assert [chunk.id for chunk in chunks] == ["chunk-tool", "chunk-final"]
        assert chunks[-1].choices[0].delta.content == "Weather result used."
        assert len(client.llm.calls) == 2
        assert client.llm.calls[0][1]["tools"] == [get_weather]
        second_messages = client.llm.calls[1][0]
        tool_messages = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_weather"
        assert "Montevideo" in tool_messages[0]["content"]


# ─── Slice 9c: AgentLoop.stream() content invariant

class TestAgentLoopStreamInvariant:
    """AgentLoop.stream() enforces the content-invariant:
    content produced alongside tool_calls MUST be suppressed — never
    persisted to chat, never accumulated for final output, never passed
    as LLM context."""

    def test_sync_stream_content_suppressed_when_tool_calls(self):
        """INVARIANT: stream() suppresses content when tool_calls are
        present in the same iteration — no add_assistant_message, no
        collected_content.append, add_tool_call_message(content=None)."""
        client = MagicMock()
        client.llm = MagicMock()

        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )

        # First iteration: content + tool_calls → content suppressed
        chunks_iter0 = [
            ChatCompletionModel(
                id="chunk-0",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="thinking..."),
                        finish_reason="tool_calls",
                    )
                ],
            )
        ]
        chunks_iter0[0].choices[0].delta.tool_calls = [
            StreamToolCall(
                index=0, id="call_1",
                function=StreamFunctionCall(name="get_weather", arguments="{}"),
            )
        ]

        # Second iteration: done (no tool_calls)
        chunks_iter1 = [
            ChatCompletionModel(
                id="chunk-1",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="done"),
                        finish_reason="stop",
                    )
                ],
            )
        ]

        client.llm.stream_generate.side_effect = [
            iter(chunks_iter0), iter(chunks_iter1),
        ]

        tool_calls = [CanonicalToolCall(
            id="call_1", name="get_weather", arguments={},
        )]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[lambda: None], adapter=adapter)

        result = list(loop.stream("hello"))

        # INVARIANT: The pre-tool content "thinking..." is NOT leaked into any
        # assistant message. The tool-call assistant gets content=None.
        # Phase 7: the final content-only iteration ("done") IS recorded as a
        # second assistant message BEFORE the break exits the loop.
        assistant_msgs = [
            m for m in loop.state.messages if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 2, (
            f"Expected 2 assistant msgs (tool-call + final answer), got {len(assistant_msgs)}"
        )
        # First assistant: tool-call message — content suppressed
        tool_call_msg = assistant_msgs[0]
        assert tool_call_msg["role"] == "assistant"
        assert "tool_calls" in tool_call_msg
        assert tool_call_msg.get("content") is None
        # Second assistant: final answer — content recorded before break
        final_msg = assistant_msgs[1]
        assert final_msg["role"] == "assistant"
        assert "tool_calls" not in final_msg
        assert final_msg.get("content") == "done"
        # Verify "thinking..." is NOT leaked into any message content
        all_contents = [
            m.get("content", "") for m in loop.state.messages
            if m.get("content")
        ]
        assert all("thinking..." not in str(c) for c in all_contents)

    def test_sync_stream_content_preserved_no_tool_calls(self):
        """Normal behavior: stream() preserves content when no tool_calls
        present — content streams through chunks as expected."""
        client = MagicMock()
        client.llm = MagicMock()

        chunks = [
            ChatCompletionModel(
                id="chunk-0",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="hello world"),
                        finish_reason="stop",
                    )
                ],
            )
        ]
        client.llm.stream_generate.return_value = iter(chunks)

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(client, tools=[], adapter=adapter)

        result = list(loop.stream("hello"))

        # Content streams through chunks correctly
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "hello world"
        # No separator (no tool calls between iterations)
        separator_chunks = [c for c in result if "separator" in c.id]
        assert len(separator_chunks) == 0


# ─── Slice 9b: AgentLoop.stream() budget enforcement

class TestAgentLoopStreamBudget:
    """AgentLoop.stream() enforces budgets and fires on_budget_exceeded hook.

    Regression: bare _check_budget() calls in stream() were missing
    try/except AgentBudgetExceeded and the on_budget_exceeded hook,
    causing on_loop_complete to fire incorrectly on budget-exceeded exit.
    """

    def test_stream_budget_exceeded_max_iterations(self):
        """Budget exceeded in stream raises AgentBudgetExceeded with correct type."""
        client = MagicMock()
        client.llm = MagicMock()
        # Empty generator — never enters streaming body
        client.llm.stream_generate.return_value = iter([])

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[],
            adapter=adapter,
            budget=AgentBudget(max_iterations=0),
        )

        with pytest.raises(AgentBudgetExceeded) as exc_info:
            list(loop.stream("hello"))
        assert exc_info.value.budget_type == "max_iterations"
        assert exc_info.value.current == 0

    def test_stream_budget_exceeded_releases_lock(self):
        """Lock is released after budget-exceeded raises."""
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.stream_generate.return_value = iter([])

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[],
            adapter=adapter,
            budget=AgentBudget(max_iterations=0),
        )

        with pytest.raises(AgentBudgetExceeded):
            list(loop.stream("hello"))
        assert loop._running is False

        # Subsequent run with valid budget should work
        loop._budget = AgentBudget(max_iterations=10)
        chunk = ChatCompletionModel(
            id="chunk-1",
            model="test-model",
            choices=[
                ChoiceModel(
                    index=0,
                    delta=DeltaModel(content="hi"),
                    finish_reason="stop",
                )
            ],
        )
        client.llm.stream_generate.return_value = iter([chunk])
        list(loop.stream("hello"))
        assert loop._running is False

    def test_stream_budget_exceeded_does_not_fire_on_loop_complete(self):
        """on_loop_complete hook must NOT fire when budget is exceeded."""
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.stream_generate.return_value = iter([])

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        from magic_llm.agent.hooks import AgentHooks

        on_complete = MagicMock()
        on_budget_exceeded = MagicMock()

        # Create hooks with both methods tracked
        hooks = MagicMock(spec=AgentHooks)
        hooks.on_loop_complete = on_complete
        hooks.on_budget_exceeded = on_budget_exceeded

        loop = AgentLoop(
            client,
            tools=[],
            adapter=adapter,
            budget=AgentBudget(max_iterations=0),
            hooks=hooks,
        )

        with pytest.raises(AgentBudgetExceeded):
            list(loop.stream("hello"))

        # on_budget_exceeded MUST have been called
        on_budget_exceeded.assert_called_once()
        # on_loop_complete MUST NOT have been called
        on_complete.assert_not_called()

    def test_stream_budget_exceeded_wall_clock(self):
        """Wall-clock budget exceeded in stream raises AgentBudgetExceeded."""
        client = MagicMock()
        client.llm = MagicMock()

        # Simulate slow streaming that exceeds wall-clock
        def slow_stream(*args, **kwargs):
            time.sleep(0.2)
            return iter([])

        client.llm.stream_generate.side_effect = slow_stream

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.agent_loop import AgentLoop
        loop = AgentLoop(
            client,
            tools=[],
            adapter=adapter,
            budget=AgentBudget(wall_clock_timeout=0.05),
        )

        with pytest.raises(AgentBudgetExceeded) as exc_info:
            list(loop.stream("hello"))
        assert exc_info.value.budget_type == "wall_clock_timeout"


# ─── Slice 14: _loop_shared.py sync-only audit

class TestLoopSharedSyncOnly:
    """_loop_shared.py must NOT contain any async code."""

    def test_loop_shared_has_no_async_functions(self):
        import magic_llm.agent._loop_shared as mod
        import inspect

        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            assert not inspect.iscoroutinefunction(obj), (
                f"Function {name} is async — _loop_shared.py must be sync-only"
            )

    def test_loop_shared_has_no_asyncio_import(self):
        import magic_llm.agent._loop_shared as mod
        import inspect
        source = inspect.getsource(mod)
        assert "import asyncio" not in source
        assert "from asyncio" not in source

    def test_loop_shared_has_no_await_keywords(self):
        import magic_llm.agent._loop_shared as mod
        import inspect
        source = inspect.getsource(mod)
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "await " not in stripped, (
                f"Found 'await' in _loop_shared.py: {stripped}"
            )


# ─── Phase 1A: engine_type Warning Test ────────────────────────────────────


class TestAgentLoopEngineTypeWarning:
    """C5: engine_type warning tests for sync AgentLoop."""

    def test_engine_type_passed_logs_warning(self):
        """engine_type='anthropic' → logger.warning is called."""
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.agent_loop import AgentLoop

        with patch("magic_llm.agent.agent_loop.logger.warning") as mock_warning:
            AgentLoop(
                client,
                tools=[],
                adapter=adapter,
                engine_type="anthropic",
            )
            mock_warning.assert_called_once()
            args, _ = mock_warning.call_args
            warning_fmt = args[0] if args else ""
            # logger.warning receives format string + args separately
            assert "ignored" in warning_fmt
            assert "adapter" in warning_fmt or "client.llm" in warning_fmt
            # Check the engine_type value is passed as a formatting arg
            assert len(args) > 1 and args[1] == "anthropic"

    def test_engine_type_not_passed_no_warning(self):
        """No engine_type → no logger.warning."""
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.agent_loop import AgentLoop

        with patch("magic_llm.agent.agent_loop.logger.warning") as mock_warning:
            AgentLoop(client, tools=[], adapter=adapter)
            mock_warning.assert_not_called()

    def test_engine_type_not_in_generate_kwargs(self):
        """engine_type is popped and NOT stored in _generate_kwargs."""
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.agent_loop import AgentLoop

        loop = AgentLoop(
            client,
            tools=[],
            adapter=adapter,
            engine_type="anthropic",
        )
        assert "engine_type" not in loop._generate_kwargs
