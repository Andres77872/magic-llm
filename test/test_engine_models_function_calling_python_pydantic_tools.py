import json

from magic_llm import MagicLLM

import pytest

from magic_llm.model import ModelChat

from conftest import get_provider_key

try:
    from pydantic import BaseModel  # v2
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

# Limit to a representative set of real providers we support end-to-end
TEST_PROVIDERS = [
    ("openai", "openai", "gpt-4o", "gpt-4o1"),
    ("anthropic", "anthropic", "claude-3-haiku-20240307", "FAIL/claude-3-haiku-20240307"),
    ("deepinfra", "deepinfra", "meta-llama/Meta-Llama-3.1-70B-Instruct", "microsoft/WizardLM-2-8x22B-model-fail"),
]

def _build_chat():
    c = ModelChat()
    c.add_user_message("Please check the weather for Bogotá and maybe use the tools if needed.")
    return c


def _keys_for(provider_keys, provider, key_name):
    return get_provider_key(provider_keys, provider, key_name)


def _python_tool_definitions():
    def get_weather(location: str):
        """
        Retrieve current weather information for a given location.

        This function fetches and returns weather details based on the
        specified location. The location is required and must be a valid
        string indicating the place for which the weather is queried.
        Ensure the location provided is recognized by the weather API used
        by the function.

        :param location: The name of the location to retrieve weather
            information for.
        :type location: str
        :return: Weather details for the specified location.
        :rtype: str
        """
        return ""

    tools = [get_weather]

    if BaseModel is not None:
        class GetForecast(BaseModel):
            """Forecast for a given location and days."""
            location: str
            days: int

        tools.append(GetForecast)

    return tools, "get_weather"


def _assert_tool_call_response(res, expected_function_name: str, allow_no_tool_call: bool = False):
    """
    Assert that the response has valid tool calls in the unified format.
    This validates provider-agnostic output consistency.

    Args:
        res: The ModelChatResponse to validate
        expected_function_name: The expected function name in tool_calls
        allow_no_tool_call: If True, skip test when model doesn't call tool (model behavior)
    """
    # Check if model called tools
    if res.tool_calls is None or len(res.tool_calls) == 0:
        if allow_no_tool_call:
            pytest.skip("Model did not call tools (model behavior, not a code bug)")
        else:
            pytest.fail("Response should have tool_calls but got None")

    # Validate first tool call structure (OpenAI-compatible format)
    tool_call = res.tool_calls[0]
    assert tool_call.type == "function", f"tool_call.type should be 'function', got '{tool_call.type}'"
    assert tool_call.id is not None, "tool_call.id should not be None"
    assert tool_call.function is not None, "tool_call.function should not be None"
    assert tool_call.function.name == expected_function_name, \
        f"Expected function name '{expected_function_name}', got '{tool_call.function.name}'"

    # Arguments should be a valid JSON string
    assert tool_call.function.arguments is not None, "tool_call.function.arguments should not be None"
    try:
        args = json.loads(tool_call.function.arguments)
        assert isinstance(args, dict), "Arguments should parse to a dict"
        # For get_weather, we expect 'location' key
        assert "location" in args, f"Arguments should contain 'location' key, got: {args}"
    except json.JSONDecodeError as e:
        pytest.fail(f"tool_call.function.arguments is not valid JSON: {e}")

    # finish_reason should be 'tool_calls' (unified format)
    # Note: Some providers may return 'stop' even when tools are called
    assert res.finish_reason in ("tool_calls", "stop"), \
        f"finish_reason should be 'tool_calls' or 'stop', got '{res.finish_reason}'"


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "_fail_model"),
    TEST_PROVIDERS,
    ids=[p[0] for p in TEST_PROVIDERS],
)
def test_python_and_pydantic_tools_at_init(provider_keys, provider, key_name, model, _fail_model):
    """Test that Python callable tools work at init time with unified output."""
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = _keys_for(provider_keys, provider, key_name)
    chat = _build_chat()

    # Pass tools at initialization time
    client = MagicLLM(model=model, tools=tools, tool_choice=tool_entry, **keys)
    res = client.llm.generate(chat)

    # Validate unified response format (allow skip for models that don't honor tool_choice)
    _assert_tool_call_response(res, primary_name, allow_no_tool_call=True)
    print(f"✓ {provider}: tool_calls validated successfully")


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "_fail_model"),
    TEST_PROVIDERS,
    ids=[p[0] for p in TEST_PROVIDERS],
)
def test_python_and_pydantic_tools_call_time_override(provider_keys, provider, key_name, model, _fail_model):
    """Test that tools can be overridden at call time with unified output."""
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = _keys_for(provider_keys, provider, key_name)
    chat = _build_chat()

    # Provide some defaults at init, override at call time
    client = MagicLLM(model=model, tools=tools[:1], tool_choice="auto", **keys)

    # Now override both tools and tool_choice at call time
    res = client.llm.generate(chat, tools=tools, tool_choice=tool_entry)

    # Validate unified response format (allow skip for models that don't honor tool_choice)
    _assert_tool_call_response(res, primary_name, allow_no_tool_call=True)
    print(f"✓ {provider}: call-time override validated successfully")


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "_fail_model"),
    TEST_PROVIDERS,
    ids=[p[0] for p in TEST_PROVIDERS],
)
def test_python_and_pydantic_tools_at_init_stream(provider_keys, provider, key_name, model, _fail_model):
    """Test that streaming with tools returns unified tool_calls in chunks."""
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = _keys_for(provider_keys, provider, key_name)
    chat = _build_chat()

    # Pass tools at initialization time
    client = MagicLLM(model=model, tools=tools, tool_choice=tool_entry, **keys)

    # Accumulate tool calls from stream chunks
    # Note: Some providers (Anthropic) send CUMULATIVE arguments in each chunk,
    # while others (OpenAI) send INCREMENTAL deltas. We handle both by:
    # - For providers with cumulative args: take the latest (longest) value
    # - For providers with incremental args: concatenate
    accumulated_tool_calls = {}  # id -> {name, arguments, is_cumulative}
    final_finish_reason = None

    for chunk in client.llm.stream_generate(chat):
        # Check for tool_calls in delta
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
            for tc in chunk.choices[0].delta.tool_calls:
                tc_id = tc.id or "default"
                if tc_id not in accumulated_tool_calls:
                    accumulated_tool_calls[tc_id] = {"name": None, "arguments": ""}

                if tc.function:
                    if tc.function.name:
                        accumulated_tool_calls[tc_id]["name"] = tc.function.name
                    if tc.function.arguments:
                        new_args = tc.function.arguments
                        current_args = accumulated_tool_calls[tc_id]["arguments"]
                        # Detect cumulative vs incremental:
                        # If new_args starts with '{' and current starts with '{',
                        # it's likely cumulative (Anthropic sends full JSON each time)
                        if new_args.startswith('{') and current_args.startswith('{'):
                            # Cumulative: take the longer/newer value
                            accumulated_tool_calls[tc_id]["arguments"] = new_args
                        else:
                            # Incremental: concatenate
                            accumulated_tool_calls[tc_id]["arguments"] += new_args

        # Capture finish_reason
        if chunk.choices and chunk.choices[0].finish_reason:
            final_finish_reason = chunk.choices[0].finish_reason

    # Validate accumulated tool calls (if any were received)
    if len(accumulated_tool_calls) == 0:
        # Some models may not stream tool calls even with tool_choice set
        pytest.skip(f"{provider}: Model did not stream tool_calls (model behavior, not a bug)")

    # Validate first accumulated tool call
    first_tc = list(accumulated_tool_calls.values())[0]
    assert first_tc["name"] == primary_name, \
        f"Expected function name '{primary_name}', got '{first_tc['name']}'"

    # Arguments should be valid JSON (if non-empty)
    if first_tc["arguments"]:
        try:
            args = json.loads(first_tc["arguments"])
            assert isinstance(args, dict), "Arguments should parse to a dict"
            assert "location" in args, f"Arguments should contain 'location' key, got: {args}"
        except json.JSONDecodeError as e:
            pytest.fail(f"Accumulated arguments is not valid JSON: '{first_tc['arguments']}' - {e}")

    # finish_reason should be 'tool_calls' (but some providers may use 'stop')
    assert final_finish_reason in ("tool_calls", "stop"), \
        f"finish_reason should be 'tool_calls' or 'stop', got '{final_finish_reason}'"

    print(f"✓ {provider}: streaming tool_calls validated successfully")
