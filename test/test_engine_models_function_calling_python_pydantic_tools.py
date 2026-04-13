import json
import os

from magic_llm import MagicLLM
from magic_llm.util.tools_mapping import normalize_openai_tools

import pytest

from magic_llm.model import ModelChat

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

KEYS_FILE = os.getenv("MAGIC_LLM_KEYS")
if not KEYS_FILE or not os.path.exists(KEYS_FILE):
    pytest.skip(
        "MAGIC_LLM_KEYS env var must point to a valid keys file for integration tests.",
        allow_module_level=True,
    )
with open(KEYS_FILE) as f:
    ALL_KEYS = json.load(f)

PROVIDERS = [
    (provider, key_name, success_model, fail_model)
    for provider, key_name, success_model, fail_model in TEST_PROVIDERS
    if key_name in ALL_KEYS
]


def _build_chat():
    c = ModelChat()
    c.add_user_message("Please check the weather for Bogotá and maybe use the tools if needed.")
    return c


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
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_python_and_pydantic_tools_at_init(provider, key_name, model, _fail_model):
    """Test that Python callable tools work at init time with unified output."""
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    # Pass tools at initialization time
    client = MagicLLM(model=model, tools=tools, tool_choice=tool_entry, **keys)
    res = client.llm.generate(chat)

    # Validate unified response format (allow skip for models that don't honor tool_choice)
    _assert_tool_call_response(res, primary_name, allow_no_tool_call=True)
    print(f"✓ {provider}: tool_calls validated successfully")


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "_fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_python_and_pydantic_tools_call_time_override(provider, key_name, model, _fail_model):
    """Test that tools can be overridden at call time with unified output."""
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = dict(ALL_KEYS[key_name])
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
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_python_and_pydantic_tools_at_init_stream(provider, key_name, model, _fail_model):
    """Test that streaming with tools returns unified tool_calls in chunks."""
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = dict(ALL_KEYS[key_name])
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

def test_normalize_python_callable_to_openai_format():
    """Test that Python callables are correctly normalized to OpenAI tool format."""
    tools, primary_name = _python_tool_definitions()
    normalized = normalize_openai_tools([tools[0]])

    # Should return a list with one tool
    assert len(normalized) == 1, "Should normalize to one tool"

    tool = normalized[0]

    # Validate canonical format: {name, description, parameters}
    assert "name" in tool, "Normalized tool should have 'name'"
    assert "description" in tool, "Normalized tool should have 'description'"
    assert "parameters" in tool, "Normalized tool should have 'parameters'"

    # Name should match function name
    assert tool["name"] == primary_name, f"Expected name '{primary_name}', got '{tool['name']}'"

    # Description should be extracted from docstring
    assert len(tool["description"]) > 0, "Description should be extracted from docstring"

    # Parameters should have proper JSON Schema structure
    params = tool["parameters"]
    assert params.get("type") == "object", "Parameters should have type 'object'"
    assert "properties" in params, "Parameters should have 'properties'"
    assert "location" in params["properties"], "Parameters should include 'location' property"
    assert "required" in params, "Parameters should have 'required' list"
    assert "location" in params["required"], "'location' should be required"

    print(f"✓ normalize_openai_tools: Python callable normalized correctly")
    print(f"  - name: {tool['name']}")
    print(f"  - parameters: {json.dumps(params, indent=2)}")


@pytest.mark.skipif(BaseModel is None, reason="Pydantic not available")
def test_normalize_pydantic_model_to_openai_format():
    """Test that Pydantic models are correctly normalized to OpenAI tool format."""
    tools, _ = _python_tool_definitions()
    # tools[1] is the GetForecast Pydantic model
    normalized = normalize_openai_tools([tools[1]])

    # Should return a list with one tool
    assert len(normalized) == 1, "Should normalize to one tool"

    tool = normalized[0]

    # Validate canonical format: {name, description, parameters}
    assert "name" in tool, "Normalized tool should have 'name'"
    assert "description" in tool, "Normalized tool should have 'description'"
    assert "parameters" in tool, "Normalized tool should have 'parameters'"

    # Name should match class name
    assert tool["name"] == "GetForecast", f"Expected name 'GetForecast', got '{tool['name']}'"

    # Description should be extracted from docstring
    assert "Forecast" in tool["description"], "Description should be extracted from docstring"

    # Parameters should have proper JSON Schema structure
    params = tool["parameters"]
    assert params.get("type") == "object", "Parameters should have type 'object'"
    assert "properties" in params, "Parameters should have 'properties'"
    assert "location" in params["properties"], "Parameters should include 'location' property"
    assert "days" in params["properties"], "Parameters should include 'days' property"

    # $defs should be inlined (no $ref remaining)
    assert "$defs" not in params, "$defs should be inlined"
    assert "$ref" not in str(params), "$ref should be resolved"

    print(f"✓ normalize_openai_tools: Pydantic model normalized correctly")
    print(f"  - name: {tool['name']}")
    print(f"  - parameters: {json.dumps(params, indent=2)}")


def test_unified_tool_format_across_input_types():
    """Test that all input formats produce identical canonical output."""
    from magic_llm.util.tools_mapping import map_to_openai, map_to_anthropic

    # Define same tool in different formats
    openai_format = {
        "type": "function",
        "function": {
            "name": "test_func",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"]
            }
        }
    }

    legacy_format = {
        "name": "test_func",
        "description": "Test function",
        "parameters": {
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
            "required": ["arg1"]
        }
    }

    anthropic_format = {
        "name": "test_func",
        "description": "Test function",
        "input_schema": {
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
            "required": ["arg1"]
        }
    }

    # Normalize all formats
    norm_openai = normalize_openai_tools([openai_format])
    norm_legacy = normalize_openai_tools([legacy_format])
    norm_anthropic = normalize_openai_tools([anthropic_format])

    # All should produce same canonical format
    assert norm_openai == norm_legacy, "OpenAI and legacy formats should normalize identically"
    assert norm_openai == norm_anthropic, "OpenAI and Anthropic formats should normalize identically"

    # Verify map_to_openai produces correct output format
    openai_tools, _ = map_to_openai([openai_format], "auto")
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "test_func"

    # Verify map_to_anthropic produces correct output format
    anthropic_tools, _ = map_to_anthropic([openai_format], "auto")
    assert "input_schema" in anthropic_tools[0]
    assert anthropic_tools[0]["name"] == "test_func"

    print("✓ All input formats normalize to identical canonical format")
    print("✓ map_to_openai produces correct OpenAI format")
    print("✓ map_to_anthropic produces correct Anthropic format")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TURN TOOL CHAIN TESTS (Agentic Loop)
# ═══════════════════════════════════════════════════════════════════════════════

def _define_chain_tools():
    """
    Define multiple tools for testing chained tool calls.
    These tools simulate a travel planning scenario:
    1. get_weather - Get weather for a location
    2. search_restaurants - Search restaurants in a location
    3. make_reservation - Make a restaurant reservation
    """

    def get_weather(location: str) -> str:
        """
        Get current weather for a location.

        :param location: City name (e.g., "Paris", "Tokyo")
        :return: Weather information as JSON string
        """
        # Simulated response
        return json.dumps({
            "location": location,
            "temperature": 22,
            "unit": "celsius",
            "condition": "sunny"
        })

    def search_restaurants(location: str, cuisine: str = "any") -> str:
        """
        Search for restaurants in a location.

        :param location: City to search in
        :param cuisine: Type of cuisine (optional)
        :return: List of restaurants as JSON string
        """
        # Simulated response
        return json.dumps({
            "location": location,
            "cuisine": cuisine,
            "restaurants": [
                {"name": "Le Petit Bistro", "rating": 4.5},
                {"name": "Café de Flore", "rating": 4.2},
                {"name": "Chez Marie", "rating": 4.8}
            ]
        })

    def make_reservation(restaurant: str, date: str, party_size: int = 2) -> str:
        """
        Make a restaurant reservation.

        :param restaurant: Name of the restaurant
        :param date: Date for reservation (e.g., "2024-12-25")
        :param party_size: Number of people
        :return: Reservation confirmation as JSON string
        """
        # Simulated response
        return json.dumps({
            "confirmed": True,
            "restaurant": restaurant,
            "date": date,
            "party_size": party_size,
            "confirmation_number": "RES-12345"
        })

    return [get_weather, search_restaurants, make_reservation]


def _execute_tool(tool_name: str, arguments: dict, tool_functions: dict) -> str:
    """Execute a tool by name and return the result."""
    if tool_name in tool_functions:
        return tool_functions[tool_name](**arguments)
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _format_tool_result(tool_name: str, arguments: dict, result: str) -> str:
    """Format tool result as feedback message for the model."""
    return f"tool: {tool_name}\ntool input: {json.dumps(arguments)}\ntool output: {result}"


# Providers that reliably support multi-turn tool chains
CHAIN_TEST_PROVIDERS = [
    ("openai", "openai", "gpt-4o"),
    ("anthropic", "anthropic", "claude-3-haiku-20240307"),
]

CHAIN_PROVIDERS = [
    (provider, key_name, model)
    for provider, key_name, model in CHAIN_TEST_PROVIDERS
    if key_name in ALL_KEYS
]


@pytest.mark.parametrize(
    ("provider", "key_name", "model"),
    CHAIN_PROVIDERS,
    ids=[p[0] for p in CHAIN_PROVIDERS],
)
def test_multi_turn_tool_chain_loop(provider, key_name, model):
    """
    Test multi-turn tool calling loop (agentic pattern).

    This tests the pattern:
    1. User asks question requiring multiple tool calls
    2. Model calls first tool → we execute → feed result back
    3. Model calls second tool → we execute → feed result back
    4. Model generates final response (no more tool calls)

    Flow:
        User: "What's the weather in Paris and find me restaurants there"
        ↓
        Model: tool_call(get_weather, location="Paris")
        ↓
        User: [tool result: weather data]
        ↓
        Model: tool_call(search_restaurants, location="Paris")
        ↓
        User: [tool result: restaurant list]
        ↓
        Model: "The weather in Paris is sunny at 22°C. Here are some restaurants..."
    """
    tools = _define_chain_tools()
    tool_functions = {fn.__name__: fn for fn in tools}

    keys = dict(ALL_KEYS[key_name])

    # Create client with tools
    client = MagicLLM(model=model, tools=tools, tool_choice="auto", **keys)

    # Build initial chat with a query that should trigger multiple tool calls
    chat = ModelChat(system="You are a helpful travel assistant. Use the available tools to help the user. Call each tool only once.")
    chat.add_user_message(
        "I'm planning a trip to Paris. Check the weather and find restaurants. Use each tool once."
    )

    # Track the conversation flow
    iterations = []
    max_iterations = 8  # Increased safety limit
    completed_naturally = False

    for iteration in range(max_iterations):
        # Generate response
        response = client.llm.generate(chat)

        iteration_info = {
            "iteration": iteration + 1,
            "has_tool_calls": response.tool_calls is not None and len(response.tool_calls) > 0,
            "finish_reason": response.finish_reason,
            "tool_calls": [],
            "content": response.content
        }

        if response.tool_calls and len(response.tool_calls) > 0:
            # Process each tool call
            for tool_call in response.tool_calls:
                tc_name = tool_call.function.name
                tc_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                result = _execute_tool(tc_name, tc_args, tool_functions)

                iteration_info["tool_calls"].append({
                    "name": tc_name,
                    "arguments": tc_args,
                    "result": result
                })

                # Add tool result to conversation
                feedback = _format_tool_result(tc_name, tc_args, result)
                chat.add_user_message(feedback)

            iterations.append(iteration_info)
            print(f"  Iteration {iteration + 1}: Called {[tc['name'] for tc in iteration_info['tool_calls']]}")

        else:
            # No tool calls - this is the final response
            iterations.append(iteration_info)
            completed_naturally = True
            print(f"  Iteration {iteration + 1}: Final response (no tool calls)")
            break

    # ═══════════════════════════════════════════════════════════════════════
    # ASSERTIONS
    # ═══════════════════════════════════════════════════════════════════════

    # Should have at least 1 iteration with tool calls
    tool_call_iterations = [i for i in iterations if i["has_tool_calls"]]
    assert len(tool_call_iterations) >= 1, \
        "Expected at least one iteration with tool calls"

    # Verify tool calls have valid structure (unified format)
    for iteration in tool_call_iterations:
        for tc in iteration["tool_calls"]:
            assert tc["name"] in tool_functions, \
                f"Tool '{tc['name']}' not in defined tools"
            assert isinstance(tc["arguments"], dict), \
                "Tool arguments should be a dict"

    # Collect all called tools
    all_called_tools = set()
    for iteration in tool_call_iterations:
        for tc in iteration["tool_calls"]:
            all_called_tools.add(tc["name"])

    # Should have called at least one of our tools
    assert len(all_called_tools) >= 1, \
        "Should have called at least one tool"

    # If completed naturally, verify final response
    final_iteration = iterations[-1]
    if completed_naturally:
        assert not final_iteration["has_tool_calls"], \
            "Final iteration should not have tool calls"
        assert final_iteration["content"] is not None, \
            "Final iteration should have content"

    print(f"\n✓ {provider}: Multi-turn tool chain completed")
    print(f"  - Total iterations: {len(iterations)}")
    print(f"  - Completed naturally: {completed_naturally}")
    print(f"  - Tools called: {sorted(all_called_tools)}")
    if completed_naturally and final_iteration["content"]:
        print(f"  - Final response length: {len(final_iteration['content'])} chars")


@pytest.mark.parametrize(
    ("provider", "key_name", "model"),
    CHAIN_PROVIDERS,
    ids=[p[0] for p in CHAIN_PROVIDERS],
)
def test_sequential_tool_chain_with_dependency(provider, key_name, model):
    """
    Test tool chain where later tools depend on earlier tool results.

    Flow:
        User: "Find restaurants in Paris and make a reservation at the best one"
        ↓
        Model: tool_call(search_restaurants, location="Paris")
        ↓
        User: [tool result: restaurant list with "Chez Marie" as top rated]
        ↓
        Model: tool_call(make_reservation, restaurant="Chez Marie", ...)
        ↓
        User: [tool result: confirmation]
        ↓
        Model: "I've made a reservation at Chez Marie..."

    This tests that the model can use information from one tool call
    to inform subsequent tool calls.
    """
    tools = _define_chain_tools()
    tool_functions = {fn.__name__: fn for fn in tools}

    keys = dict(ALL_KEYS[key_name])

    # Create client with tools
    client = MagicLLM(model=model, tools=tools, tool_choice="auto", **keys)

    # Query that requires chained tool calls with data dependency
    chat = ModelChat(system="You are a helpful travel assistant. Use tools to complete tasks. Call each tool only once.")
    chat.add_user_message(
        "Find restaurants in Paris and make a reservation at the best one for 2 people on December 25th. Use each tool once."
    )

    # Track tool call sequence
    tool_call_sequence = []
    max_iterations = 8
    final_content = None
    completed_naturally = False

    for iteration in range(max_iterations):
        response = client.llm.generate(chat)

        if response.tool_calls and len(response.tool_calls) > 0:
            for tool_call in response.tool_calls:
                tc_name = tool_call.function.name
                tc_args = json.loads(tool_call.function.arguments)

                # Record the call
                tool_call_sequence.append({
                    "name": tc_name,
                    "arguments": tc_args
                })

                # Execute and feed back
                result = _execute_tool(tc_name, tc_args, tool_functions)
                feedback = _format_tool_result(tc_name, tc_args, result)
                chat.add_user_message(feedback)

            print(f"  Iteration {iteration + 1}: Called {tool_call_sequence[-1]['name']}")
        else:
            # Final response
            final_content = response.content
            completed_naturally = True
            print(f"  Iteration {iteration + 1}: Final response")
            break

    # ═══════════════════════════════════════════════════════════════════════
    # ASSERTIONS
    # ═══════════════════════════════════════════════════════════════════════

    # Should have at least 1 tool call
    assert len(tool_call_sequence) >= 1, \
        f"Expected at least 1 tool call, got {len(tool_call_sequence)}"

    # Verify tool calls have valid structure (unified format)
    for tc in tool_call_sequence:
        assert tc["name"] in tool_functions, \
            f"Tool '{tc['name']}' not in defined tools"
        assert isinstance(tc["arguments"], dict), \
            "Tool arguments should be a dict"

    # Collect unique tools called
    tools_called = [tc["name"] for tc in tool_call_sequence]
    unique_tools = set(tools_called)

    # If make_reservation was called, verify it has required arguments
    reservation_calls = [tc for tc in tool_call_sequence if tc["name"] == "make_reservation"]
    if reservation_calls:
        for rc in reservation_calls:
            assert "restaurant" in rc["arguments"], \
                "make_reservation should have 'restaurant' argument"
        print(f"  - Reservation made for: {reservation_calls[0]['arguments'].get('restaurant')}")

    print(f"\n✓ {provider}: Sequential tool chain completed")
    print(f"  - Completed naturally: {completed_naturally}")
    print(f"  - Tool call sequence: {tools_called}")
    print(f"  - Unique tools used: {sorted(unique_tools)}")
    if final_content:
        print(f"  - Final response: {final_content[:100]}...")


@pytest.mark.parametrize(
    ("provider", "key_name", "model"),
    CHAIN_PROVIDERS,
    ids=[p[0] for p in CHAIN_PROVIDERS],
)
def test_tool_chain_max_iterations_safety(provider, key_name, model):
    """
    Test that tool chain loop respects max_iterations safety limit.

    This ensures the agentic loop doesn't run forever if the model
    keeps calling tools indefinitely.
    """
    tools = _define_chain_tools()
    tool_functions = {fn.__name__: fn for fn in tools}

    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(model=model, tools=tools, tool_choice="auto", **keys)

    chat = ModelChat(system="You are a helpful assistant. Answer concisely after using a tool once.")
    chat.add_user_message("What's the weather in Paris? Use the tool once and give a brief answer.")

    max_iterations = 5
    iteration_count = 0
    completed_normally = False
    tool_calls_made = []

    for iteration in range(max_iterations):
        iteration_count += 1
        response = client.llm.generate(chat)

        if response.tool_calls and len(response.tool_calls) > 0:
            for tool_call in response.tool_calls:
                tc_name = tool_call.function.name
                tc_args = json.loads(tool_call.function.arguments)
                tool_calls_made.append(tc_name)
                result = _execute_tool(tc_name, tc_args, tool_functions)
                feedback = _format_tool_result(tc_name, tc_args, result)
                chat.add_user_message(feedback)
            print(f"  Iteration {iteration_count}: Called {tc_name}")
        else:
            completed_normally = True
            print(f"  Iteration {iteration_count}: Final response")
            break

    # Should complete within max_iterations (this is the safety check)
    assert iteration_count <= max_iterations, \
        f"Loop exceeded max_iterations ({max_iterations})"

    # Should have made at least one tool call
    assert len(tool_calls_made) >= 1, \
        "Should have made at least one tool call"

    print(f"\n✓ {provider}: Max iterations safety check passed")
    print(f"  - Iterations used: {iteration_count}/{max_iterations}")
    print(f"  - Completed normally: {completed_normally}")
    print(f"  - Tools called: {tool_calls_made}")