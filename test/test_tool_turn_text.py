"""A tool-calling turn reaches the provider with the text the model wrote.

The agent loop keeps the text beside the tool calls in the tool-call
message (see the content-invariant tests in test_agent_loop.py). These tests
check that each wire format sends it back, before the calls, the way the
providers document an assistant turn.
"""

import json

from magic_llm.engine.engine_anthropic import EngineAnthropic
from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.model import ModelChat

CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}
NOTE = "Two sources agree; checking the third."


def _chat(content):
    chat = ModelChat()
    chat.add_user_message("Find it.")
    chat.add_tool_call_message(tool_calls=[CALL], content=content)
    return chat


def test_openai_chat_sends_text_with_tool_calls():
    engine = EngineOpenAI(api_key="test-only", model="gpt-4o")
    payload = json.loads(engine.transform_request(_chat(NOTE))[0])
    turn = payload["messages"][-1]
    assert turn["role"] == "assistant"
    assert turn["content"] == NOTE
    assert turn["tool_calls"][0]["id"] == "call_1"


def test_anthropic_sends_text_block_before_tool_use():
    engine = EngineAnthropic(api_key="test-key", model="claude-3-haiku-20240307")
    payload = json.loads(engine.prepare_data(_chat(NOTE))[0])
    blocks = payload["messages"][-1]["content"]
    assert blocks[0] == {"type": "text", "text": NOTE}
    assert blocks[1]["type"] == "tool_use" and blocks[1]["id"] == "call_1"


def test_anthropic_tool_turn_without_text_has_no_empty_text_block():
    engine = EngineAnthropic(api_key="test-key", model="claude-3-haiku-20240307")
    payload = json.loads(engine.prepare_data(_chat(None))[0])
    blocks = payload["messages"][-1]["content"]
    assert [block["type"] for block in blocks] == ["tool_use"]


def test_gemini_sends_text_part_before_function_call():
    engine = EngineGoogle(api_key="test-api-key", model="gemini-2.5-flash")
    payload = json.loads(engine.transform_request(_chat(NOTE))[0])
    parts = payload["contents"][-1]["parts"]
    assert parts[0] == {"text": NOTE}
    assert "functionCall" in parts[1]
