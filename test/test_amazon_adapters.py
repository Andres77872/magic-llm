"""Tests for Amazon Bedrock adapter transform methods.

Classical TDD — feed known JSON payloads, verify ModelChatResponse/ChatCompletionModel output.
No network calls, no mocks of HTTP — pure input→output verification.
"""

import pytest

from magic_llm.engine.amazon_adapters.amazon_nova import ProviderAmazonNova
from magic_llm.engine.amazon_adapters.amazon_titan import ProviderAmazonTitan
from magic_llm.engine.amazon_adapters.amazon_anthropic import ProviderAmazonAnthropic
from magic_llm.engine.amazon_adapters.amazon_meta import ProviderAmazonMeta
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_chat(messages=None):
    chat = ModelChat()
    if messages:
        for msg in messages:
            chat.add_user_message(msg)
    else:
        chat.add_user_message("Hello")
    return chat


# ═══════════════════════════════════════════════════════════════════════════
# Slice 23 — ProviderAmazonNova transform tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAmazonNovaTransformResponse:
    """ProviderAmazonNova.transform_response — realistic Nova payloads."""

    def setup_method(self):
        self.adapter = ProviderAmazonNova(model="amazon.nova-lite-v1", api_key="test")

    def test_basic_text_response(self):
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello, world!"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "totalTokens": 15
            }
        }

        result = self.adapter.transform_response(raw)

        assert isinstance(result, ModelChatResponse)
        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_multi_content_response(self):
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "First part. "},
                        {"text": "Second part."}
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30}
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].message.content == "First part. Second part."

    def test_max_tokens_stop_reason(self):
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "truncated"}]
                }
            },
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 5, "outputTokens": 100, "totalTokens": 105}
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].finish_reason == "length"

    def test_tool_use_stop_reason(self):
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": ""}]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].finish_reason == "tool_calls"

    def test_empty_content(self):
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": []
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 0, "totalTokens": 5}
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].message.content is None


class TestAmazonNovaTransformStreamChunk:
    """ProviderAmazonNova.transform_stream_chunk — streaming events."""

    def setup_method(self):
        self.adapter = ProviderAmazonNova(model="amazon.nova-lite-v1", api_key="test")

    def test_content_block_delta(self):
        event = {
            "contentBlockDelta": {
                "delta": {"text": "Hello"}
            },
            "index": 0
        }

        result = self.adapter.transform_stream_chunk(event)

        assert isinstance(result, ChatCompletionModel)
        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].finish_reason is None

    def test_message_stop_event(self):
        event = {
            "messageStop": {
                "stopReason": "end_turn"
            }
        }

        result = self.adapter.transform_stream_chunk(event)

        assert result.choices[0].delta.content is None
        assert result.choices[0].finish_reason == "stop"

    def test_metadata_usage(self):
        event = {
            "contentBlockDelta": {"delta": {"text": "x"}},
            "metadata": {
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 50
                }
            }
        }

        result = self.adapter.transform_stream_chunk(event)

        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150


class TestAmazonNovaTransformRequest:
    """ProviderAmazonNova.transform_request — input transformation."""

    def setup_method(self):
        self.adapter = ProviderAmazonNova(model="amazon.nova-lite-v1", api_key="test")

    def test_string_content_converted_to_array(self):
        chat = _make_chat(["Hello"])
        result = self.adapter.transform_request(chat)

        import json
        body = json.loads(result)
        assert body["messages"][0]["content"] == [{"text": "Hello"}]

    def test_inference_config_defaults(self):
        chat = _make_chat(["test"])
        result = self.adapter.transform_request(chat)

        import json
        body = json.loads(result)
        assert body["inferenceConfig"]["max_new_tokens"] == 4096
        assert body["inferenceConfig"]["temperature"] == 1
        assert body["inferenceConfig"]["topP"] == 1

    def test_custom_inference_config(self):
        chat = _make_chat(["test"])
        result = self.adapter.transform_request(chat, max_new_tokens=512, temperature=0.7)

        import json
        body = json.loads(result)
        assert body["inferenceConfig"]["max_new_tokens"] == 512
        assert body["inferenceConfig"]["temperature"] == 0.7


# ═══════════════════════════════════════════════════════════════════════════
# Slice 23 — ProviderAmazonTitan transform tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAmazonTitanTransformResponse:
    """ProviderAmazonTitan.transform_response — realistic Titan payloads."""

    def setup_method(self):
        self.adapter = ProviderAmazonTitan(model="amazon.titan-text-lite-v1", api_key="test")

    def test_basic_response(self):
        raw = {
            "inputTextTokenCount": 10,
            "results": [{
                "tokenCount": 5,
                "outputText": "The answer is 42.",
                "completionReason": "FINISH"
            }]
        }

        result = self.adapter.transform_response(raw)

        assert isinstance(result, ModelChatResponse)
        assert result.choices[0].message.content == "The answer is 42."
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.choices[0].finish_reason == "stop"

    def test_max_tokens_reason(self):
        raw = {
            "inputTextTokenCount": 10,
            "results": [{
                "tokenCount": 100,
                "outputText": "truncated text",
                "completionReason": "LENGTH"
            }]
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].finish_reason == "length"


class TestAmazonTitanTransformStreamChunk:
    """ProviderAmazonTitan.transform_stream_chunk — streaming events."""

    def setup_method(self):
        self.adapter = ProviderAmazonTitan(model="amazon.titan-text-lite-v1", api_key="test")

    def test_text_chunk(self):
        event = {
            "outputText": "Hello",
            "index": 0
        }

        result = self.adapter.transform_stream_chunk(event)

        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].finish_reason is None

    def test_completion_event(self):
        event = {
            "outputText": "",
            "completionReason": "FINISH"
        }

        result = self.adapter.transform_stream_chunk(event)

        assert result.choices[0].finish_reason == "stop"


class TestAmazonTitanTransformRequest:
    """ProviderAmazonTitan.transform_request — input transformation."""

    def setup_method(self):
        self.adapter = ProviderAmazonTitan(model="amazon.titan-text-lite-v1", api_key="test")

    def test_uses_generic_chat_titan_format(self):
        chat = _make_chat(["Hello"])
        result = self.adapter.transform_request(chat)

        import json
        body = json.loads(result)
        assert "inputText" in body
        assert body["textGenerationConfig"]["maxTokenCount"] == 4096
        assert body["textGenerationConfig"]["temperature"] == 0
        assert body["textGenerationConfig"]["stopSequences"] == ["User:"]


# ═══════════════════════════════════════════════════════════════════════════
# Slice 23 — ProviderAmazonAnthropic transform tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAmazonAnthropicTransformResponse:
    """ProviderAmazonAnthropic.transform_response — realistic Claude Bedrock payloads."""

    def setup_method(self):
        self.adapter = ProviderAmazonAnthropic(model="anthropic.claude-v2", api_key="test")

    def test_basic_response(self):
        raw = {
            "completion": "Hello, I'm Claude.",
            "stop_reason": "stop_sequence",
            "prompt": "\n\nHuman: Hi\n\nAssistant:"
        }

        result = self.adapter.transform_response(raw)

        assert isinstance(result, ModelChatResponse)
        assert result.choices[0].message.content == "Hello, I'm Claude."
        assert result.choices[0].finish_reason == "stop"
        # Usage is approximated from character count
        assert result.usage.completion_tokens == len("Hello, I'm Claude.")

    def test_max_tokens_reason(self):
        raw = {
            "completion": "truncated",
            "stop_reason": "max_tokens",
            "prompt": "test"
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].finish_reason == "length"

    def test_end_turn_reason(self):
        raw = {
            "completion": "done",
            "stop_reason": "end_turn",
            "prompt": "test"
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].finish_reason == "stop"


class TestAmazonAnthropicTransformStreamChunk:
    """ProviderAmazonAnthropic.transform_stream_chunk — streaming events."""

    def setup_method(self):
        self.adapter = ProviderAmazonAnthropic(model="anthropic.claude-v2", api_key="test")

    def test_text_chunk(self):
        event = {"completion": "Hello"}

        result = self.adapter.transform_stream_chunk(event)

        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].finish_reason is None

    def test_stop_event(self):
        event = {"completion": "", "stop_reason": "stop_sequence"}

        result = self.adapter.transform_stream_chunk(event)

        assert result.choices[0].finish_reason == "stop"


class TestAmazonAnthropicTransformRequest:
    """ProviderAmazonAnthropic.transform_request — input transformation."""

    def setup_method(self):
        self.adapter = ProviderAmazonAnthropic(model="anthropic.claude-v2", api_key="test")

    def test_uses_claude_prompt_format(self):
        chat = _make_chat(["Hello"])
        result = self.adapter.transform_request(chat)

        import json
        body = json.loads(result)
        assert "prompt" in body
        assert body["max_tokens_to_sample"] == 1024
        assert body["temperature"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Slice 23 — ProviderAmazonMeta transform tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAmazonMetaTransformResponse:
    """ProviderAmazonMeta.transform_response — realistic Llama Bedrock payloads."""

    def setup_method(self):
        self.adapter = ProviderAmazonMeta(model="meta.llama2-13b-chat-v1", api_key="test")

    def test_basic_response(self):
        raw = {
            "generation": "Hello, I'm Llama.",
            "prompt_token_count": 10,
            "generation_token_count": 5
        }

        result = self.adapter.transform_response(raw)

        assert isinstance(result, ModelChatResponse)
        assert result.choices[0].message.content == "Hello, I'm Llama."
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_stop_reason_sets_finish_reason(self):
        raw = {
            "generation": "done",
            "prompt_token_count": 5,
            "generation_token_count": 3,
            "stop_reason": "stop"
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].finish_reason == "stop"

    def test_no_stop_reason(self):
        raw = {
            "generation": "incomplete",
            "prompt_token_count": 5,
            "generation_token_count": 3
        }

        result = self.adapter.transform_response(raw)

        assert result.choices[0].finish_reason is None


class TestAmazonMetaTransformStreamChunk:
    """ProviderAmazonMeta.transform_stream_chunk — streaming events."""

    def setup_method(self):
        self.adapter = ProviderAmazonMeta(model="meta.llama2-13b-chat-v1", api_key="test")

    def test_generation_chunk(self):
        event = {"generation": "Hello"}

        result = self.adapter.transform_stream_chunk(event)

        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].finish_reason is None

    def test_stop_event(self):
        event = {"generation": "", "stop_reason": "stop"}

        result = self.adapter.transform_stream_chunk(event)

        assert result.choices[0].finish_reason == "stop"


class TestAmazonMetaTransformRequest:
    """ProviderAmazonMeta.transform_request — input transformation."""

    def setup_method(self):
        self.adapter = ProviderAmazonMeta(model="meta.llama2-13b-chat-v1", api_key="test")

    def test_uses_llama2_prompt_format(self):
        chat = _make_chat(["Hello"])
        result = self.adapter.transform_request(chat)

        import json
        body = json.loads(result)
        assert "prompt" in body
        assert body["max_gen_len"] == 1024
        assert body["temperature"] == 0.2
