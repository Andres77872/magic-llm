"""OpenRouter stream, usage and error formats (ported from ColWrite-api, where the adapter began)."""

from __future__ import annotations

import json

import pytest

from magic_llm.engine.openai_adapters.openai_openrouter import (
    OPENROUTER_BASE_URL,
    ProviderOpenRouter as OpenRouterProvider,
    error_from_payload,
    usage_from_payload,
)
from magic_llm.util.http import HttpError as OpenRouterError


def provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="k", model="anthropic/claude-sonnet-5.5")


def data(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def event(delta=None, finish=None, **extra) -> dict:
    return {
        "id": "gen-1",
        "created": 1,
        "model": "anthropic/claude-sonnet-5.5",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        **extra,
    }


class TestStream:
    def test_defaults_to_the_openrouter_endpoint_and_accepts_images(self):
        assert provider().base_url == OPENROUTER_BASE_URL
        assert OpenRouterProvider.supports_vision is True

    @pytest.mark.parametrize(
        "line",
        [": OPENROUTER PROCESSING", "", "   ", "event: ping", "data: [DONE]", "data:"],
    )
    def test_keep_alives_and_terminators_carry_nothing(self, line):
        assert provider().process_chunk(line) is None

    def test_content_and_reasoning_details_are_read(self):
        chunk = provider().process_chunk(
            data(
                event(
                    {
                        "content": "Hello",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "think", "index": 0}
                        ],
                    }
                )
            )
        )
        delta = chunk.choices[0].delta
        assert delta.content == "Hello"
        assert delta.reasoning_details == [
            {"type": "reasoning.text", "text": "think", "index": 0}
        ]
        assert chunk.model == "anthropic/claude-sonnet-5.5"

    def test_final_usage_chunk_keeps_cache_accounting_and_cost(self):
        chunk = provider().process_chunk(
            data(
                event(
                    {"content": ""},
                    "stop",
                    usage={
                        "prompt_tokens": 12_000,
                        "completion_tokens": 300,
                        "total_tokens": 12_300,
                        "cost": 0.0123,
                        "prompt_tokens_details": {
                            "cached_tokens": 11_000,
                            "cache_write_tokens": 800,
                        },
                        "completion_tokens_details": {"reasoning_tokens": 120},
                    },
                )
            )
        )
        usage = chunk.usage
        assert (usage.prompt_tokens, usage.completion_tokens) == (12_000, 300)
        assert usage.prompt_tokens_details.cached_tokens == 11_000
        assert usage.cached_tokens_write == 800
        assert usage.completion_tokens_details.reasoning_tokens == 120
        assert usage.provider_extra == {"cost": 0.0123}
        assert chunk.choices[0].finish_reason == "stop"

    def test_a_usage_frame_without_choices_does_not_replay_the_last_delta(self):
        adapter = provider()
        last = adapter.process_chunk(data(event({"content": "tail"}, "stop")))
        chunk = adapter.process_chunk(
            data({"id": "gen-1", "model": "m", "choices": [], "usage": {"prompt_tokens": 7}}),
            last_chunk=last,
        )
        assert chunk.usage.prompt_tokens == 7
        assert chunk.choices[0].delta.content == ""
        assert chunk.choices[0].finish_reason == "stop"

    def test_a_frame_with_neither_choices_nor_usage_is_skipped(self):
        assert provider().process_chunk(data({"id": "gen-1", "choices": []})) is None

    def test_a_mid_stream_error_fails_the_call(self):
        line = data(
            {
                "id": "gen-1",
                "error": {
                    "code": 502,
                    "message": "upstream said: PRIVATE DOCUMENT TEXT",
                    "metadata": {"error_type": "provider_unavailable"},
                },
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
            }
        )
        with pytest.raises(OpenRouterError) as raised:
            provider().process_chunk(line)
        assert raised.value.status_code == 502
        assert raised.value.error_type == "provider_unavailable"
        assert "PRIVATE DOCUMENT TEXT" not in str(raised.value)

    def test_an_error_finish_without_details_still_fails(self):
        with pytest.raises(OpenRouterError) as raised:
            provider().process_chunk(data(event({}, "error")))
        assert raised.value.status_code is None

    def test_malformed_events_raise_instead_of_ending_the_stream_quietly(self):
        with pytest.raises(ValueError):
            provider().process_chunk("data: {not json")
        with pytest.raises(ValueError):
            provider().process_chunk("data: [1, 2]")


class TestErrors:
    @pytest.mark.parametrize(
        ("error", "status", "error_type"),
        [
            ({"code": 429, "message": "slow down"}, 429, None),
            ({"code": "rate_limit_exceeded"}, 429, "rate_limit_exceeded"),
            ({"code": 500, "metadata": {"error_type": "timeout"}}, 500, "timeout"),
            ({"metadata": {"error_type": "provider_overloaded"}}, 503, "provider_overloaded"),
            ({"metadata": {"error_type": "context_length_exceeded"}}, 400, "context_length_exceeded"),
            ({"metadata": {"error_type": "authentication"}}, 401, "authentication"),
            # Only recognized types are kept: a body's strings never survive as such.
            ({"code": "something_new"}, None, None),
            ({"code": True}, None, None),
            ("boom", None, None),
            (None, None, None),
        ],
    )
    def test_status_and_type_come_from_the_typed_error(self, error, status, error_type):
        failure = error_from_payload(error)
        assert failure.status_code == status
        assert failure.error_type == error_type


class TestResponse:
    def test_usage_is_normalized(self):
        response = provider().transform_response(
            {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-6.1-sol",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2_000,
                    "completion_tokens": 5,
                    "cost": 0.001,
                    "prompt_tokens_details": {"cached_tokens": 1_500},
                },
            }
        )
        assert response.content == "ok"
        assert response.usage.prompt_tokens_details.cached_tokens == 1_500
        assert response.usage.total_tokens == 2_005
        assert response.usage.provider_extra == {"cost": 0.001}

    def test_an_error_body_under_http_200_raises(self):
        with pytest.raises(OpenRouterError) as raised:
            provider().transform_response({"error": {"code": 503, "message": "x"}})
        assert raised.value.status_code == 503

    def test_no_choices_is_a_retryable_failure(self):
        with pytest.raises(OpenRouterError) as raised:
            provider().transform_response({"id": "gen-1", "choices": []})
        assert raised.value.status_code is None
        assert raised.value.error_type == "no_content"

    def test_cost_with_your_own_provider_key_includes_the_provider_charge(self):
        # As returned for a BYOK account: OpenRouter's own fee is zero.
        usage = usage_from_payload(
            {
                "prompt_tokens": 8,
                "completion_tokens": 80,
                "cost": 0,
                "is_byok": True,
                "cost_details": {"upstream_inference_cost": 0.000612},
            }
        )
        assert usage.provider_extra == {"cost": 0.000612}

    def test_cost_on_openrouter_credits_is_not_counted_twice(self):
        usage = usage_from_payload(
            {"cost": 0.002, "cost_details": {"upstream_inference_cost": 0.002}}
        )
        assert usage.provider_extra == {"cost": 0.002}

    def test_usage_ignores_garbage(self):
        assert usage_from_payload(None) is None
        usage = usage_from_payload(
            {"prompt_tokens": "many", "cost": "free", "prompt_tokens_details": 4}
        )
        assert usage.prompt_tokens == 0
        assert usage.provider_extra is None
