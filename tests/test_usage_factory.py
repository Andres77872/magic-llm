"""Provider-neutral usage accounting and sanitization contracts."""

import pytest

from magic_llm.engine._usage_factory import (
    attach_attempt_metadata,
    build_usage_model,
    estimated_usage_model,
    sanitize_provider_extra,
    usage_attempt_dict,
    usage_from_bedrock_invocation_metrics,
    usage_from_cohere_meta,
    usage_from_google_usage,
    usage_from_openai_payload,
    usage_has_tokens,
)


def test_openai_usage_preserves_details_and_derives_missing_total():
    usage = usage_from_openai_payload({
        "id": "chatcmpl_123",
        "service_tier": "default",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 4, "audio_tokens": 2},
            "completion_tokens_details": {
                "accepted_prediction_tokens": 3,
                "rejected_prediction_tokens": 2,
                "reasoning_tokens": 1,
                "audio_tokens": 6,
            },
        },
    })

    assert usage.total_tokens == 15
    assert usage.provider_request_id == "chatcmpl_123"
    assert usage.service_tier == "default"
    assert usage.prompt_tokens_details.cached_tokens == 4
    assert usage.prompt_tokens_details.audio_tokens == 2
    assert usage.completion_tokens_details.model_dump() == {
        "reasoning_tokens": 1,
        "audio_tokens": 6,
        "accepted_prediction_tokens": 3,
        "rejected_prediction_tokens": 2,
    }


def test_google_usage_preserves_cache_thought_tool_use_and_response_id():
    usage = usage_from_google_usage(
        {
            "promptTokenCount": 12,
            "candidatesTokenCount": 3,
            "totalTokenCount": 0,
            "cachedContentTokenCount": 4,
            "thoughtsTokenCount": 5,
            "toolUsePromptTokenCount": 6,
        },
        provider_request_id="gemini_response_123",
    )

    assert usage.total_tokens == 15
    assert usage.provider_request_id == "gemini_response_123"
    assert usage.prompt_tokens_details.cached_tokens == 4
    assert usage.completion_tokens_details.reasoning_tokens == 5
    assert usage.provider_extra == {"tool_use_prompt_tokens": 6}


@pytest.mark.parametrize(
    ("meta", "expected_basis"),
    [
        ({"tokens": {"input_tokens": 21, "output_tokens": 9}}, "tokens"),
        ({"billed_units": {"input_tokens": 21, "output_tokens": 9}}, "billed_units"),
    ],
)
def test_cohere_usage_records_which_accounting_basis_was_reported(meta, expected_basis):
    usage = usage_from_cohere_meta(meta, provider_request_id="generation_123")

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (21, 9, 30)
    assert usage.provider_request_id == "generation_123"
    assert usage.provider_extra == {"cohere_usage_basis": expected_basis}


def test_bedrock_usage_keeps_only_accounting_metadata():
    usage = usage_from_bedrock_invocation_metrics(
        {
            "inputTokenCount": 50,
            "outputTokenCount": 10,
            "invocationLatency": 123,
            "firstByteLatencyMs": 45,
            "prompt": "sensitive sentinel",
        },
        provider_request_id="bedrock_invocation_123",
    )

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (50, 10, 60)
    assert usage.provider_request_id == "bedrock_invocation_123"
    assert usage.provider_extra == {
        "invocation_latency_ms": 123,
        "first_byte_latency_ms": 45,
    }


def test_provider_extra_sanitizer_recurses_and_drops_secret_bearing_fields():
    unsafe_value = object()
    sanitized = sanitize_provider_extra({
        "safe_counter": 3,
        "api_key": "sensitive sentinel",
        "headers": {"Authorization": "sensitive sentinel"},
        "nested": {"safe_counter": 4, "client_secret": "sensitive sentinel"},
        "items": [
            {"safe_counter": 5, "authorization": "sensitive sentinel"},
            "safe-label",
            unsafe_value,
        ],
    })

    assert sanitized == {
        "safe_counter": 3,
        "nested": {"safe_counter": 4},
        "items": [{"safe_counter": 5}, "safe-label"],
    }


def test_estimated_usage_is_explicit_and_merges_safe_metadata():
    usage = estimated_usage_model(
        prompt_tokens=3,
        completion_tokens=4,
        provider_request_id="estimate_1",
        estimation_method="test_estimator",
        provider_extra={"provider_counter": 8, "raw_response": "sensitive sentinel"},
    )

    assert usage.total_tokens == 7
    assert usage.usage_source == "estimated"
    assert usage.provider_request_id == "estimate_1"
    assert usage.provider_extra == {
        "estimated": True,
        "estimation_method": "test_estimator",
        "provider_counter": 8,
    }


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, False),
        (build_usage_model(), False),
        (build_usage_model(cached_tokens_read=1), True),
        (build_usage_model(reasoning_tokens=1), True),
    ],
)
def test_usage_has_tokens_includes_cache_and_reasoning_details(usage, expected):
    assert usage_has_tokens(usage) is expected


def test_attempt_metadata_is_attached_without_mutating_original_usage():
    usage = build_usage_model(
        prompt_tokens=5,
        completion_tokens=2,
        provider_request_id="request-1",
        provider_extra={"safe_counter": 1},
    )
    attempt = usage_attempt_dict(usage, attempt_index=1, status="failed")

    attached = attach_attempt_metadata(usage, [attempt])

    assert usage.provider_extra == {"safe_counter": 1}
    assert attached is not usage
    assert attached.provider_extra == {"safe_counter": 1, "attempts": [attempt]}
