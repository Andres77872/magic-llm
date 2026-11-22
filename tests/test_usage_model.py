"""Public model contracts for additive provider-usage accounting fields."""

from magic_llm.engine._usage_factory import build_usage_model
from magic_llm.model.ModelChatResponse import Choice, Message, ModelChatResponse
from magic_llm.model.ModelChatStream import UsageModel


def test_usage_model_additive_fields_have_backward_compatible_defaults():
    usage = UsageModel()

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.cached_tokens_write == 0
    assert usage.provider_request_id is None
    assert usage.service_tier is None
    assert usage.usage_source == "provider"
    assert usage.provider_extra is None


def test_usage_model_accepts_additive_accounting_fields():
    usage = UsageModel(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        cached_tokens_write=3,
        provider_request_id="chatcmpl_123",
        service_tier="default",
        usage_source="provider",
        provider_extra={"provider": "openai"},
    )

    assert usage.cached_tokens_write == 3
    assert usage.provider_request_id == "chatcmpl_123"
    assert usage.service_tier == "default"
    assert usage.usage_source == "provider"
    assert usage.provider_extra == {"provider": "openai"}


def test_usage_factory_derives_total_when_provider_total_is_missing_or_zero():
    assert build_usage_model(prompt_tokens=7, completion_tokens=5).total_tokens == 12
    assert build_usage_model(prompt_tokens=7, completion_tokens=5, total_tokens=0).total_tokens == 12


def test_model_chat_response_preserves_usage_accounting_fields():
    response = ModelChatResponse(
        id="chatcmpl_123",
        object="chat.completion",
        created=1700000000.0,
        model="gpt-test",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=UsageModel(
            prompt_tokens=11,
            completion_tokens=2,
            total_tokens=13,
            cached_tokens_write=1,
            provider_request_id="chatcmpl_123",
            service_tier="default",
            provider_extra={"region": "test"},
        ),
    )

    assert response.usage.provider_request_id == "chatcmpl_123"
    assert response.usage.cached_tokens_write == 1
    assert response.usage.service_tier == "default"
    assert response.usage.provider_extra == {"region": "test"}

