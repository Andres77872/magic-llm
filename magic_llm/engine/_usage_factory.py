from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from magic_llm.model.ModelChatStream import (
    CompletionsTokensDetailsModel,
    PromptTokensDetailsModel,
    UsageModel,
)


_SENSITIVE_PROVIDER_EXTRA_KEYS = {
    'prompt',
    'content',
    'completion',
    'messages',
    'message',
    'input',
    'output',
    'text',
    'raw',
    'raw_request',
    'raw_response',
    'request',
    'response',
    'headers',
    'authorization',
    'api_key',
    'token',
    'secret',
}


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _merge_provider_extra(*extras: Mapping[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for extra in extras:
        if extra:
            merged.update(dict(extra))
    return sanitize_provider_extra(merged)


def sanitize_provider_extra(provider_extra: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep provider accounting metadata but drop raw prompts, content, headers, and secrets."""
    if not provider_extra:
        return None

    sanitized: dict[str, Any] = {}
    for key, value in provider_extra.items():
        normalized_key = str(key).lower()
        if normalized_key in _SENSITIVE_PROVIDER_EXTRA_KEYS:
            continue
        if any(token in normalized_key for token in ('authorization', 'api_key', 'secret')):
            continue
        if isinstance(value, Mapping):
            nested = sanitize_provider_extra(value)
            if nested:
                sanitized[key] = nested
            continue
        if isinstance(value, list):
            safe_items: list[Any] = []
            for item in value:
                if isinstance(item, Mapping):
                    nested = sanitize_provider_extra(item)
                    if nested:
                        safe_items.append(nested)
                elif _is_safe_scalar(item):
                    safe_items.append(item)
            if safe_items:
                sanitized[key] = safe_items
            continue
        if _is_safe_scalar(value):
            sanitized[key] = value

    return sanitized or None


def _is_safe_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def build_usage_model(
    *,
    prompt_tokens: Any = 0,
    completion_tokens: Any = 0,
    total_tokens: Any = None,
    cached_tokens_read: Any = 0,
    cached_tokens_write: Any = 0,
    reasoning_tokens: Any = 0,
    audio_tokens: Any = 0,
    prompt_audio_tokens: Any = 0,
    accepted_prediction_tokens: Any = 0,
    rejected_prediction_tokens: Any = 0,
    provider_request_id: str | None = None,
    service_tier: str | None = None,
    usage_source: str | None = 'provider',
    provider_extra: Mapping[str, Any] | None = None,
    attempt_index: int | None = None,
    attempt_status: str | None = None,
) -> UsageModel:
    prompt = _to_int(prompt_tokens)
    completion = _to_int(completion_tokens)
    total = _to_int(total_tokens, default=0)
    if total == 0 and (prompt or completion):
        total = prompt + completion

    return UsageModel(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=PromptTokensDetailsModel(
            cached_tokens=_to_int(cached_tokens_read),
            audio_tokens=_to_int(prompt_audio_tokens),
        ),
        completion_tokens_details=CompletionsTokensDetailsModel(
            reasoning_tokens=_to_int(reasoning_tokens),
            audio_tokens=_to_int(audio_tokens),
            accepted_prediction_tokens=_to_int(accepted_prediction_tokens),
            rejected_prediction_tokens=_to_int(rejected_prediction_tokens),
        ),
        cached_tokens_write=_to_int(cached_tokens_write),
        provider_request_id=provider_request_id,
        service_tier=service_tier,
        usage_source=usage_source or 'provider',
        provider_extra=sanitize_provider_extra(provider_extra),
        attempt_index=attempt_index,
        attempt_status=attempt_status,
    )


def usage_from_openai_payload(payload: Mapping[str, Any]) -> UsageModel:
    usage = payload.get('usage') or {}
    prompt_details = usage.get('prompt_tokens_details') or {}
    completion_details = usage.get('completion_tokens_details') or {}
    provider_extra = usage.get('provider_extra') or {}

    return build_usage_model(
        prompt_tokens=usage.get('prompt_tokens', 0),
        completion_tokens=usage.get('completion_tokens', 0),
        total_tokens=usage.get('total_tokens'),
        cached_tokens_read=prompt_details.get('cached_tokens', usage.get('cached_tokens', 0)),
        cached_tokens_write=usage.get('cached_tokens_write', 0),
        reasoning_tokens=completion_details.get('reasoning_tokens', usage.get('reasoning_tokens', 0)),
        audio_tokens=completion_details.get('audio_tokens', usage.get('audio_tokens', 0)),
        prompt_audio_tokens=prompt_details.get('audio_tokens', 0),
        accepted_prediction_tokens=completion_details.get(
            'accepted_prediction_tokens',
            usage.get('accepted_prediction_tokens', 0),
        ),
        rejected_prediction_tokens=completion_details.get(
            'rejected_prediction_tokens',
            usage.get('rejected_prediction_tokens', 0),
        ),
        provider_request_id=usage.get('provider_request_id') or payload.get('id'),
        service_tier=usage.get('service_tier') or payload.get('service_tier'),
        usage_source=usage.get('usage_source', 'provider'),
        provider_extra=provider_extra,
    )


def usage_from_anthropic_usage(
    usage_meta: Mapping[str, Any],
    *,
    provider_request_id: str | None = None,
    service_tier: str | None = None,
) -> UsageModel:
    cache_read = _to_int(usage_meta.get('cache_read_input_tokens', 0))
    cache_creation = _to_int(usage_meta.get('cache_creation_input_tokens', 0))
    input_tokens = _to_int(usage_meta.get('input_tokens', 0))
    output_tokens = _to_int(usage_meta.get('output_tokens', 0))

    return build_usage_model(
        prompt_tokens=input_tokens + cache_read + cache_creation,
        completion_tokens=output_tokens,
        total_tokens=usage_meta.get('total_tokens'),
        cached_tokens_read=cache_read,
        cached_tokens_write=cache_creation,
        provider_request_id=provider_request_id,
        service_tier=service_tier or usage_meta.get('service_tier'),
        provider_extra={
            'anthropic_input_tokens': input_tokens,
        },
    )


def usage_from_google_usage(
    usage_metadata: Mapping[str, Any],
    *,
    provider_request_id: str | None = None,
) -> UsageModel:
    tool_use_prompt_tokens = _to_int(usage_metadata.get('toolUsePromptTokenCount', 0))
    provider_extra = {'tool_use_prompt_tokens': tool_use_prompt_tokens} if tool_use_prompt_tokens else None

    return build_usage_model(
        prompt_tokens=usage_metadata.get('promptTokenCount', 0),
        completion_tokens=usage_metadata.get('candidatesTokenCount', 0),
        total_tokens=usage_metadata.get('totalTokenCount'),
        cached_tokens_read=usage_metadata.get('cachedContentTokenCount', 0),
        reasoning_tokens=usage_metadata.get('thoughtsTokenCount', 0),
        provider_request_id=provider_request_id,
        provider_extra=provider_extra,
    )


def usage_from_cohere_meta(
    meta: Mapping[str, Any],
    *,
    provider_request_id: str | None = None,
) -> UsageModel:
    tokens = meta.get('tokens') or meta.get('billed_units') or {}
    basis = 'tokens' if meta.get('tokens') else 'billed_units'

    return build_usage_model(
        prompt_tokens=tokens.get('input_tokens', 0),
        completion_tokens=tokens.get('output_tokens', 0),
        total_tokens=tokens.get('total_tokens'),
        provider_request_id=provider_request_id,
        provider_extra={'cohere_usage_basis': basis},
    )


def usage_from_bedrock_invocation_metrics(
    metrics: Mapping[str, Any],
    *,
    provider_request_id: str | None = None,
) -> UsageModel:
    provider_extra: dict[str, Any] = {}
    latency = _first_present(metrics, 'invocationLatency', 'invocationLatencyMs')
    if latency is not None:
        provider_extra['invocation_latency_ms'] = latency
    first_byte_latency = _first_present(metrics, 'firstByteLatency', 'firstByteLatencyMs')
    if first_byte_latency is not None:
        provider_extra['first_byte_latency_ms'] = first_byte_latency

    return build_usage_model(
        prompt_tokens=metrics.get('inputTokenCount', 0),
        completion_tokens=metrics.get('outputTokenCount', 0),
        total_tokens=metrics.get('totalTokenCount'),
        provider_request_id=provider_request_id,
        provider_extra=provider_extra,
    )


def estimated_usage_model(
    *,
    prompt_tokens: Any = 0,
    completion_tokens: Any = 0,
    total_tokens: Any = None,
    provider_request_id: str | None = None,
    estimation_method: str,
    provider_extra: Mapping[str, Any] | None = None,
) -> UsageModel:
    return build_usage_model(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        provider_request_id=provider_request_id,
        usage_source='estimated',
        provider_extra=_merge_provider_extra(
            {'estimated': True, 'estimation_method': estimation_method},
            provider_extra,
        ),
    )


def usage_has_tokens(usage: UsageModel | None) -> bool:
    if usage is None:
        return False
    return any(
        _to_int(value) > 0
        for value in (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.cached_tokens_write,
            getattr(usage.prompt_tokens_details, 'cached_tokens', 0) if usage.prompt_tokens_details else 0,
            getattr(usage.completion_tokens_details, 'reasoning_tokens', 0) if usage.completion_tokens_details else 0,
        )
    )


def usage_attempt_dict(
    usage: UsageModel,
    *,
    attempt_index: int,
    status: str,
) -> dict[str, Any]:
    details = usage.completion_tokens_details or CompletionsTokensDetailsModel()
    prompt_details = usage.prompt_tokens_details or PromptTokensDetailsModel()
    return {
        'attempt_index': attempt_index,
        'status': status,
        'provider_request_id': usage.provider_request_id,
        'usage_source': usage.usage_source or 'provider',
        'prompt_tokens': _to_int(usage.prompt_tokens),
        'completion_tokens': _to_int(usage.completion_tokens),
        'total_tokens': _to_int(usage.total_tokens) or _to_int(usage.prompt_tokens) + _to_int(usage.completion_tokens),
        'cached_tokens_read': _to_int(prompt_details.cached_tokens),
        'cached_tokens_write': _to_int(usage.cached_tokens_write),
        'reasoning_tokens': _to_int(details.reasoning_tokens),
        'audio_tokens': _to_int(details.audio_tokens),
        'accepted_prediction_tokens': _to_int(details.accepted_prediction_tokens),
        'rejected_prediction_tokens': _to_int(details.rejected_prediction_tokens),
    }


def attach_attempt_metadata(
    usage: UsageModel | None,
    attempts: list[dict[str, Any]],
) -> UsageModel | None:
    if usage is None or not attempts:
        return usage
    cloned = usage.model_copy(deep=True)
    existing_extra = deepcopy(cloned.provider_extra) if cloned.provider_extra else {}
    existing_attempts = existing_extra.get('attempts') or []
    existing_extra['attempts'] = existing_attempts + attempts
    cloned.provider_extra = sanitize_provider_extra(existing_extra)
    return cloned
