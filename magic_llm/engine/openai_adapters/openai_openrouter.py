"""OpenRouter's Chat Completions wire format.

Read the way OpenRouter's API reference describes its stream
(https://openrouter.ai/docs/api/reference/streaming):

* ``: OPENROUTER PROCESSING`` comment lines keep the connection alive and
  carry nothing.
* Usage is sent in the stream, in a last chunk whose single choice has an
  empty delta (or no choice). It reports cache reads and writes under
  ``prompt_tokens_details`` and the charged ``cost``; nothing is polled
  afterwards.
* A failure after the HTTP 200 arrives as a top-level ``error`` or a choice
  with ``finish_reason: "error"``. It fails the call with an ``HttpError``
  carrying the status and the typed ``error_type``, never ends it quietly.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model import ModelChatResponse
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel,
    CompletionsTokensDetailsModel,
    PromptTokensDetailsModel,
    UsageModel,
)
from magic_llm.util.http import HttpError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: OpenRouter's typed ``error.metadata.error_type`` values and the HTTP status
#: it documents for each. A mid-stream error carries the type more reliably
#: than a numeric code.
ERROR_TYPE_STATUS = {
    "authentication": 401,
    "payment_required": 402,
    "permission_denied": 403,
    "content_policy_violation": 403,
    "refusal": 403,
    "not_found": 404,
    "precondition_failed": 412,
    "payload_too_large": 413,
    "unprocessable": 422,
    "rate_limit_exceeded": 429,
    "context_length_exceeded": 400,
    "max_tokens_exceeded": 400,
    "token_limit_exceeded": 400,
    "string_too_long": 400,
    "invalid_request": 400,
    "invalid_prompt": 400,
    "provider_unavailable": 502,
    "provider_overloaded": 503,
    "timeout": 504,
    "server": 500,
    "unmapped": 500,
}


def error_from_payload(error: Any) -> HttpError:
    """The typed failure for an OpenRouter ``error`` object.

    Keeps only the status and the error type; the message and metadata can
    echo prompt text and are dropped.
    """
    status: Optional[int] = None
    error_type: Optional[str] = None
    if isinstance(error, Mapping):
        metadata = error.get("metadata")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("error_type"), str):
            error_type = metadata["error_type"]
        code = error.get("code")
        if isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
            status = code
        elif isinstance(code, str) and error_type is None:
            error_type = code
    if status is None and error_type is not None:
        status = ERROR_TYPE_STATUS.get(error_type)
    return HttpError("Provider stream failed", status_code=status, error_type=error_type)


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _cost(usage: Mapping[str, Any]) -> Optional[float]:
    """What the call cost in credits, whoever billed it.

    With a provider key of the caller's own (BYOK), OpenRouter's ``cost`` is
    only its fee, and the provider's charge is reported separately.
    """
    cost = _number(usage.get("cost"))
    details = usage.get("cost_details")
    upstream = (
        _number(details.get("upstream_inference_cost"))
        if usage.get("is_byok") is True and isinstance(details, Mapping)
        else None
    )
    if cost is None and upstream is None:
        return None
    return (cost or 0.0) + (upstream or 0.0)


def usage_from_payload(usage: Any, request_id: Optional[str] = None) -> Optional[UsageModel]:
    """OpenRouter usage with cache reads, cache writes and cost (``provider_extra["cost"]``)."""
    if not isinstance(usage, Mapping):
        return None
    prompt_details = usage.get("prompt_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, Mapping) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, Mapping) else {}
    prompt_tokens = _int(usage.get("prompt_tokens"))
    completion_tokens = _int(usage.get("completion_tokens"))
    cost = _cost(usage)
    return UsageModel(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsModel(
            cached_tokens=_int(prompt_details.get("cached_tokens")),
            audio_tokens=_int(prompt_details.get("audio_tokens")),
        ),
        completion_tokens_details=CompletionsTokensDetailsModel(
            reasoning_tokens=_int(completion_details.get("reasoning_tokens")),
            audio_tokens=_int(completion_details.get("audio_tokens")),
        ),
        cached_tokens_write=_int(prompt_details.get("cache_write_tokens")),
        provider_request_id=request_id,
        provider_extra={"cost": cost} if cost is not None else None,
    )


def raise_for_error(payload: Mapping[str, Any]) -> None:
    """Raise the typed failure a 200 body or stream frame reports, if any."""
    if payload.get("error"):
        raise error_from_payload(payload["error"])
    for choice in payload.get("choices") or ():
        if not isinstance(choice, Mapping):
            continue
        if choice.get("error") or choice.get("finish_reason") == "error":
            raise error_from_payload(choice.get("error"))


class ProviderOpenRouter(OpenAiBaseProvider):
    supports_vision = True

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", OPENROUTER_BASE_URL)
        super().__init__(**kwargs)

    def process_chunk(
        self,
        chunk: Any,
        id_generation: str = "",
        last_chunk: Optional[ChatCompletionModel] = None,
    ) -> Optional[ChatCompletionModel]:
        line = chunk.strip() if isinstance(chunk, str) else ""
        # Comments (keep-alives) and SSE fields other than ``data`` carry no
        # completion data.
        if not line.startswith("data:"):
            return None
        body = line[5:].strip()
        if not body or body == "[DONE]":
            return None
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Provider stream sent a non-object event")
        raise_for_error(payload)

        request_id = payload.get("id") or id_generation or None
        usage = usage_from_payload(payload.get("usage"), request_id)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            if usage is None:
                return None
            # An accounting frame without a choice: keep the usage visible
            # without replaying the previous delta.
            previous = (
                last_chunk.choices[0]
                if isinstance(last_chunk, ChatCompletionModel) and last_chunk.choices
                else None
            )
            choices = [{
                "index": previous.index if previous else 0,
                "delta": {"content": ""},
                "finish_reason": previous.finish_reason if previous else None,
            }]
        return ChatCompletionModel(
            id=str(payload.get("id") or id_generation or "openrouter"),
            created=payload.get("created"),
            model=str(payload.get("model") or self.model or "unknown"),
            choices=choices,
            usage=usage if usage is not None else UsageModel(),
        )

    def transform_response(self, raw: dict[str, Any]) -> ModelChatResponse:
        raise_for_error(raw)
        if not raw.get("choices"):
            # A 200 with neither an error nor a choice: the provider produced
            # nothing (cold start, scaling). OpenRouter documents a retry.
            raise HttpError("Provider returned no choice", error_type="no_content")
        normalized = dict(raw)
        usage = usage_from_payload(normalized.get("usage"), normalized.get("id"))
        if usage is not None:
            normalized["usage"] = usage
        else:
            normalized.pop("usage", None)
        return ModelChatResponse(**normalized)


__all__ = [
    "ERROR_TYPE_STATUS",
    "OPENROUTER_BASE_URL",
    "ProviderOpenRouter",
    "error_from_payload",
    "raise_for_error",
    "usage_from_payload",
]
